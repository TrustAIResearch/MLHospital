# MIT License

# Copyright (c) 2022 The Machine Learning Hospital (MLH) Authors

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import torch
import numpy as np
import os
import time
from runx.logx import logx
import torch.nn as nn
from mlh.defenses.membership_inference.trainer import Trainer
import torch.nn.functional as F
from tqdm import tqdm
import copy

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


class FeatureExtractor(nn.Module):
    def __init__(self, model, layers):
        super().__init__()
        self.model = model
        self.layers = layers
        self._features = {layer: torch.empty(0) for layer in layers}

        for layer_id in layers:
            layer = dict([*self.model.named_modules()])[layer_id]
            layer.register_forward_hook(self.save_outputs_hook(layer_id))

    def save_outputs_hook(self, layer_id):
        def fn(_, __, output):
            self._features[layer_id] = output
        return fn

    def forward(self, x):
        _ = self.model(x)
        return self._features


class AttackAdvTrain(nn.Module):
    def __init__(self, dim_in, dim_out):
        super(AttackAdvTrain, self).__init__()
        self.dim_in = dim_in
        # self.bn1 = nn.BatchNorm1d(self.dim_in)
        self.fc1 = nn.Linear(self.dim_in, 64)
        self.fc2 = nn.Linear(64, 32)
        # self.fc3 = nn.Linear(10, 10)
        self.fc4 = nn.Linear(32, dim_out)

    def forward(self, x):
        x = x.view(-1, self.dim_in)
        # x = self.bn1(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        # x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x


class AutoEncoder(nn.Module):
    """Simple multi-layer autoencoder structure for Olympus"""

    def __init__(self, input_dim, latent_dim=16):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.Linear(512, 64),
            nn.Linear(64, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.Linear(64, 512),
            nn.Linear(512, input_dim)
        )

        self.encoder.apply(self._init_weights)
        self.decoder.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            module.bias.data.fill_(0.01)

    def forward(self, x):
        latent = self.encoder(x)
        out = self.decoder(latent)
        return out


class TrainTargetOlympus(Trainer):
    def __init__(self, target_model, alpha, device="cuda:0", emb_dim=512, tar_att='Young', sen_att='Male', num_class=2,
                 epochs=100, learning_rate=0.01, momentum=0.9, weight_decay=5e-4, log_path="./"):

        super().__init__()

        # self.opt = opt
        self.target_model = target_model
        self.alpha = alpha
        self.device = device
        self.emb_dim = emb_dim
        self.tar_att = tar_att
        self.sen_att = sen_att
        self.num_class = num_class
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.weight_decay = weight_decay

        self.attack_model = AttackAdvTrain(self.emb_dim, self.num_class)
        self.obfuscator = AutoEncoder(self.emb_dim, latent_dim=16)
        self.target_model_E_fixed = copy.deepcopy(self.target_model)
        self.target_model_E_fixed.fc = nn.Sequential()
        self.target_model_C_fixed = copy.deepcopy(self.target_model.fc)
        self.target_model_E_fixed.to(self.device)
        self.target_model_C_fixed.to(self.device)
        self.attack_model.to(self.device)
        self.obfuscator.to(self.device)

        self.optimizer_adv = torch.optim.SGD(self.attack_model.parameters(
        ), learning_rate, momentum=momentum, weight_decay=weight_decay)
        self.scheduler_adv = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer_adv, T_max=self.epochs)

        self.optimizer_obfs = torch.optim.SGD(self.obfuscator.parameters(
        ), learning_rate, momentum=momentum, weight_decay=weight_decay)
        self.scheduler_obfs = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer_obfs, T_max=self.epochs)

        self.criterion = torch.nn.CrossEntropyLoss()

        self.log_path = log_path
        logx.initialize(logdir=self.log_path,
                        coolname=False, tensorboard=False)

    def SoftCrossEntropy(self, inputs, target, reduction='average'):
        log_likelihood = -torch.nn.functional.log_softmax(inputs, dim=1)
        batch = inputs.shape[0]
        if reduction == 'average':
            loss = torch.sum(torch.mul(log_likelihood, target)) / batch
        elif reduction == 'sum':
            loss = torch.sum(torch.mul(log_likelihood, target))
        return loss

    def eval(self, data_laoder):

        correct = 0
        total = 0
        self.target_model_E_fixed.eval()
        self.obfuscator.eval()
        self.target_model_C_fixed.eval()

        with torch.no_grad():
            for img, targets in data_laoder:
                labels = targets[self.tar_att]
                img, labels = img.to(self.device), labels.to(self.device)
                emb = self.target_model_E_fixed(img)
                emb = self.obfuscator(emb)
                logits = self.target_model_C_fixed(emb)

                predicted = torch.argmax(logits, dim=1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            final_acc = 100 * correct / total
        return final_acc

    def eval_attack(self, data_laoder):

        correct = 0
        total = 0
        self.target_model_E_fixed.eval()
        self.obfuscator.eval()
        self.attack_model.eval()

        with torch.no_grad():
            for img, targets in data_laoder:
                labels = targets[self.sen_att]
                img, labels = img.to(self.device), labels.to(self.device)
                emb = self.target_model_E_fixed(img)
                emb = self.obfuscator(emb)
                logits = self.attack_model(emb)

                predicted = torch.argmax(logits, dim=1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            final_acc = 100 * correct / total

        return final_acc

    def train_attack_advtrain(self, train_loader):
        """
        train the aia classifier to infer sensitive attribute from embeddings
        """
        self.target_model_E_fixed.eval()
        self.obfuscator.eval()
        self.attack_model.train()

        for batch_idx, (img, targets) in tqdm(enumerate(train_loader)):
            train_target = targets[self.sen_att]
            train_data, train_target = img.to(
                self.device), train_target.to(self.device)
            emb = self.target_model_E_fixed(train_data)
            emb = self.obfuscator(emb)
            attack_outputs = self.attack_model(emb)
            loss = self.criterion(attack_outputs, train_target)

            self.optimizer_adv.zero_grad()
            loss.backward()
            self.optimizer_adv.step()

    def train_obfuscator_privately(self, train_loader):
        self.target_model_E_fixed.eval()
        self.target_model_C_fixed.eval()
        self.attack_model.eval()
        self.obfuscator.train()

        for batch_idx, (img, targets) in tqdm(enumerate(train_loader)):
            tar_target = targets[self.tar_att]
            sen_target = targets[self.sen_att]
            data, tar_target, sen_target = img.to(self.device), tar_target.to(
                self.device), sen_target.to(self.device)

            emb_fixed = self.target_model_E_fixed(data)
            emb_obfs = self.obfuscator(emb_fixed)
            attack_output = self.attack_model(emb_obfs)
            target_output = self.target_model_C_fixed(emb_obfs)

            self.optimizer_obfs.zero_grad()

            '''Force to attack classifier yields uniform prediction on sensitive task while maintaining utility on original task'''
            uniform_target = torch.tensor([[1 / self.num_class] * self.num_class] * sen_target.shape[0]).to(
                self.device)
            loss = self.criterion(target_output, tar_target) + self.alpha * self.SoftCrossEntropy(attack_output,
                                                                                                  uniform_target)

            loss.backward()
            self.optimizer_obfs.step()

    def train(self, train_loader, test_loader):

        t_start = time.time()
        # check whether path exist
        if not os.path.exists(self.log_path):
            os.makedirs(self.log_path)

        for e in range(1, self.epochs+1):
            print("Epoch: " + str(e))
            self.train_attack_advtrain(train_loader)
            self.train_obfuscator_privately(train_loader)

            if e % 10 == 0:
                train_acc = self.eval(train_loader)
                test_acc = self.eval(test_loader)
                attack_train_acc = self.eval_attack(train_loader)
                attack_test_acc = self.eval_attack(test_loader)

                logx.msg('Train Epoch: %d, Total Sample: %d, Train Acc: %.3f, Test Acc: %.3f, Train Attack Acc: %.3f, Test Attack Acc: %.3f Total Time: %.3fs' % (
                    e, len(train_loader.dataset), train_acc, test_acc, attack_train_acc, attack_test_acc, time.time() - t_start))
            self.scheduler_obfs.step()
            self.scheduler_adv.step()
