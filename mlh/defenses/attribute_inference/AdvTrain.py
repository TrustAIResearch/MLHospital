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


class TrainTargetAdvTrain(Trainer):
    def __init__(self, target_model, alpha, device="cuda:0", emd_dim=512, tar_att='Young', sen_att='Male', num_class=2,
                 epochs=100, learning_rate=0.01, momentum=0.9, weight_decay=5e-4, log_path="./"):

        super().__init__()

        # self.opt = opt
        self.target_model = target_model
        self.alpha = alpha
        self.device = device
        self.emd_dim = emd_dim
        self.tar_att = tar_att
        self.sen_att = sen_att
        self.num_class = num_class
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.attack_model = AttackAdvTrain(self.emd_dim, self.num_class)
        self.target_model_E = copy.deepcopy(self.target_model)
        self.target_model_E.fc = nn.Sequential()
        self.target_model_C = copy.deepcopy(self.target_model.fc)
        self.target_model_E.to(self.device)
        self.target_model_C.to(self.device)
        self.attack_model.to(self.device)

        self.optimizer_E = torch.optim.SGD(self.target_model_E.parameters(
        ), learning_rate, momentum=momentum, weight_decay=self.weight_decay)
        self.scheduler_E = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer_E, T_max=self.epochs)

        self.optimizer_C = torch.optim.SGD(self.target_model_C.parameters(
        ), learning_rate, momentum=momentum, weight_decay=self.weight_decay)
        self.scheduler_C = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer_C, T_max=self.epochs)

        self.optimizer_adv = torch.optim.SGD(self.attack_model.parameters(
        ), learning_rate, momentum=momentum, weight_decay=weight_decay)
        self.scheduler_adv = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer_adv, T_max=self.epochs)

        self.criterion = torch.nn.CrossEntropyLoss()

        self.log_path = log_path
        logx.initialize(logdir=self.log_path,
                        coolname=False, tensorboard=False)

    def eval(self, data_laoder):

        correct = 0
        total = 0
        self.target_model_E.eval()
        self.target_model_C.eval()
        with torch.no_grad():

            for img, targets in data_laoder:
                labels = targets[self.tar_att]
                img, labels = img.to(self.device), labels.to(self.device)
                emb = self.target_model_E(img)
                logits = self.target_model_C(emb)

                predicted = torch.argmax(logits, dim=1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            final_acc = 100 * correct / total

        return final_acc

    def eval_attack(self, data_laoder):

        correct = 0
        total = 0
        self.target_model_E.eval()
        self.attack_model.eval()
        with torch.no_grad():

            for img, targets in data_laoder:
                labels = targets[self.sen_att]
                img, labels = img.to(self.device), labels.to(self.device)
                emb = self.target_model_E(img)
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
        self.target_model_E.eval()
        self.target_model_C.eval()
        self.attack_model.train()

        for batch_idx, (img, targets) in tqdm(enumerate(train_loader)):
            train_target = targets[self.sen_att]
            train_data, train_target = img.to(
                self.device), train_target.to(self.device)
            emb = self.target_model_E(train_data)

            attack_outputs = self.attack_model(emb)
            loss = self.criterion(attack_outputs, train_target)

            self.optimizer_adv.zero_grad()
            loss.backward()
            self.optimizer_adv.step()

    def train_target_privately(self, train_loader):
        self.target_model_E.train()
        self.target_model_C.train()
        self.attack_model.eval()

        for batch_idx, (img, targets) in tqdm(enumerate(train_loader)):
            tar_target = targets[self.tar_att]
            sen_target = targets[self.sen_att]
            data, tar_target, sen_target = img.to(self.device), tar_target.to(
                self.device), sen_target.to(self.device)

            emb = self.target_model_E(data)
            attack_output = self.attack_model(emb)
            target_output = self.target_model_C(emb)

            self.optimizer_C.zero_grad()
            self.optimizer_E.zero_grad()
            # Random Attack Target
            # sen_target = torch.tensor(np.random.randint(2, size=sen_target.shape[0])).to(self.device)
            # # Fixed Attack Target
            sen_target = torch.tensor(
                [1] * sen_target.shape[0]).to(self.device)
            # Design the loss function with bias
            loss = self.criterion(target_output, tar_target) + \
                self.alpha * self.criterion(attack_output, sen_target)
            # loss = self.criterion(target_output, tar_target) + (self.alpha) * (torch.mean((attack_output)) - 0.5)

            loss.backward()
            self.optimizer_E.step()
            self.optimizer_C.step()

    def train(self, train_loader, test_loader):

        best_accuracy = 0
        t_start = time.time()
        # check whether path exist
        if not os.path.exists(self.log_path):
            os.makedirs(self.log_path)

        # torch.save(self.model.state_dict(), os.path.join(
        #     self.log_path, '%s_0.pth' % (self.model_save_name)))

        # first train target model for 5 epochs
        for e in range(1, self.epochs+1):
            print("Epoch: " + str(e))

            if e <= 5:
                self.train_target_privately(train_loader)
            else:
                self.train_attack_advtrain(train_loader)
                self.train_target_privately(train_loader)

            if e % 10 == 0:
                train_acc = self.eval(train_loader)
                test_acc = self.eval(test_loader)
                attack_train_acc = self.eval_attack(train_loader)
                attack_test_acc = self.eval_attack(test_loader)

                logx.msg('Train Epoch: %d, Total Sample: %d, Train Acc: %.3f, Test Acc: %.3f, Train Attack Acc: %.3f, Test Attack Acc: %.3f Total Time: %.3fs' % (
                    e, len(train_loader.dataset), train_acc, test_acc, attack_train_acc, attack_test_acc, time.time() - t_start))
            self.scheduler_E.step()
            self.scheduler_C.step()
            self.scheduler_adv.step()
