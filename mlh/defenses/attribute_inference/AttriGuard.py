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


class TrainTargetAttriGuard(Trainer):
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
        self.target_model_E_fixed = copy.deepcopy(self.target_model)
        self.target_model_E_fixed.fc = nn.Sequential()
        self.target_model_C_fixed = copy.deepcopy(self.target_model.fc)
        self.target_model_E_fixed.to(self.device)
        self.target_model_C_fixed.to(self.device)
        self.attack_model.to(self.device)

        self.optimizer_adv = torch.optim.SGD(self.attack_model.parameters(
        ), learning_rate, momentum=momentum, weight_decay=weight_decay)
        self.scheduler_adv = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer_adv, T_max=self.epochs)

        self.criterion = torch.nn.CrossEntropyLoss()
        self.KLD = torch.nn.KLDivLoss()

        self.log_path = log_path
        logx.initialize(logdir=self.log_path,
                        coolname=False, tensorboard=False)

    def eval_embs(self, test_loader):

        correct = 0
        test_loss = 0
        self.target_model_C_fixed.eval()

        with torch.no_grad():
            for emb, target in test_loader:
                print(emb.shape, target.shape)
                emb, target = emb.to(self.device), target.to(self.device)
                logits = self.target_model_C_fixed(emb)
                test_loss += self.criterion(logits, target).item()
                pred = logits.max(1, keepdim=True)[1]
                correct += pred.eq(target.view_as(pred)).sum().item()
        test_loss /= len(test_loader.dataset)

        accuracy = 100. * correct / len(test_loader.dataset)
        # logx.msg('Model Test: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        #     test_loss, correct, len(test_loader.dataset), accuracy))

        return accuracy

    def eval_attack(self, data_laoder):

        correct = 0
        total = 0
        self.target_model_E_fixed.eval()
        self.attack_model.eval()

        with torch.no_grad():
            for img, targets in data_laoder:
                labels = targets[self.sen_att]
                img, labels = img.to(self.device), labels.to(self.device)
                emb = self.target_model_E_fixed(img)
                logits = self.attack_model(emb)

                predicted = torch.argmax(logits, dim=1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            final_acc = 100 * correct / total

        return final_acc

    def eval_attack_embs(self, data_laoder):

        correct = 0
        total = 0
        self.attack_model.eval()

        with torch.no_grad():
            for emb, labels in data_laoder:
                emb, labels = emb.to(self.device), labels.to(self.device)
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
        self.attack_model.train()

        for batch_idx, (img, targets) in tqdm(enumerate(train_loader)):
            train_target = targets[self.sen_att]
            train_data, train_target = img.to(
                self.device), train_target.to(self.device)
            emb = self.target_model_E_fixed(train_data)
            attack_outputs = self.attack_model(emb)
            loss = self.criterion(attack_outputs, train_target)

            self.optimizer_adv.zero_grad()
            loss.backward()
            self.optimizer_adv.step()

    class GetEmbsDataset(torch.utils.data.Dataset):
        def __init__(self, data_root, data_label):
            self.data = data_root
            self.label = data_label

        def __getitem__(self, index):
            data = self.data[index]
            labels = self.label[index]
            return data, labels

        def __len__(self):
            return len(self.data)

    def optimize_embs_privately(self, test_loader):
        self.target_model_E_fixed.eval()
        self.attack_model.eval()

        for batch_idx, (img, targets) in tqdm(enumerate(test_loader)):
            tar_target = targets[self.tar_att]
            sen_target = targets[self.sen_att]
            data, tar_target, sen_target = img.to(self.device), tar_target.to(
                self.device), sen_target.to(self.device)

            emb_fixed = self.target_model_E_fixed(data)
            emb_noise = emb_fixed.clone().detach()
            emb_noise.requires_grad = True
            optimizer_noise = torch.optim.Adam([emb_noise], lr=0.001)

            uniform_target = torch.tensor([[1 / self.num_class] * self.num_class] * sen_target.shape[0]).to(
                self.device)

            for iter in range(100):
                attack_output = self.attack_model(emb_noise)
                # We take both l0 and l2 norm
                norm_term = (torch.norm(emb_noise - emb_fixed, p=0) + torch.norm(emb_noise - emb_fixed, p=2)) / \
                    tar_target.shape[0]  # keep utility
                adv_term = self.KLD(torch.nn.functional.log_softmax(
                    attack_output, dim=1), uniform_target)  # against attacker
                loss = 0.5 * norm_term + self.alpha * adv_term

                if (iter + 1) % 50 == 0:
                    logx.msg('Optimize Embeddings iter: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        iter + 1,
                        batch_idx * len(data),
                        len(test_loader.dataset),
                        100. * batch_idx / len(test_loader),
                        loss.item()))

                optimizer_noise.zero_grad()
                loss.backward(retain_graph=True)
                optimizer_noise.step()

            emb_noise = emb_noise.cpu().detach().numpy()
            emb_tar_target = tar_target.cpu().numpy()
            emb_sen_target = sen_target.cpu().numpy()
            if batch_idx == 0:
                self.embs_noise = emb_noise
                self.embs_tar_target = emb_tar_target
                self.embs_sen_target = emb_sen_target
            else:
                self.embs_noise = np.vstack([self.embs_noise, emb_noise])
                self.embs_tar_target = np.hstack(
                    [self.embs_tar_target, emb_tar_target])
                self.embs_sen_target = np.hstack(
                    [self.embs_sen_target, emb_sen_target])
            # print(self.embs_tar_target.shape)

            self.emb_tar_dataset = self.GetEmbsDataset(
                self.embs_noise, self.embs_tar_target)
            self.emb_tar_dataloader = torch.utils.data.DataLoader(self.emb_tar_dataset, batch_size=512,
                                                                  shuffle=False, drop_last=True, num_workers=1)
            self.emb_sen_dataset = self.GetEmbsDataset(
                self.embs_noise, self.embs_sen_target)
            self.emb_sen_dataloader = torch.utils.data.DataLoader(self.emb_sen_dataset, batch_size=512,
                                                                  shuffle=False, drop_last=True, num_workers=1)

    def train(self, infer_loader, test_loader, real_attack=False, attack_loader=None):

        t_start = time.time()
        # check whether path exist
        if not os.path.exists(self.log_path):
            os.makedirs(self.log_path)

        # First train an attack classifier based on defender's infer dataset and fix it
        for e in range(1, self.epochs + 1):
            print("Epoch: " + str(e))
            self.train_attack_advtrain(infer_loader)
            if e % 10 == 0:
                attack_test_acc = self.eval_attack(test_loader)
                logx.msg('Train Epoch: %d, Total Sample: %d, Adv Test Attack Acc: %.4f Total Time: %.3fs' % (
                    e, len(infer_loader.dataset), attack_test_acc, time.time() - t_start))

        # Then optimize noise on test dataset to obtain emb_noise datasets
        for e in range(1, 1 + 1):
            print("Epoch: " + str(e))
            self.optimize_embs_privately(test_loader)
            test_acc = self.eval_embs(self.emb_tar_dataloader)
            attack_test_acc = self.eval_attack_embs(self.emb_sen_dataloader)
            logx.msg(
                'Train Epoch: %d, Total Sample: %d, Test Acc: %.3f, Adv Test Attack Acc: %.3f Total Time: %.3fs' % (
                    e, len(test_loader.dataset), test_acc, attack_test_acc, time.time() - t_start))

        if real_attack == True:
            # Train an attack classifier based on attacker's train dataset and fix it
            for e in range(1, self.epochs + 1):
                print("Epoch: " + str(e))
                self.train_attack_advtrain(attack_loader)
                if e % 10 == 0:
                    attack_test_acc = self.eval_attack(test_loader)
                    logx.msg('Train Epoch: %d, Total Sample: %d, Real Test Attack Acc: %.4f Total Time: %.3fs' % (
                        e, len(attack_loader.dataset), attack_test_acc, time.time() - t_start))
            # Evaluate attack performance on emb_noise datasets
            test_acc = self.eval_embs(self.emb_tar_dataloader)
            attack_test_acc = self.eval_attack_embs(self.emb_sen_dataloader)
            logx.msg('Total Sample: %d, Test Acc: %.3f, Real Test Attack Acc: %.3f Total Time: %.3fs' % (
                len(test_loader.dataset), test_acc, attack_test_acc, time.time() - t_start))
        self.scheduler_adv.step()
