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

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


class AttackAdvReg(nn.Module):
    def __init__(self, posterior_dim, class_dim):
        self.posterior_dim = posterior_dim
        self.class_dim = posterior_dim
        super(AttackAdvReg, self).__init__()
        self.features = nn.Sequential(
            nn.Linear(self.posterior_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 64),
            nn.ReLU(),
        )
        self.labels = nn.Sequential(
            nn.Linear(self.class_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
        )
        self.combine = nn.Sequential(
            nn.Linear(64*2, 256),
            nn.ReLU(),
            nn.Linear(256, 128),

            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

        self.output = nn.Sigmoid()

    def forward(self, x, l):
        out_x = self.features(x)
        out_l = self.labels(l)
        is_member = self.combine(torch.cat((out_x, out_l), 1))
        return self.output(is_member)


class TrainTargetAdvReg(Trainer):
    """
    We take
    https://github.com/SPIN-UMass/ML-Privacy-Regulization
    as the reference
    """

    def __init__(self, model, device="cuda:0", num_class=10, epochs=100, learning_rate=0.01, momentum=0.9, weight_decay=5e-4, log_path="./"):

        super().__init__()

        # self.opt = opt
        self.model = model
        self.device = device
        self.num_class = num_class
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.attack_model = AttackAdvReg(self.num_class, self.num_class)
        self.model.to(self.device)
        self.attack_model.to(self.device)

        self.optimizer = torch.optim.SGD(self.model.parameters(
        ), learning_rate, momentum=momentum, weight_decay=self.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=self.epochs)

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
        self.model.eval()
        with torch.no_grad():

            for img, label in data_laoder:
                img, label = img.to(self.device), label.to(self.device)
                logits = self.model.eval().forward(img)

                predicted = torch.argmax(logits, dim=1)
                total += label.size(0)
                correct += (predicted == label).sum().item()

            final_acc = 100 * correct / total

        return final_acc

    def train_attack_advreg(self, train_loader, inference_loader):
        """
        train the mia classifier to distinguish train and inference data
        """
        self.model.eval()
        self.attack_model.train()

        for batch_idx, ((train_data, train_target), (inference_data, inference_target)) in enumerate(zip(train_loader, inference_loader)):
            train_data, train_target = train_data.to(
                self.device), train_target.to(self.device)
            inference_data, inference_target = inference_data.to(
                self.device), inference_target.to(self.device)

            all_data = torch.cat([train_data, inference_data], dim=0)
            all_target = torch.cat([train_target, inference_target], dim=0)
            all_output = self.model(all_data)

            one_hot_tr = torch.from_numpy((np.zeros(
                (all_target.size(0), self.num_class))-1)).type(torch.FloatTensor).to(self.device)
            infer_input_one_hot = one_hot_tr.scatter_(1, all_target.type(
                torch.LongTensor).view([-1, 1]).data.to(self.device), 1)
            attack_output = self.attack_model(all_output, infer_input_one_hot)
            # get something like [[1], [1], [1], [0], [0], [0]]
            att_labels = torch.cat([torch.unsqueeze(torch.ones(
                train_data.shape[0]), 1), torch.unsqueeze(torch.zeros(train_data.shape[0]), 1)], dim=0).to(self.device)
            # att_labels = torch.cat([torch.ones(train_data.shape[0]), torch.zeros(
            #     train_data.shape[0])], dim=0).type(torch.LongTensor).to(self.device)
            # print(att_labels, train_target)
            loss = torch.nn.functional.binary_cross_entropy(
                attack_output, att_labels)
            self.attack_model.zero_grad()
            loss.backward()
            self.optimizer_adv.step()

    def train_target_privately(self, train_loader):
        self.model.train()
        self.attack_model.eval()
        alpha = 1

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)
            output = self.model(data)
            one_hot_tr = torch.from_numpy((np.zeros((output.size(
                0), self.num_class)) - 1)).type(torch.FloatTensor).to(self.device)
            target_one_hot_tr = one_hot_tr.scatter_(1, target.type(
                torch.LongTensor).view([-1, 1]).data.to(self.device), 1)

            member_output = self.attack_model(output, target_one_hot_tr)
            loss = self.criterion(output, target) + \
                (alpha)*(torch.mean((member_output)) - 0.5)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def train(self, train_loader, inference_loader, test_loader):

        best_accuracy = 0
        t_start = time.time()
        # check whether path exist
        if not os.path.exists(self.log_path):
            os.makedirs(self.log_path)

        # torch.save(self.model.state_dict(), os.path.join(
        #     self.log_path, '%s_0.pth' % (self.model_save_name)))

        # first train target model for 5 epochs
        for e in range(1, self.epochs+1):

            if e < 5:
                self.train_target_privately(train_loader)
            else:
                self.train_attack_advreg(train_loader, inference_loader)
                self.train_target_privately(train_loader)

            train_acc = self.eval(train_loader)
            test_acc = self.eval(test_loader)

            logx.msg('Train Epoch: %d, Total Sample: %d, Train Acc: %.3f, Test Acc: %.3f, Total Time: %.3fs' % (
                e, len(train_loader.dataset), train_acc, test_acc, time.time() - t_start))
            self.scheduler.step()
            self.scheduler_adv.step()

        #     if e % 10 == 0:
        #         torch.save(self.model.state_dict(), os.path.join(
        #             self.log_path, '%s_%d.pth' % (self.model_save_name, e)))

        # torch.save(self.model.state_dict(), os.path.join(
        #     self.log_path, "%s.pth" % self.model_save_name))
