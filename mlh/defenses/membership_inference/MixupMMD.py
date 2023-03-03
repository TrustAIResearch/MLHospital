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
import torch.nn.functional as F
from tqdm import tqdm
from mlh.defenses.membership_inference.trainer import Trainer


"""
Modify a bit from:
https://github.com/colaalex111/MMD-mixup-Defense
"""


def _mix_rbf_kernel(X, Y, sigma_list):
    assert (X.size(0) == Y.size(0))
    m = X.size(0)

    Z = torch.cat((X, Y), 0)

    ZZT = torch.mm(Z, Z.t())

    diag_ZZT = torch.diag(ZZT).unsqueeze(1)

    Z_norm_sqr = diag_ZZT.expand_as(ZZT)

    exponent = Z_norm_sqr - 2 * ZZT + Z_norm_sqr.t()

    # print (exponent.size(),exponent)

    K = 0.0
    for sigma in sigma_list:
        gamma = 1.0 / (2 * sigma**2)
        K += torch.exp(-gamma * exponent)

    return K[:m, :m], K[:m, m:], K[m:, m:], len(sigma_list)


def _mmd2(K_XX, K_XY, K_YY, const_diagonal=False, biased=False):
    m = K_XX.size(0)    # assume X, Y are same shape

    # Get the various sums of kernels that we'll use
    # Kts drop the diagonal, but we don't need to compute them explicitly
    if const_diagonal is not False:
        diag_X = diag_Y = const_diagonal
        sum_diag_X = sum_diag_Y = m * const_diagonal
    else:
        diag_X = torch.diag(K_XX)                       # (m,)
        diag_Y = torch.diag(K_YY)                       # (m,)
        sum_diag_X = torch.sum(diag_X)
        sum_diag_Y = torch.sum(diag_Y)

    # \tilde{K}_XX * e = K_XX * e - diag_X
    Kt_XX_sums = K_XX.sum(dim=1) - diag_X
    # \tilde{K}_YY * e = K_YY * e - diag_Y
    Kt_YY_sums = K_YY.sum(dim=1) - diag_Y
    K_XY_sums_0 = K_XY.sum(dim=0)                     # K_{XY}^T * e

    Kt_XX_sum = Kt_XX_sums.sum()                       # e^T * \tilde{K}_XX * e
    Kt_YY_sum = Kt_YY_sums.sum()                       # e^T * \tilde{K}_YY * e
    K_XY_sum = K_XY_sums_0.sum()                       # e^T * K_{XY} * e

    if biased:
        mmd2 = ((Kt_XX_sum + sum_diag_X) / (m * m)
                + (Kt_YY_sum + sum_diag_Y) / (m * m)
                - 2.0 * K_XY_sum / (m * m))
    else:
        mmd2 = (Kt_XX_sum / (m * (m - 1))
                + Kt_YY_sum / (m * (m - 1))
                - 2.0 * K_XY_sum / (m * m))

    return mmd2


def mix_rbf_mmd2(X, Y, sigma_list, biased=True):
    K_XX, K_XY, K_YY, d = _mix_rbf_kernel(X, Y, sigma_list)
    # return _mmd2(K_XX, K_XY, K_YY, const_diagonal=d, biased=biased)
    return _mmd2(K_XX, K_XY, K_YY, const_diagonal=False, biased=biased)


def mixup_data(x, y, alpha=1.0, use_cuda=True):
    '''Compute the mixup data. Return mixed inputs, pairs of targets, and lambda'''
    if alpha > 0.:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.
    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(y_a, y_b, lam):
    return lambda criterion, pred: lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


class TrainTargetMixupMMD(Trainer):
    def __init__(self, model, device="cuda:0", num_class=10, epochs=100, learning_rate=0.01, momentum=0.9, weight_decay=5e-4, mixup=1, mixup_alpha=1.0, mmd_loss_lambda=3, log_path="./"):

        super().__init__()

        self.model = model
        self.device = device
        self.num_class = num_class
        self.epochs = epochs
        self.mixup = mixup
        self.mixup_alpha = mixup_alpha
        self.mmd_loss_lambda = mmd_loss_lambda

        self.model = self.model.to(self.device)

        self.optimizer = torch.optim.SGD(self.model.parameters(
        ), learning_rate, momentum=momentum, weight_decay=weight_decay)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=self.epochs)
        self.criterion = torch.nn.CrossEntropyLoss()

        self.log_path = log_path
        logx.initialize(logdir=self.log_path,
                        coolname=False, tensorboard=False)

    @staticmethod
    def _sample_weight_decay():
        # We selected the l2 regularization parameter from a range of 45 logarithmically spaced values between 10−6 and 105
        weight_decay = np.logspace(-6, 5, num=45, base=10.0)
        weight_decay = np.random.choice(weight_decay)
        print("Sampled weight decay:", weight_decay)
        return weight_decay

    def eval(self, data_loader):
        t1 = time.time()

        correct = 0
        total = 0
        self.model.eval()
        with torch.no_grad():

            for img, label in data_loader:
                img, label = img.to(self.device), label.to(self.device)
                logits = self.model(img)

                predicted = torch.argmax(logits, dim=1)
                total += label.size(0)
                correct += (predicted == label).sum().item()

            final_acc = 100 * correct / total
            # print("in eval, total=", total)
        # self.model.train()

        return final_acc

    def train(self, train_loader, train_loader_ordered, inference_loader_ordered, test_loader, starting_index, inference_sorted):

        best_accuracy = 0
        t_start = time.time()
        # check whether path exist
        if not os.path.exists(self.log_path):
            os.makedirs(self.log_path)

        # torch.save(self.model.state_dict(), os.path.join(
        #     self.log_path, '%s_0.pth' % (self.model_save_name)))
        for e in range(1, self.epochs+1):
            batch_n = 0
            self.model.train()
            for img, label in tqdm(train_loader, desc="train mixup"):
                self.model.zero_grad()
                batch_n += 1

                if (self.mixup):
                    inputs, targets_a, targets_b, lam = mixup_data(
                        img, label, self.mixup_alpha)
                    inputs, targets_a, targets_b = inputs.to(self.device), targets_a.to(
                        self.device), targets_b.to(self.device)
                    outputs = self.model(inputs)
                    loss_func = mixup_criterion(targets_a, targets_b, lam)
                    loss = loss_func(self.criterion, outputs)
                    loss.backward()
                    self.optimizer.step()

                else:
                    img, label = img.to(self.device), label.to(self.device)
                    # print("img", img.shape)
                    logits = self.model(img)
                    loss = self.criterion(logits, label)
                    loss.backward()
                    self.optimizer.step()

            train_acc = self.eval(train_loader)
            eval_acc = self.eval(inference_loader_ordered)
            # test_acc = self.eval(test_loader)
            # print("epoch:%d\ttrain_acc:%.3f\ttest_acc:%.3f\ttotal_time:%.3fs" % (e, train_acc, test_acc, time.time() - t_start))
            # logx.msg('[After MMD] Train Epoch: %d, Total Sample: %d, Test Acc: %.3f, Total Time: %.3fs' % (
            #     e, len(train_loader.dataset), test_acc, time.time() - t_start))
            logx.msg('[Before MMD] Train Epoch: %d, Total Sample: %d, Train Acc: %.3f, Eval Acc: %.3f, Total Time: %.3fs' % (
                e, len(train_loader.dataset), train_acc, eval_acc, time.time() - t_start))

            # scheduler.step()

            if (abs(eval_acc - train_acc) < 3 or self.mmd_loss_lambda < 1e-5):
                continue
            else:
                pass

            # validation_label_in_training = []
            # validation_confidence_in_training = []

            for train_images, train_labels in tqdm(train_loader_ordered, desc="train MMD"):
                batch_num = train_labels.size()[0]
                self.model.zero_grad()
                valid_images = torch.zeros_like(train_images).type(
                    torch.FloatTensor).to(self.device)
                valid_labels = torch.zeros_like(train_labels).type(
                    torch.LongTensor).to(self.device)
                valid_index = 0

                for i in torch.unique(train_labels):
                    this_frequency = torch.bincount(
                        train_labels)[i].to(self.device)
                    this_class_start = starting_index[i]

                    if (i < self.num_class - 1):
                        this_class_end = starting_index[i+1]-1
                    else:
                        this_class_end = len(inference_sorted) - 1

                    for j in range(this_frequency):
                        random_index = np.random.randint(
                            this_class_start, this_class_end)
                        new_images, new_labels = (
                            (inference_sorted).__getitem__(random_index))
                        valid_images[valid_index] = new_images.to(
                            self.device)
                        valid_labels[valid_index] = (torch.ones(
                            1)*new_labels).type(torch.LongTensor).to(self.device)
                        valid_index += 1

                train_images = train_images.to(self.device)
                train_labels = train_labels.to(self.device)
                outputs = self.model(train_images)
                all_train_outputs = F.softmax(outputs, dim=1)
                # all_train_outputs = all_train_outputs.view(-1,num_classes)
                train_labels = train_labels.view(batch_num, 1)

                valid_images = valid_images.to(self.device)
                valid_labels = valid_labels.to(self.device)
                outputs = self.model(valid_images)
                all_valid_outputs = F.softmax(outputs, dim=1)
                all_valid_outputs = (all_valid_outputs).detach_()
                valid_labels = valid_labels.view(batch_num, 1)

                if (self.mmd_loss_lambda > 0):
                    mmd_loss = mix_rbf_mmd2(all_train_outputs, all_valid_outputs, sigma_list=[
                                            1]) * self.mmd_loss_lambda
                    mmd_loss.backward()

                # MMD regularization shouldn't be applied in the last training epoch for better testing accuracy
                if (e != self.epochs):
                    self.optimizer.step()

            train_acc = self.eval(train_loader)
            eval_acc = self.eval(inference_loader_ordered)
            test_acc = self.eval(test_loader)
            # print("epoch:%d\ttrain_acc:%.3f\ttest_acc:%.3f\ttotal_time:%.3fs" % (e, train_acc, test_acc, time.time() - t_start))
            logx.msg('[After MMD] Train Epoch: %d, Total Sample: %d, Train Acc: %.3f, Eval Acc: %.3f, Test Acc: %.3f, Total Time: %.3fs' % (
                e, len(train_loader.dataset), train_acc, eval_acc, test_acc, time.time() - t_start))
            self.scheduler.step()
        #     # save normal model for every 10 epochs
        #     if e % 10 == 0:
        #         torch.save(self.model.state_dict(), os.path.join(
        #             self.log_path, '%s_%d.pth' % (self.model_save_name, e)))

        # if not os.path.exists(self.log_path):
        #     os.makedirs(self.log_path)

        # torch.save(self.model.state_dict(), os.path.join(
        #     self.log_path, "%s.pth" % self.model_save_name))
