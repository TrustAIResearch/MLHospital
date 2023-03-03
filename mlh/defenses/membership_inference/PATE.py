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
import torch.nn as nn
from tqdm import tqdm
import torch.optim as optim
from torch.utils.data import TensorDataset, Subset


from mlh.defenses.membership_inference.trainer import Trainer

from mlh.defenses.pate import perform_analysis


torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


class Classifier(nn.Module):
    def __init__(self, num_class=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_class)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x)

# https://github.com/tensorflow/privacy maybe also check it to update?


class TrainTargetPATE(object):
    def __init__(self, model, device='cuda:0', num_class=10, epochs=100, learning_rate=0.01, num_teacher=20, pate_epsilon=0.2, log_path="./"):

        self.device = device
        self.model = model.to(self.device)
        self.num_class = num_class
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.num_teacher = num_teacher
        self.pate_epsilon = pate_epsilon

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

    def eval(self, test_loader):

        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        correct = 0
        total = 0
        self.model.eval()
        with torch.no_grad():

            for img, label in test_loader:
                img, label = img.to(self.device), label.to(self.device)
                logits = self.model.eval().forward(img)

                predicted = torch.argmax(logits, dim=1)
                total += label.size(0)
                correct += (predicted == label).sum().item()

            final_acc = 100 * correct / total
            # print("in eval, total=", total)
        # self.model.train()
        return final_acc

    def train_teacher(self, model, trainloader, criterion, optimizer, epochs=20):
        running_loss = 0
        # each teacher use 'split_idx' batches

        for e in range(epochs):
            model.train()
            for images, labels in trainloader:

                images, labels = images.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                output = model.forward(images)
                loss = criterion(output, labels)
                loss.backward()
                optimizer.step()

    def predict(self, model, dataloader):
        outputs = torch.zeros(0, dtype=torch.long).to(self.device)
        model.eval()

        for images, labels in dataloader:
            images, labels = images.to(self.device), labels.to(self.device)
            output = model.forward(images)
            ps = torch.argmax(torch.exp(output), dim=1)
            outputs = torch.cat((outputs, ps))

        return outputs

    def train_models(self, teacher_loaders):
        models = []
        for i in tqdm(range(self.num_teacher)):
            model = Classifier(num_class=self.num_class).to(self.device)
            criterion = nn.NLLLoss()
            optimizer = optim.Adam(model.parameters(), lr=0.003)
            self.train_teacher(model, teacher_loaders[i], criterion, optimizer)
            models.append(model)
        return models

    def aggregated_teacher(self, models, dataloader):
        epsilon = self.pate_epsilon
        # here 10000 is the total number of student training samples (need to update)
        preds = torch.torch.zeros((len(models), 10000), dtype=torch.long)

        for i, model in enumerate(models):
            results = self.predict(model, dataloader)
            preds[i] = results

        labels = np.array([]).astype(int)
        for image_preds in np.transpose(preds):
            label_counts = np.bincount(image_preds, minlength=self.num_class)
            beta = 1 / epsilon

            for i in range(len(label_counts)):
                label_counts[i] += np.random.laplace(0, beta, 1)

            new_label = np.argmax(label_counts)
            labels = np.append(labels, new_label)

        return preds.numpy(), labels

    def student_loader(self, student_train_loader, labels):
        for i, (data, _) in enumerate(iter(student_train_loader)):
            yield data, torch.from_numpy(labels[i*len(data): (i+1)*len(data)])

    def get_data_loaders(self, train_data):
        """ Function to create data loaders for the Teacher classifier """
        teacher_loaders = []
        data_size = len(train_data) // self.num_teacher

        for i in range(data_size):
            indices = list(range(i*data_size, (i+1)*data_size))
            subset_data = Subset(train_data, indices)
            loader = torch.utils.data.DataLoader(subset_data, batch_size=50)
            teacher_loaders.append(loader)

        return teacher_loaders

    def dataloader2dataset(self, dataloader):
        img_list = []
        label_list = []
        for images, labels in dataloader:
            img_list.append(images)
            label_list.append(labels)
        img_list = torch.cat(img_list)
        label_list = torch.cat(label_list)
        dataset = TensorDataset(img_list, label_list)
        return dataset

    def train(self, train_loader, inference_loader, test_loader):

        train_loader.shuffle = False
        inference_loader.shuffle = False
        test_loader.shuffle = False

        # don't know why we need this step... but if we remove, the results are wrong... (dataloader -> dataset -> dataloader)
        train_data = self.dataloader2dataset(train_loader)
        inference_data = self.dataloader2dataset(inference_loader)
        test_data = self.dataloader2dataset(test_loader)

        inference_loader = torch.utils.data.DataLoader(
            inference_data, batch_size=128)
        test_loader = torch.utils.data.DataLoader(test_data, batch_size=128)

        teacher_loaders = self.get_data_loaders(train_data)
        teacher_models = self.train_models(teacher_loaders)

        preds, student_labels = self.aggregated_teacher(
            teacher_models, inference_loader)

        optimizer = optim.Adam(self.model.parameters(), lr=0.003)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.epochs)

        criterion = torch.nn.CrossEntropyLoss()

        best_accuracy = 0
        t_start = time.time()

        for e in range(1, self.epochs+1):
            batch_n = 0
            self.model.train()
            student_train_loader = self.student_loader(
                inference_loader, student_labels)
            # import pdb; pdb.set_trace()
            for img, label in student_train_loader:
                optimizer.zero_grad()
                batch_n += 1
                img, label = img.to(self.device), label.to(self.device)
                output = self.model.forward(img)
                loss = criterion(output, label)

                loss.backward()
                optimizer.step()

            student_train_loader = self.student_loader(
                inference_loader, student_labels)
            train_acc = self.eval(student_train_loader)
            train_acc_original = self.eval(train_loader)
            test_acc = self.eval(test_loader)
            print('Train Epoch: %d, Total Sample: %d, Train Acc: %.3f, Test Acc: %.3f, Total Time: %.3fs' % (
                e, len(train_loader.dataset), train_acc, test_acc, time.time() - t_start))
            scheduler.step()
            lr = scheduler.get_lr()
            print(train_acc_original, lr)

        data_dep_eps, data_ind_eps = perform_analysis(
            teacher_preds=preds, indices=student_labels, noise_eps=self.pate_epsilon, delta=1e-5)
        print("Data Independent Epsilon:", data_ind_eps)
        print("Data Dependent Epsilon:", data_dep_eps)
