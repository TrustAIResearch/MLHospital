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

import abc
from mlh.models.attack_model import MLP_BLACKBOX
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report
from runx.logx import logx
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import os
import pandas as pd
import time
import copy
from torch.utils.data.dataset import Dataset


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


class VerboseExecution(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

        # Register a hook for each layer
        for name, layer in self.model.named_children():
            layer.__name__ = name
            layer.register_forward_hook(
                lambda layer, _, output: print(
                    f"{layer.__name__}: {output.shape}")
            )

    def forward(self, x):
        return self.model(x)


class ModelParser():
    """
    ModelParser handles what information should be extracted from the target/shadow model
    """

    def __init__(self, args, model):
        self.args = args
        self.device = self.args.device
        self.model = model.to(self.device)

    def parse_info_whitebox(self, dataloader, layers, obfuscator):
        info = {}
        target_list = []
        aux_target_list = []
        posteriors_list = []
        embedding_list = []
        loss_list = []
        self.individual_criterion = nn.CrossEntropyLoss(reduction='none')
        self.model_feature = FeatureExtractor(self.model, layers=layers)

        # for batch_idx, (inputs, targets, aux_targtets) in tqdm(enumerate(dataloader)):
        #     inputs, targets, aux_targtets = inputs.to(self.device), targets.to(
        #         self.device), aux_targtets.to(self.device)
        #     outputs = self.model(inputs)  # can be reduced
        #     features = self.model_feature(inputs)
        #
        #     emb = features[layers[-2]]  # can further specified
        #     emb = torch.flatten(emb, start_dim=1).detach().cpu().tolist()
        #
        #     losses = self.individual_criterion(outputs, targets)
        #
        #     losses = losses.detach().cpu().tolist()
        #     posteriors = F.softmax(outputs, dim=1).detach().cpu().tolist()
        #
        #     target_list += targets.cpu().tolist()
        #     aux_target_list += aux_targtets.cpu().tolist()
        #     embedding_list += emb
        #     posteriors_list += posteriors
        #     loss_list += losses
        # info = {"targets": target_list, "aux_targets": aux_target_list, "embeddings": embedding_list,
        #         "posteriors": posteriors_list, "losses": loss_list}
        # return info

        for batch_idx, (inputs, all_targets) in tqdm(enumerate(dataloader)):
            # The original task is young, the inference task is male
            inputs, targets, aux_targtets = inputs.to(self.device), all_targets['Young'].to(
                self.device), all_targets['Male'].to(self.device)
            outputs = self.model(inputs)  # can be reduced
            features = self.model_feature(inputs)

            emb = features[layers[-2]]  # can further specified
            emb = torch.flatten(emb, start_dim=1).detach().cpu().tolist()

            losses = self.individual_criterion(outputs, targets)

            losses = losses.detach().cpu().tolist()
            posteriors = F.softmax(outputs, dim=1).detach().cpu().tolist()

            target_list += targets.cpu().tolist()
            aux_target_list += aux_targtets.cpu().tolist()
            embedding_list += emb
            posteriors_list += posteriors
            loss_list += losses
        info = {"targets": target_list, "aux_targets": aux_target_list, "embeddings": embedding_list,
                "posteriors": posteriors_list, "losses": loss_list}
        return info


class AttackDataset():
    """
    Generate attack dataset
    """

    def __init__(self, args, attack_type, target_model, target_train_dataloader, target_test_dataloader, layers, obfuscator):
        self.args = args
        self.attack_type = attack_type
        self.target_model_parser = ModelParser(args, target_model)

        self.target_train_info = self.target_model_parser.parse_info_whitebox(
            target_train_dataloader, layers, obfuscator)
        self.target_test_info = self.target_model_parser.parse_info_whitebox(
            target_test_dataloader, layers, obfuscator)

        # get attack dataset
        self.attack_train_dataset, self.attack_test_dataset = self.generate_attack_dataset()

    def parse_info(self, info):
        original_labels = info["targets"]
        aux_labels = info["aux_targets"]
        data = info["embeddings"]

        return data, aux_labels, original_labels

    def generate_attack_dataset(self):
        train_data, train_aux_labels, train_original_labels = self.parse_info(
            self.target_train_info)
        test_data, test_aux_labels, test_original_labels = self.parse_info(
            self.target_test_info)

        attack_train_dataset = torch.utils.data.TensorDataset(
            torch.from_numpy(np.array(train_data, dtype='f')),
            torch.from_numpy(np.array(train_aux_labels)).type(torch.long),
            torch.from_numpy(np.array(train_original_labels)).type(torch.long),
        )

        attack_test_dataset = torch.utils.data.TensorDataset(
            torch.from_numpy(np.array(test_data, dtype='f')),
            torch.from_numpy(np.array(test_aux_labels)).type(torch.long),
            torch.from_numpy(np.array(test_original_labels)).type(torch.long),
        )

        return attack_train_dataset, attack_test_dataset


class AttributeInferenceAttack(abc.ABC):
    """
    Abstract base class for attribute inference attack classes.
    """

    def __init__(self,):

        super().__init__()

    @staticmethod
    def cal_metrics(label, pred_label, pred_posteriors):
        acc = accuracy_score(label, pred_label)
        precision = precision_score(label, pred_label)
        recall = recall_score(label, pred_label)
        f1 = f1_score(label, pred_label)
        auc = roc_auc_score(label, pred_posteriors)

        return acc, precision, recall, f1, auc


class AttributeInference(AttributeInferenceAttack):
    """
    We follow the paper "Overlearning Reveals Sensitive Attributes" (https://arxiv.org/abs/1905.11742) 
    to implement our attribute inference attacks
    """

    def __init__(
            self,
            emb_dim,
            device,
            attack_train_dataset,
            attack_test_dataset,
            results_path,
            batch_size=128):

        super().__init__()

        self.emb_dim = emb_dim
        self.device = device
        self.attack_train_dataset = attack_train_dataset
        self.attack_test_dataset = attack_test_dataset
        self.attack_train_loader = torch.utils.data.DataLoader(
            attack_train_dataset, batch_size=batch_size, shuffle=True)
        self.attack_test_loader = torch.utils.data.DataLoader(
            attack_test_dataset, batch_size=batch_size, shuffle=False)

        self.attack_model = MLP_BLACKBOX(dim_in=self.emb_dim)
        self.attack_model = self.attack_model.to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.results_path = results_path
        self.train(self.attack_train_loader)

    def train(self, dataloader, train_epoch=100):
        results = pd.DataFrame(columns = ['Epoch', 'Train Acc', 'Test Acc'])
        self.attack_model.train()
        self.optimizer = torch.optim.Adam(
            self.attack_model.parameters(), lr=0.001)

        for e in range(1, train_epoch + 1):
            train_loss = 0

            labels = []
            pred_labels = []
            pred_posteriors = []
            for inputs, targets, original_labels in dataloader:
                self.optimizer.zero_grad()
                inputs, targets = inputs.to(
                    self.device), targets.to(self.device)
                outputs = self.attack_model(inputs)
                posteriors = F.softmax(outputs, dim=1)
                loss = self.criterion(outputs, targets)
                loss.backward()
                self.optimizer.step()

                train_loss += loss.item()
                _, predicted = outputs.max(1)

                labels += targets.cpu().tolist()
                pred_labels += predicted.cpu().tolist()
                pred_posteriors += posteriors.cpu().tolist()

            pred_posteriors = [row[1] for row in pred_posteriors]

            train_acc, train_precision, train_recall, train_f1, train_auc = super().cal_metrics(
                labels, pred_labels, pred_posteriors)
            test_acc, test_precision, test_recall, test_f1, test_auc, test_results = self.infer(
                self.attack_test_loader)
            results = results.append({'Epoch': e, 'Train Acc': train_acc, 'Test Acc': test_acc}, ignore_index=True)
            results.to_csv(self.results_path)
            print('Epoch: %d, Overall Train Attack Acc: %.3f%%, precision:%.3f, recall:%.3f, f1:%.3f, auc: %.3f' % (
                e, 100. * train_acc, train_precision, train_recall, train_f1, train_auc))
            print('Epoch: %d, Overall Test Attack Acc: %.3f%%, precision:%.3f, recall:%.3f, f1:%.3f, auc: %.3f\n\n' % (
                e, 100. * test_acc, test_precision, test_recall, test_f1, test_auc))

            train_tuple = (train_acc, train_precision,
                           train_recall, train_f1, train_auc)
            test_tuple = (test_acc, test_precision,
                          test_recall, test_f1, test_auc)

        return train_tuple, test_tuple, test_results

    def infer(self, dataloader):
        self.attack_model.eval()
        original_target_labels = []
        labels = []
        pred_labels = []
        pred_posteriors = []
        with torch.no_grad():
            for inputs, targets, original_labels in dataloader:

                inputs, targets, original_labels = inputs.to(self.device), targets.to(
                    self.device), original_labels.to(self.device)
                outputs = self.attack_model(inputs)
                posteriors = F.softmax(outputs, dim=1)
                _, predicted = outputs.max(1)

                labels += targets.cpu().tolist()
                pred_labels += predicted.cpu().tolist()
                pred_posteriors += posteriors.cpu().tolist()
                original_target_labels += original_labels.cpu().tolist()

            pred_posteriors = [row[1] for row in pred_posteriors]

            test_acc, test_precision, test_recall, test_f1, test_auc = super().cal_metrics(
                labels, pred_labels, pred_posteriors)
                
            test_results = {"test_aux_label": labels,
                            "test_pred_label": pred_labels,
                            "test_pred_prob": pred_posteriors,
                            "test_target_label": original_target_labels}

        self.attack_model.train()
        return test_acc, test_precision, test_recall, test_f1, test_auc, test_results


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


class AIA_attack(AttributeInferenceAttack):
    def __init__(self, target_model, device="cuda:0", emd_dim=512, tar_att='Young', sen_att='Male', num_class=2,
                 epochs=10, learning_rate=0.01, momentum=0.9, weight_decay=5e-4, log_path="./", obfuscator=None):

        super().__init__()

        # self.opt = opt
        self.target_model = target_model
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
        self.target_model_E.to(self.device)
        self.attack_model.to(self.device)

        self.optimizer_adv = torch.optim.SGD(self.attack_model.parameters(
        ), learning_rate, momentum=momentum, weight_decay=weight_decay)
        self.scheduler_adv = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer_adv, T_max=self.epochs)

        self.criterion = torch.nn.CrossEntropyLoss()

        self.log_path = log_path
        logx.initialize(logdir=self.log_path,
                        coolname=False, tensorboard=False)
        self.obfuscator = obfuscator
        if obfuscator != None:
            self.obfuscator.to(self.device)


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
                if self.obfuscator != None:
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
        self.target_model_E.eval()
        self.attack_model.train()

        for batch_idx, (img, targets) in tqdm(enumerate(train_loader)):
            train_target = targets[self.sen_att]
            train_data, train_target = img.to(
                self.device), train_target.to(self.device)
            emb = self.target_model_E(train_data)
            if self.obfuscator != None:
                emb = self.obfuscator(emb)
            attack_outputs = self.attack_model(emb)
            loss = self.criterion(attack_outputs, train_target)

            self.optimizer_adv.zero_grad()
            loss.backward()
            self.optimizer_adv.step()


    def train(self, train_loader, test_loader):

        t_start = time.time()
        # check whether path exist
        if not os.path.exists(self.log_path):
            os.makedirs(self.log_path)

        # first train target model for 5 epochs
        for e in range(1, self.epochs+1):
            print("Epoch: " + str(e))

            self.train_attack_advtrain(train_loader)

            if e % 10 == 0:
                attack_train_acc = self.eval_attack(train_loader)
                attack_test_acc = self.eval_attack(test_loader)

                logx.msg('Train Epoch: %d, Total Sample: %d, Train Attack Acc: %.3f, Test Attack Acc: %.3f Total Time: %.3fs' % (
                    e, len(train_loader.dataset), attack_train_acc, attack_test_acc, time.time() - t_start))
            self.scheduler_adv.step()
