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


# from mlh.models.utils import FeatureExtractor, VerboseExecution

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.dataset import Dataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, auc, classification_report
from runx.logx import logx
from tqdm import tqdm
import numpy as np
import os
from art.attacks.evasion import HopSkipJump
from art.utils import compute_success
from art.estimators.classification.pytorch import PyTorchClassifier
from art.attacks.inference.membership_inference import LabelOnlyDecisionBoundary
import abc
from mlh.models.attack_model import MLP_BLACKBOX


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

    def get_posteriors(self, dataloader):
        info = {}
        target_list = []
        posteriors_list = []
        for btch_idx, (inputs, targets) in tqdm(enumerate(dataloader)):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            outputs = self.model(inputs)
            posteriors = F.softmax(outputs, dim=1)

            target_list += targets.cpu().tolist()
            posteriors_list += posteriors.detach().cpu().numpy().tolist()

        return {"targets": target_list, "posteriors": posteriors_list}

    def parse_info_whitebox(self, dataloader, layers):
        info = {}
        target_list = []
        posteriors_list = []
        embedding_list = []
        loss_list = []
        self.individual_criterion = nn.CrossEntropyLoss(reduction='none')
        self.model_feature = FeatureExtractor(self.model, layers=layers)

        for batch_idx, (inputs, targets) in tqdm(enumerate(dataloader)):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            outputs = self.model(inputs)  # can be reduced
            features = self.model_feature(inputs)

            emb = features[layers[-2]]  # can further specified
            emb = torch.flatten(emb, start_dim=1).detach().cpu().tolist()

            losses = self.individual_criterion(outputs, targets)

            losses = losses.detach().cpu().tolist()
            posteriors = F.softmax(outputs, dim=1).detach().cpu().tolist()

            target_list += targets.cpu().tolist()
            embedding_list += emb
            posteriors_list += posteriors
            loss_list += losses
        info = {"targets": target_list, "embeddings": embedding_list,
                "posteriors": posteriors_list, "losses": loss_list}
        return info


class AttackDataset():
    """
    Generate attack dataset
    """

    def __init__(self, args, attack_type, target_model, shadow_model, target_train_dataloader, target_test_dataloader, shadow_train_dataloader, shadow_test_dataloader):
        self.args = args
        self.attack_type = attack_type
        self.target_model_parser = ModelParser(args, target_model)
        self.shadow_model_parser = ModelParser(args, shadow_model)

        # if attack_type == "black-box":
        self.target_train_info = self.target_model_parser.get_posteriors(
            target_train_dataloader)
        self.target_test_info = self.target_model_parser.get_posteriors(
            target_test_dataloader)
        self.shadow_train_info = self.shadow_model_parser.get_posteriors(
            shadow_train_dataloader)
        self.shadow_test_info = self.shadow_model_parser.get_posteriors(
            shadow_test_dataloader)

        # get attack dataset
        self.attack_train_dataset, self.attack_test_dataset = self.generate_attack_dataset()

    def parse_info(self, info, label=0):
        mem_label = [label] * len(info["targets"])
        original_label = info["targets"]
        parse_type = self.attack_type
        mem_data = []
        if parse_type == "black-box":
            mem_data = info["posteriors"]
        elif parse_type == "black-box-sorted":
            mem_data = [sorted(row, reverse=True)
                        for row in info["posteriors"]]
        elif parse_type == "black-box-top3":
            mem_data = [sorted(row, reverse=True)[:3]
                        for row in info["posteriors"]]
        elif parse_type == "metric-based":
            mem_data = info["posteriors"]
        else:
            raise ValueError("More implementation is needed :P")
        return mem_data, mem_label, original_label

    def generate_attack_dataset(self):
        mem_data0, mem_label0, original_label0 = self.parse_info(
            self.target_train_info, label=1)
        mem_data1, mem_label1, original_label1 = self.parse_info(
            self.target_test_info, label=0)
        mem_data2, mem_label2, original_label2 = self.parse_info(
            self.shadow_train_info, label=1)
        mem_data3, mem_label3, original_label3 = self.parse_info(
            self.shadow_test_info, label=0)

        attack_train_dataset = torch.utils.data.TensorDataset(
            torch.from_numpy(np.array(mem_data2 + mem_data3, dtype='f')),
            torch.from_numpy(np.array(mem_label2 + mem_label3)
                             ).type(torch.long),
            torch.from_numpy(np.array(original_label2 +
                             original_label3)).type(torch.long),
        )

        attack_test_dataset = torch.utils.data.TensorDataset(
            torch.from_numpy(np.array(mem_data0 + mem_data1, dtype='f')),
            torch.from_numpy(np.array(mem_label0 + mem_label1)
                             ).type(torch.long),
            torch.from_numpy(np.array(original_label0 +
                             original_label1)).type(torch.long),
        )

        return attack_train_dataset, attack_test_dataset


class MembershipInferenceAttack(abc.ABC):
    """
    Abstract base class for membership inference attack classes.
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

    @staticmethod
    def cal_metric_for_class(self, label, pred_label, pred_posteriors, original_target_labels):
        """
        Calculate metrics for each class of the train (shadow) or test (target) dataset
        """

        class_list = sorted(list(set(original_target_labels)))
        for class_idx in class_list:
            subset_label = []
            subset_pred_label = []
            subset_pred_posteriors = []
            for i in range(len(label)):
                if original_target_labels[i] == class_idx:
                    subset_label.append(label[i])
                    subset_pred_label.append(pred_label[i])
                    subset_pred_posteriors.append(pred_posteriors[i])

            if len(subset_label) != 0:
                acc, precision, recall, f1, auc = self.cal_metrics(
                    subset_label, subset_pred_label, subset_pred_posteriors)

                print('Acc for class %d: %.3f%%, precision:%.3f, recall:%.3f, f1:%.3f, auc: %.3f' %
                      (class_idx, 100. * acc, precision, recall, f1, auc))


class MetricBasedMIA(MembershipInferenceAttack):
    def __init__(
            self,
            num_class,
            device,
            attack_type,
            attack_train_dataset,
            attack_test_dataset,
            batch_size=128):

        super().__init__()

        self.num_class = num_class
        self.device = device
        self.attack_type = attack_type
        self.attack_train_dataset = attack_train_dataset
        self.attack_test_dataset = attack_test_dataset
        self.attack_train_loader = torch.utils.data.DataLoader(
            attack_train_dataset, batch_size=batch_size, shuffle=True)
        self.attack_test_loader = torch.utils.data.DataLoader(
            attack_test_dataset, batch_size=batch_size, shuffle=False)

        if self.attack_type == "metric-based":
            self.metric_based_attacks()
        else:
            raise ValueError("Not implemented yet")

    def metric_based_attacks(self):
        """
        a little bit redundant since we make the data into torch dataset,
        but reverse them back into the original data...
        """
        self.parse_data_metric_based_attacks()

        train_tuple0, test_tuple0, test_results0 = self._mem_inf_via_corr()
        self.print_result("correct train", train_tuple0)
        self.print_result("correct test", test_tuple0)

        train_tuple1, test_tuple1, test_results1 = self._mem_inf_thre(
            'confidence', self.s_tr_conf, self.s_te_conf, self.t_tr_conf, self.t_te_conf)
        self.print_result("confidence train", train_tuple1)
        self.print_result("confidence test", test_tuple1)

        train_tuple2, test_tuple2, test_results2 = self._mem_inf_thre(
            'entropy', -self.s_tr_entr, -self.s_te_entr, -self.t_tr_entr, -self.t_te_entr)
        self.print_result("entropy train", train_tuple2)
        self.print_result("entropy test", test_tuple2)

        train_tuple3, test_tuple3, test_results3 = self._mem_inf_thre(
            'modified entropy', -self.s_tr_m_entr, -self.s_te_m_entr, -self.t_tr_m_entr, -self.t_te_m_entr)
        self.print_result("modified entropy train", train_tuple3)
        self.print_result("modified entropy test", test_tuple3)

    def print_result(self, name, given_tuple):
        print("%s" % name, "acc:%.3f, precision:%.3f, recall:%.3f, f1:%.3f, auc:%.3f" % given_tuple)

    def parse_data_metric_based_attacks(self):
        # shadow model
        self.s_tr_outputs, self.s_tr_labels = [], []
        self.s_te_outputs, self.s_te_labels = [], []
        for i in range(len(self.attack_train_dataset)):
            data, mem_label, target_label = self.attack_train_dataset[i]
            data, mem_label, target_label = data.numpy(), mem_label.item(), target_label.item()

            if mem_label == 1:
                self.s_tr_outputs.append(data)
                self.s_tr_labels.append(target_label)
            elif mem_label == 0:
                self.s_te_outputs.append(data)
                self.s_te_labels.append(target_label)

        # target model
        self.t_tr_outputs, self.t_tr_labels = [], []
        self.t_te_outputs, self.t_te_labels = [], []
        for i in range(len(self.attack_test_dataset)):
            data, mem_label, target_label = self.attack_test_dataset[i]
            data, mem_label, target_label = data.numpy(), mem_label.item(), target_label.item()
            if mem_label == 1:
                self.t_tr_outputs.append(data)
                self.t_tr_labels.append(target_label)
            elif mem_label == 0:
                self.t_te_outputs.append(data)
                self.t_te_labels.append(target_label)

        # change them into numpy array
        self.s_tr_outputs, self.s_tr_labels = np.array(
            self.s_tr_outputs), np.array(self.s_tr_labels)
        self.s_te_outputs, self.s_te_labels = np.array(
            self.s_te_outputs), np.array(self.s_te_labels)
        self.t_tr_outputs, self.t_tr_labels = np.array(
            self.t_tr_outputs), np.array(self.t_tr_labels)
        self.t_te_outputs, self.t_te_labels = np.array(
            self.t_te_outputs), np.array(self.t_te_labels)

        self.s_tr_mem_labels = np.ones(len(self.s_tr_labels))
        self.s_te_mem_labels = np.zeros(len(self.s_te_labels))
        self.t_tr_mem_labels = np.ones(len(self.t_tr_labels))
        self.t_te_mem_labels = np.zeros(len(self.t_te_labels))

        # prediction correctness
        self.s_tr_corr = (np.argmax(self.s_tr_outputs, axis=1)
                          == self.s_tr_labels).astype(int)
        self.s_te_corr = (np.argmax(self.s_te_outputs, axis=1)
                          == self.s_te_labels).astype(int)
        self.t_tr_corr = (np.argmax(self.t_tr_outputs, axis=1)
                          == self.t_tr_labels).astype(int)
        self.t_te_corr = (np.argmax(self.t_te_outputs, axis=1)
                          == self.t_te_labels).astype(int)

        # prediction confidence
        self.s_tr_conf = np.array(
            [self.s_tr_outputs[i, self.s_tr_labels[i]] for i in range(len(self.s_tr_labels))])
        self.s_te_conf = np.array(
            [self.s_te_outputs[i, self.s_te_labels[i]] for i in range(len(self.s_te_labels))])
        self.t_tr_conf = np.array(
            [self.t_tr_outputs[i, self.t_tr_labels[i]] for i in range(len(self.t_tr_labels))])
        self.t_te_conf = np.array(
            [self.t_te_outputs[i, self.t_te_labels[i]] for i in range(len(self.t_te_labels))])

        # prediction entropy
        self.s_tr_entr = self._entr_comp(self.s_tr_outputs)
        self.s_te_entr = self._entr_comp(self.s_te_outputs)
        self.t_tr_entr = self._entr_comp(self.t_tr_outputs)
        self.t_te_entr = self._entr_comp(self.t_te_outputs)

        # prediction modified entropy
        self.s_tr_m_entr = self._m_entr_comp(
            self.s_tr_outputs, self.s_tr_labels)
        self.s_te_m_entr = self._m_entr_comp(
            self.s_te_outputs, self.s_te_labels)
        self.t_tr_m_entr = self._m_entr_comp(
            self.t_tr_outputs, self.t_tr_labels)
        self.t_te_m_entr = self._m_entr_comp(
            self.t_te_outputs, self.t_te_labels)

    def _log_value(self, probs, small_value=1e-30):
        return -np.log(np.maximum(probs, small_value))

    def _entr_comp(self, probs):
        return np.sum(np.multiply(probs, self._log_value(probs)), axis=1)

    def _m_entr_comp(self, probs, true_labels):
        log_probs = self._log_value(probs)
        reverse_probs = 1-probs
        log_reverse_probs = self._log_value(reverse_probs)
        modified_probs = np.copy(probs)
        modified_probs[range(true_labels.size), true_labels] = reverse_probs[range(
            true_labels.size), true_labels]
        modified_log_probs = np.copy(log_reverse_probs)
        modified_log_probs[range(true_labels.size), true_labels] = log_probs[range(
            true_labels.size), true_labels]
        return np.sum(np.multiply(modified_probs, modified_log_probs), axis=1)

    def _thre_setting(self, tr_values, te_values):
        value_list = np.concatenate((tr_values, te_values))
        thre, max_acc = 0, 0
        for value in value_list:
            tr_ratio = np.sum(tr_values >= value)/(len(tr_values)+0.0)
            te_ratio = np.sum(te_values < value)/(len(te_values)+0.0)
            acc = 0.5*(tr_ratio + te_ratio)
            if acc > max_acc:
                thre, max_acc = value, acc
        return thre

    def _mem_inf_via_corr(self):
        # # perform membership inference attack based on whether the input is correctly classified or not
        train_mem_label = np.concatenate(
            [self.s_tr_mem_labels, self.s_te_mem_labels], axis=-1)
        train_pred_label = np.concatenate(
            [self.s_tr_corr, self.s_te_corr], axis=-1)
        train_pred_posteriors = np.concatenate(
            [self.s_tr_corr, self.s_te_corr], axis=-1)  # same as train_pred_label
        train_target_label = np.concatenate(
            [self.s_tr_labels, self.s_te_labels], axis=-1)

        test_mem_label = np.concatenate(
            [self.t_tr_mem_labels, self.t_te_mem_labels], axis=-1)
        test_pred_label = np.concatenate(
            [self.t_tr_corr, self.t_te_corr], axis=-1)
        test_pred_posteriors = np.concatenate(
            [self.t_tr_corr, self.t_te_corr], axis=-1)  # same as train_pred_label
        test_target_label = np.concatenate(
            [self.t_tr_labels, self.t_te_labels], axis=-1)

        train_acc, train_precision, train_recall, train_f1, train_auc = super().cal_metrics(
            train_mem_label, train_pred_label, train_pred_posteriors)
        test_acc, test_precision, test_recall, test_f1, test_auc = super().cal_metrics(
            test_mem_label, test_pred_label, test_pred_posteriors)

        test_results = {"test_mem_label": test_mem_label,
                        "test_pred_label": test_pred_label,
                        "test_pred_prob": test_pred_posteriors,
                        "test_target_label": test_target_label}

        train_tuple = (train_acc, train_precision,
                       train_recall, train_f1, train_auc)
        test_tuple = (test_acc, test_precision,
                      test_recall, test_f1, test_auc)
        # print(train_tuple, test_tuple)
        return train_tuple, test_tuple, test_results

    def _mem_inf_thre(self, v_name, s_tr_values, s_te_values, t_tr_values, t_te_values):
        # perform membership inference attack by thresholding feature values: the feature can be prediction confidence,
        # (negative) prediction entropy, and (negative) modified entropy

        train_mem_label = []
        train_pred_label = []
        train_pred_posteriors = []
        train_target_label = []

        test_mem_label = []
        test_pred_label = []
        test_pred_posteriors = []
        test_target_label = []

        thre_list = [self._thre_setting(s_tr_values[self.s_tr_labels == num],
                                        s_te_values[self.s_te_labels == num]) for num in range(self.num_class)]

        # shadow train
        for i in range(len(s_tr_values)):
            original_label = self.s_tr_labels[i]
            thre = thre_list[original_label]
            pred = s_tr_values[i]
            pred_label = int(s_tr_values[i] >= thre)

            train_mem_label.append(1)
            train_pred_label.append(pred_label)
            # indicator function, so the posterior equals to 0 or 1
            train_pred_posteriors.append(pred)
            train_target_label.append(original_label)

        # shadow test
        for i in range(len(s_te_values)):
            original_label = self.s_te_labels[i]
            thre = thre_list[original_label]
            pred = s_te_values[i]
            pred_label = int(s_te_values[i] >= thre)

            train_mem_label.append(0)
            train_pred_label.append(pred_label)
            # indicator function, so the posterior equals to 0 or 1
            train_pred_posteriors.append(pred)
            train_target_label.append(original_label)

        # target train
        for i in range(len(t_tr_values)):
            original_label = self.t_tr_labels[i]
            thre = thre_list[original_label]
            pred = t_tr_values[i]
            pred_label = int(t_tr_values[i] >= thre)

            test_mem_label.append(1)
            test_pred_label.append(pred_label)
            # indicator function, so the posterior equals to 0 or 1
            test_pred_posteriors.append(pred)
            test_target_label.append(original_label)

        # target test
        for i in range(len(t_te_values)):
            original_label = self.t_te_labels[i]
            thre = thre_list[original_label]
            pred = t_te_values[i]
            pred_label = int(t_te_values[i] >= thre)

            test_mem_label.append(0)
            test_pred_label.append(pred_label)
            # indicator function, so the posterior equals to 0 or 1
            test_pred_posteriors.append(pred)
            test_target_label.append(original_label)

        train_acc, train_precision, train_recall, train_f1, train_auc = super().cal_metrics(
            train_mem_label, train_pred_label, train_pred_posteriors)
        test_acc, test_precision, test_recall, test_f1, test_auc = super().cal_metrics(
            test_mem_label, test_pred_label, test_pred_posteriors)

        train_tuple = (train_acc, train_precision,
                       train_recall, train_f1, train_auc)
        test_tuple = (test_acc, test_precision,
                      test_recall, test_f1, test_auc)
        test_results = {"test_mem_label": test_mem_label,
                        "test_pred_label": test_pred_label,
                        "test_pred_prob": test_pred_posteriors,
                        "test_target_label": test_target_label}

        return train_tuple, test_tuple, test_results


class BlackBoxMIA(MembershipInferenceAttack):
    def __init__(
            self,
            num_class,
            device,
            attack_type,
            attack_train_dataset,
            attack_test_dataset,
            batch_size=128):

        super().__init__()

        self.num_class = num_class
        self.device = device
        self.attack_type = attack_type
        self.attack_train_dataset = attack_train_dataset
        self.attack_test_dataset = attack_test_dataset
        self.attack_train_loader = torch.utils.data.DataLoader(
            attack_train_dataset, batch_size=batch_size, shuffle=True)
        self.attack_test_loader = torch.utils.data.DataLoader(
            attack_test_dataset, batch_size=batch_size, shuffle=False)

        if self.attack_type == "black-box":
            self.attack_model = MLP_BLACKBOX(dim_in=self.num_class)
        elif self.attack_type == "black-box-sorted":
            self.attack_model = MLP_BLACKBOX(dim_in=self.num_class)
        elif self.attack_type == "black-box-top3":
            self.attack_model = MLP_BLACKBOX(dim_in=3)
        else:
            raise ValueError("Not implemented yet")

        self.attack_model = self.attack_model.to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.train(self.attack_train_loader)

    def train(self, dataloader, train_epoch=100):
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
            print('Epoch: %d, Overall Train Acc: %.3f%%, precision:%.3f, recall:%.3f, f1:%.3f, auc: %.3f' % (
                e, 100. * train_acc, train_precision, train_recall, train_f1, train_auc))
            print('Epoch: %d, Overall Test Acc: %.3f%%, precision:%.3f, recall:%.3f, f1:%.3f, auc: %.3f\n\n' % (
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
            # print single class performance
            super().cal_metric_for_class(super(),
                                         labels, pred_labels, pred_posteriors, original_target_labels)

            test_results = {"test_mem_label": labels,
                            "test_pred_label": pred_labels,
                            "test_pred_prob": pred_posteriors,
                            "test_target_label": original_target_labels}

        self.attack_model.train()
        return test_acc, test_precision, test_recall, test_f1, test_auc, test_results


def MSE(y, t):
    return 0.5 * np.sum((y - t)**2)


def prediction(x):
    x_list = x[0].tolist()
    x_sort = sorted(x_list)
    max_index = x_list.index(x_sort[-1])

    x_row_max = x.max(axis=-1)
    x_row_max = x_row_max.reshape(list(x.shape)[:-1]+[1])
    x = x - x_row_max
    x_exp = np.exp(x)
    x_exp_row_sum = x_exp.sum(axis=-1).reshape(list(x.shape)[:-1]+[1])
    softmax = x_exp / x_exp_row_sum

    return softmax, max_index  # , sec_index


class LabelOnlyMIA(MembershipInferenceAttack):
    def __init__(
            self,
            device,
            target_model,
            shadow_model,
            target_loader=None,
            shadow_loader=None,
            input_shape=(3, 32, 32),
            nb_classes=10,
            batch_size=1000,
            ):

        super().__init__()
        
        self.device = device
        self.target_train_dataset = target_loader[0].dataset
        self.target_test_dataset = target_loader[1].dataset
        self.target_train_loader = torch.utils.data.DataLoader(
            self.target_train_dataset, batch_size=batch_size, shuffle=True)
        self.target_test_loader = torch.utils.data.DataLoader(
            self.target_test_dataset, batch_size=batch_size, shuffle=True)
        self.target_model = target_model

        self.shadow_train_dataset = shadow_loader[0].dataset
        self.shadow_test_dataset = shadow_loader[1].dataset
        self.shadow_train_loader = torch.utils.data.DataLoader(
            self.shadow_train_dataset, batch_size=batch_size, shuffle=True)
        self.shadow_test_loader = torch.utils.data.DataLoader(
            self.shadow_test_dataset, batch_size=batch_size, shuffle=True)
        self.shadow_model = shadow_model

        self.input_shape = input_shape
        self.nb_classes = nb_classes

    def SearchThreshold(self):
        ARTclassifier = PyTorchClassifier(
            model=self.shadow_model,
            clip_values=None,
            loss=F.cross_entropy,
            input_shape=self.input_shape,
            nb_classes=self.nb_classes
        )

        for idx, (data, label) in enumerate(self.shadow_train_loader):
            x_train = data.numpy() if idx == 0 else np.concatenate((x_train, data.numpy()), axis=0)
            y_train = label.numpy() if idx == 0 else np.concatenate((y_train, label.numpy()), axis=0)

        for idx, (data, label) in enumerate(self.shadow_test_loader):
            x_test = data.numpy() if idx == 0 else np.concatenate((x_test, data.numpy()), axis=0)
            y_test = label.numpy() if idx == 0 else np.concatenate((y_test, label.numpy()), axis=0)

        Attack = LabelOnlyDecisionBoundary(estimator=ARTclassifier)
        Attack.calibrate_distance_threshold(x_train, y_train, x_test, y_test)
        distance_threshold = Attack.distance_threshold_tau
        return distance_threshold

    def Infer(self):

        thd = self.SearchThreshold()

        ARTclassifier = PyTorchClassifier(
            model=self.target_model,
            clip_values=None,
            loss=F.cross_entropy,
            input_shape=self.input_shape,
            nb_classes=self.nb_classes
        )

        for idx, (data, label) in enumerate(self.target_train_loader):
            x_train = data.numpy() if idx == 0 else np.concatenate((x_train, data.numpy()), axis=0)
            y_train = label.numpy() if idx == 0 else np.concatenate((y_train, label.numpy()), axis=0)

        for idx, (data, label) in enumerate(self.target_test_loader):
            x_test = data.numpy() if idx == 0 else np.concatenate((x_test, data.numpy()), axis=0)
            y_test = label.numpy() if idx == 0 else np.concatenate((y_test, label.numpy()), axis=0)

        Attack = LabelOnlyDecisionBoundary(estimator=ARTclassifier, distance_threshold_tau=thd)

        x_train_test = np.concatenate((x_train, x_test), axis=0)
        y_train_test = np.concatenate((y_train, y_test), axis=0)
        prediction = Attack.infer(x_train_test, y_train_test)
   
        member_groundtruth = np.ones(int(len(x_train)))
        non_member_groundtruth = np.zeros(int(len(x_train)))
        groundtruth = np.concatenate((member_groundtruth, non_member_groundtruth))
   
        fpr, tpr, _ = roc_curve(groundtruth, prediction, pos_label=1, drop_intermediate=False)
        AUC = round(auc(fpr, tpr), 4)
        return AUC
