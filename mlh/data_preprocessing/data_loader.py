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

import torchvision.transforms as transforms
import torch
import numpy as np
from io import RawIOBase
from mlh.data_preprocessing.dataset_preprocessing import prepare_dataset, cut_dataset, prepare_inference_dataset
from torchvision import datasets
from PIL import Image
from tqdm import tqdm
from mlh.data_preprocessing import configs

torch.manual_seed(0)


class GetDataLoader(object):
    def __init__(self, args):
        self.args = args
        self.data_path = args.data_path
        self.input_shape = args.input_shape

    def parse_dataset(self, dataset, train_transform, test_transform):

        if dataset in configs.SUPPORTED_IMAGE_DATASETS:
            _loader = getattr(datasets, dataset)
            if dataset != "EMNIST":
                train_dataset = _loader(root=self.data_path,
                                        train=True,
                                        transform=train_transform,
                                        download=True)
                test_dataset = _loader(root=self.data_path,
                                       train=False,
                                       transform=test_transform,
                                       download=True)
            else:
                train_dataset = _loader(root=self.data_path,
                                        train=True,
                                        split="byclass",
                                        transform=train_transform,
                                        download=True)
                test_dataset = _loader(root=self.data_path,
                                       train=False,
                                       split="byclass",
                                       transform=test_transform,
                                       download=True)
            dataset = train_dataset + test_dataset

        elif dataset in configs.SUPPORTED_IMAGE_DATASETS:
            from mlh.data_preprocessing.attribute_data_parser import CelebA
            _loader = CelebA
            train_dataset = _loader(root=self.data_path,
                                    train=True,
                                    transform=train_transform,
                                    download=True)
            test_dataset = _loader(root=self.data_path,
                                   train=False,
                                   transform=test_transform,
                                   download=True)
            dataset = train_dataset + test_dataset

        else:
            raise ValueError("Dataset Not Supported: ", dataset)
        return dataset

    def get_data_transform(self, dataset, use_transform="simple"):
        transform_list = [transforms.Resize(
            (self.input_shape[0], self.input_shape[0])), ]

        if use_transform == "simple":
            transform_list += [transforms.RandomCrop(
                32, padding=4), transforms.RandomHorizontalFlip(), ]

            print("add simple data augmentation!")

        transform_list.append(transforms.ToTensor())

        if dataset in ["MNIST", "FashionMNIST", "EMNIST"]:
            transform_list = [
                transforms.Grayscale(3), ] + transform_list

        transform_ = transforms.Compose(transform_list)
        return transform_

    # def get_data_transform(self, dataset):
    #     train_transform_list = [transforms.Resize(
    #         (self.input_shape[0], self.input_shape[0])), ]
    #     test_transform_list = [transforms.Resize(
    #         (self.input_shape[0], self.input_shape[0])), ]

    #     train_transform_list += [transforms.RandomCrop(
    #         32, padding=4), transforms.RandomHorizontalFlip(), ]
    #     test_transform_list += [transforms.RandomCrop(
    #         32, padding=4), transforms.RandomHorizontalFlip(), ]
    #     print("add simple data augmentation!")

    #     train_transform_list.append(transforms.ToTensor())
    #     test_transform_list.append(transforms.ToTensor())

    #     if dataset in ["MNIST", "FashionMNIST", "EMNIST"]:
    #         train_transform_list = [
    #             transforms.Grayscale(3), ] + train_transform_list
    #         test_transform_list = [
    #             transforms.Grayscale(3), ] + test_transform_list

    #     train_transform = transforms.Compose(train_transform_list)
    #     test_transform = transforms.Compose(test_transform_list)

    #     return train_transform, test_transform

    def get_dataset(self, train_transform, test_transform):
        dataset = self.parse_dataset(
            self.args.dataset, train_transform, test_transform)
        return dataset

    def get_inference_dataset(self, train_transform, test_transform):
        dataset = self.parse_dataset(
            self.args.inference_dataset, train_transform, test_transform)
        return dataset

    def get_data_supervised(self, batch_size=128, num_workers=2):

        train_transform = self.get_data_transform(self.args.dataset)
        test_transform = self.get_data_transform(self.args.dataset)

        dataset = self.get_dataset(train_transform, test_transform)

        target_train, target_inference, target_test, shadow_train, shadow_inference, shadow_test = prepare_dataset(
            dataset, select_num=None)

        print("Preparing dataloader!")
        print("dataset: ", len(dataset))
        print("target_train: %d \t target_inference: %s \t target_test: %s" %
              (len(target_train), len(target_inference), len(target_test)))

        target_train_loader = torch.utils.data.DataLoader(
            target_train, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
        target_inference_loader = torch.utils.data.DataLoader(
            target_inference, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
        target_test_loader = torch.utils.data.DataLoader(
            target_test, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
        shadow_train_loader = torch.utils.data.DataLoader(
            shadow_train, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
        shadow_inference_loader = torch.utils.data.DataLoader(
            shadow_inference, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
        shadow_test_loader = torch.utils.data.DataLoader(
            shadow_test, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)

        return target_train_loader, target_inference_loader, target_test_loader, shadow_train_loader, shadow_inference_loader, shadow_test_loader

    def get_ordered_dataset(self, target_dataset):
        """
        Inspired by https://stackoverflow.com/questions/66695251/define-manually-sorted-mnist-dataset-with-batch-size-1-in-pytorch
        """
        label = np.array([row[1] for row in target_dataset])
        sorted_index = np.argsort(label)
        sorted_dataset = torch.utils.data.Subset(target_dataset, sorted_index)
        return sorted_dataset

    def get_label_index(self, target_dataset):
        """
        return starting index for different labels in the sorted dataset
        """
        label_index = []
        start_label = 0
        label = np.array([row[1] for row in target_dataset])
        for i in range(len(label)):
            if label[i] == start_label:
                label_index.append(i)
                start_label += 1
        return label_index

    def get_sorted_data_mixup_mmd(self):

        train_transform = self.get_data_transform(self.args.dataset)
        test_transform = self.get_data_transform(self.args.dataset)
        dataset = self.get_dataset(train_transform, test_transform)

        target_train, target_inference, target_test, shadow_train, shadow_inference, shadow_test = prepare_dataset(
            dataset, select_num=None)

        target_train_sorted = self.get_ordered_dataset(target_train)
        target_inference_sorted = self.get_ordered_dataset(target_inference)
        shadow_train_sorted = self.get_ordered_dataset(shadow_train)
        shadow_inference_sorted = self.get_ordered_dataset(shadow_inference)

        start_index_target_inference = self.get_label_index(
            target_inference_sorted)
        start_index_shadow_inference = self.get_label_index(
            shadow_inference_sorted)

        # note that we set the inference loader's batch size to 1
        target_train_sorted_loader = torch.utils.data.DataLoader(
            target_train_sorted, batch_size=self.args.batch_size, shuffle=False, num_workers=self.args.num_workers, pin_memory=True)
        target_inference_sorted_loader = torch.utils.data.DataLoader(
            target_inference_sorted, batch_size=self.args.batch_size, shuffle=False, num_workers=self.args.num_workers, pin_memory=True)
        shadow_train_sorted_loader = torch.utils.data.DataLoader(
            shadow_train_sorted, batch_size=self.args.batch_size, shuffle=False, num_workers=self.args.num_workers, pin_memory=True)
        shadow_inference_sorted_loader = torch.utils.data.DataLoader(
            shadow_inference_sorted, batch_size=self.args.batch_size, shuffle=False, num_workers=self.args.num_workers, pin_memory=True)

        return target_train_sorted_loader, target_inference_sorted_loader, shadow_train_sorted_loader, shadow_inference_sorted_loader, start_index_target_inference, start_index_shadow_inference, target_inference_sorted, shadow_inference_sorted


class GetDataLoaderPoison(object):
    def __init__(self, args):
        self.args = args
        self.data_path = args.data_path
        self.input_shape = args.input_shape

    def parse_dataset(self, dataset):

        if dataset in configs.SUPPORTED_IMAGE_DATASETS:
            _loader = getattr(datasets, dataset)
            if dataset != "EMNIST":
                train_dataset = _loader(root=self.data_path,
                                        train=True,
                                        transform=None,
                                        download=True)
                test_dataset = _loader(root=self.data_path,
                                       train=False,
                                       transform=None,
                                       download=True)
            else:
                train_dataset = _loader(root=self.data_path,
                                        train=True,
                                        split="byclass",
                                        transform=None,
                                        download=True)
                test_dataset = _loader(root=self.data_path,
                                       train=False,
                                       split="byclass",
                                       transform=None,
                                       download=True)

        else:
            raise ValueError("Dataset Not Supported: ", dataset)
        return train_dataset, test_dataset

    def get_data_transform(self, dataset, use_transform="simple"):
        transform_list = [transforms.Resize(
            (self.input_shape[0], self.input_shape[0])), ]

        if use_transform == "simple":
            transform_list += [transforms.RandomCrop(
                32, padding=4), transforms.RandomHorizontalFlip(), ]

            print("add simple data augmentation!")

        transform_list.append(transforms.ToTensor())

        if dataset in ["MNIST", "FashionMNIST", "EMNIST"]:
            transform_list = [
                transforms.Grayscale(3), ] + transform_list

        transform_ = transforms.Compose(transform_list)
        return transform_

    def get_data_loader(self):

        train_transform = self.get_data_transform(
            self.args.dataset, use_transform=None)
        test_transform = self.get_data_transform(
            self.args.dataset, use_transform=None)

        train_dataset, test_dataset = self.parse_dataset(self.args.dataset)

        bad_params = {
            'trigger': self.args.trigger,
            'attack': self.args.attack,
            'src_label': self.args.atk_src_label,
            'tar_label': self.args.atk_tar_label,
        }

        # get train data
        bad_train_indices, trigger_trans, attack_trans = prepare_backdoor_attack(in_size=self.input_shape[0],
                                                                                 classes=self.args.num_class,
                                                                                 labels=train_dataset.targets,
                                                                                 proportion=self.args.atk_proportion,
                                                                                 **bad_params)
        clr_trainset = BackdoorDataset(train_dataset, train_transform)
        bad_trainset = BackdoorDataset(
            train_dataset, train_transform, bad_train_indices, trigger_trans, attack_trans)

        # get test data
        bad_test_indices, trigger_trans, attack_trans = prepare_backdoor_attack(in_size=self.input_shape[0],
                                                                                classes=self.args.num_class,
                                                                                labels=test_dataset.targets,
                                                                                proportion=1.0,
                                                                                **bad_params)

        clr_testset = BackdoorDataset(test_dataset, test_transform)
        bad_testset = BackdoorDataset(
            test_dataset, test_transform, bad_test_indices, trigger_trans, attack_trans)

        bad_testset = IndicesDataset(bad_testset, bad_test_indices)
        # print(bad_test_indices)
        # print(bad_testset[0])
        # print(len(bad_testset))
        # exit()

        # generate dataloader
        bad_trainloader = torch.utils.data.DataLoader(
            bad_trainset, batch_size=self.args.batch_size, shuffle=True, drop_last=True, num_workers=self.args.num_workers)

        clr_testloader = torch.utils.data.DataLoader(
            clr_testset,
            batch_size=self.args.batch_size, shuffle=False, drop_last=False, num_workers=self.args.num_workers)

        bad_testloader = torch.utils.data.DataLoader(
            bad_testset,
            batch_size=self.args.batch_size, shuffle=False, drop_last=False, num_workers=self.args.num_workers)

        return bad_trainloader, clr_testloader, bad_testloader


def prepare_backdoor_attack(
        in_size: int,
        classes: int,
        labels: list,
        proportion: float = 0.0,
        trigger: str = None,
        attack: str = None,
        src_label: int = None,
        tar_label: int = None) -> tuple:

    if (trigger is None) or (attack is None):
        return None, None, None

    mask = np.ones([in_size, in_size], dtype=np.uint8)
    pattern = np.zeros([in_size, in_size], dtype=np.uint8)

    if trigger == 'single-pixel':
        mask[-2, -2], pattern[-2, -2] = 0, 255

    elif trigger == 'pattern':
        mask[-2, -2], pattern[-2, -2] = 0, 255
        mask[-2, -4], pattern[-2, -4] = 0, 255
        mask[-4, -2], pattern[-4, -2] = 0, 255
        mask[-3, -3], pattern[-3, -3] = 0, 255

    else:
        raise ValueError(
            'backdoor trigger {} is not supported'.format(trigger))

    trigger_trans = TriggerTrans(mask, pattern)

    if attack == 'single-target':
        indices = np.where(np.array(labels) == src_label)[0]
        attack_trans = SingleTargetTrans(src_label, tar_label)

    elif attack == 'all-to-all':
        indices = np.arange(len(labels))
        attack_trans = AlltoAllTrans(classes)

    elif attack == "all-to-single":
        indices = np.arange(len(labels))
        attack_trans = AlltoSingleTrans(tar_label)

    else:
        raise ValueError('backdoor attack {} is not supported'.format(attack))

    backdoor_num = int(len(indices) * proportion)
    backdoor_indices = np.random.permutation(indices)[:backdoor_num]

    return backdoor_indices, trigger_trans, attack_trans


class BackdoorDataset:
    def __init__(
            self, dataset, transform=None,
            backdoor_indices=None, trigger_trans=None, attack_trans=None):

        self.dataset = dataset
        self.transform = transform

        self.backdoor_indices = set() if (
            backdoor_indices is None) else set(backdoor_indices)
        self.trigger_trans = trigger_trans
        self.attack_trans = attack_trans

    def __getitem__(self, i):
        x, y = self.dataset[i]

        if i in self.backdoor_indices:
            x, y = self.trigger_trans(x), self.attack_trans(y)

        if self.transform is not None:
            x = self.transform(x)

        return x, y

    def __len__(self):
        return len(self.dataset)


class IndicesDataset:
    def __init__(self, dataset, indices=None):
        self.dataset = dataset
        self.indices = range(len(dataset)) if (indices is None) else indices

    def __getitem__(self, i):
        return self.dataset[self.indices[i]]

    def __len__(self):
        return len(self.indices)


class TriggerTrans:
    def __init__(self, mask, pattern):
        self.mask = mask
        self.pattern = pattern

    def __call__(self, x):
        x = np.asarray(x)
        mask, pattern = self.mask, self.pattern

        if len(x.shape) == 3:
            mask = mask.reshape(*mask.shape, 1)
            pattern = pattern.reshape(*pattern.shape, 1)
        x_trigger = x * mask + pattern
        # change numpy array back to PIL.Image
        x_trigger = Image.fromarray(x_trigger)
        return x_trigger


class SingleTargetTrans:
    def __init__(self, src_label, tar_label):
        self.src_label = src_label
        self.tar_label = tar_label

    def __call__(self, y):
        assert y == self.src_label
        return self.tar_label


class AlltoAllTrans:
    def __init__(self, classes):
        self.classes = classes

    def __call__(self, y):
        return (y+1) % self.classes


class AlltoSingleTrans:
    def __init__(self, tar_label):
        self.tar_label = tar_label

    def __call__(self, y):
        return self.tar_label
