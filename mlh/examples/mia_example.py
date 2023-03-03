import torchvision
from mlh.attacks.membership_inference.attacks import AttackDataset, BlackBoxMIA, MetricBasedMIA, LabelOnlyMIA
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from mlh.data_preprocessing.data_loader import GetDataLoader
from torchvision import datasets
import torchvision.transforms as transforms
import argparse
import numpy as np
import torch.optim as optim
torch.manual_seed(0)
np.random.seed(0)
torch.set_num_threads(1)


def parse_args():
    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--batch-size', type=int, default=512,
                        help='batch_size')
    parser.add_argument('--num-workers', type=int, default=10,
                        help='num of workers to use')

    parser.add_argument('--epochs', type=int, default=100,
                        help='number of training epochs')
    parser.add_argument('--gpu', type=int, default=0,
                        help='gpu index used for training')

    # model dataset
    parser.add_argument('--model', type=str, default='resnet18')
    parser.add_argument('--load-pretrained', type=str, default='no')
    parser.add_argument('--dataset', type=str, default='CIFAR10',
                        help='dataset')
    parser.add_argument('--num-class', type=int, default=10,
                        help='number of classes')
    parser.add_argument('--training_type', type=str, default="Normal",
                        help='Normal, LabelSmoothing, AdvReg, DP, MixupMMD, PATE')
    parser.add_argument('--inference-dataset', type=str, default='CIFAR10',
                        help='if yes, load pretrained attack model to inference')
    parser.add_argument('--attack_type', type=str, default='black-box',
                        help='attack type: "black-box", "black-box-sorted", "black-box-top3", "metric-based", and "label-only"')
    parser.add_argument('--data-path', type=str, default='../datasets/',
                        help='data_path')
    parser.add_argument('--input-shape', type=str, default="32,32,3",
                        help='comma delimited input shape input')
    parser.add_argument('--log_path', type=str,
                        default='./save', help='')

    args = parser.parse_args()

    args.input_shape = [int(item) for item in args.input_shape.split(',')]
    args.device = 'cuda:%d' % args.gpu if torch.cuda.is_available() else 'cpu'

    return args


def get_target_model(name="resnet18", num_classes=10):
    if name == "resnet18":
        model = torchvision.models.resnet18()
        model.fc = nn.Sequential(nn.Linear(512, 10))
    else:
        raise ValueError("Model not implemented yet :P")
    return model


def evaluate(model, dataloader):
    model.eval()
    correct = 0
    total = 0
    for data in dataloader:
        inputs, labels = data
        inputs, labels = inputs.to(args.device), labels.to(args.device)
        outputs = model(inputs)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    model.train()
    return correct / total


if __name__ == "__main__":

    args = parse_args()
    s = GetDataLoader(args)
    target_train_loader, target_inference_loader, target_test_loader, shadow_train_loader, shadow_inference_loader, shadow_test_loader = s.get_data_supervised()

    target_model = get_target_model(name="resnet18", num_classes=10)
    shadow_model = get_target_model(name="resnet18", num_classes=10)

    # load target/shadow model to conduct the attacks
    target_model.load_state_dict(torch.load(
        f'{args.log_path}/{args.dataset}/{args.training_type}/target/{args.model}.pth'))
    target_model = target_model.to(args.device)

    shadow_model.load_state_dict(torch.load(
        f'{args.log_path}/{args.dataset}/{args.training_type}/shadow/{args.model}.pth'))
    shadow_model = shadow_model.to(args.device)

    # generate attack dataset
    # or "black-box, black-box-sorted", "black-box-top3", "metric-based", and "label-only"
    attack_type = args.attack_type

    # attack_type = "metric-based"

    if attack_type == "label-only":
        attack_model = LabelOnlyMIA(
            device=args.device,
            target_model=target_model.eval(),
            shadow_model=shadow_model.eval(),
            target_loader=(target_train_loader, target_test_loader),
            shadow_loader=(shadow_train_loader, shadow_test_loader),
            input_shape=(3, 32, 32),
            nb_classes=10)
        auc = attack_model.Infer()
        print(auc)

    else:

        attack_dataset = AttackDataset(args, attack_type, target_model, shadow_model,
                                       target_train_loader, target_test_loader, shadow_train_loader, shadow_test_loader)

        # train attack model

        if "black-box" in attack_type:
            attack_model = BlackBoxMIA(
                num_class=args.num_class,
                device=args.device,
                attack_type=attack_type,
                attack_train_dataset=attack_dataset.attack_train_dataset,
                attack_test_dataset=attack_dataset.attack_test_dataset,
                batch_size=128)
        elif "metric-based" in attack_type:
            attack_model = MetricBasedMIA(
                num_class=args.num_class,
                device=args.device,
                attack_type=attack_type,
                attack_train_dataset=attack_dataset.attack_train_dataset,
                attack_test_dataset=attack_dataset.attack_test_dataset,
                batch_size=128)
