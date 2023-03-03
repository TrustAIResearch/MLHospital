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

import torchvision
from mlh.attacks.attribute_inference.attacks import AttackDataset, AttributeInference, AIA_attack
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from mlh.data_preprocessing.data_loader import GetDataLoader
import argparse
import numpy as np
import torch.optim as optim
from mlh.defenses.attribute_inference.AdvTrain import TrainTargetAdvTrain
from mlh.defenses.attribute_inference.Olympus import TrainTargetOlympus, AutoEncoder
from mlh.defenses.attribute_inference.AttriGuard import TrainTargetAttriGuard
from runx.logx import logx
torch.manual_seed(0)
np.random.seed(0)
torch.set_num_threads(1)


def parse_args():
    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--batch-size', type=int, default=512,
                        help='batch_size')
    parser.add_argument('--num-workers', type=int, default=10,
                        help='num of workers to use')

    parser.add_argument('--epochs', type=int, default=10,
                        help='number of training epochs')
    parser.add_argument('--gpu', type=int, default=0,
                        help='gpu index used for training')

    # model dataset
    parser.add_argument('--model', type=str, default='resnet18')
    parser.add_argument('--load-pretrained', type=str, default='no')
    parser.add_argument('--task', type=str, default='aia',
                        help='specify the attack task, mia or ol')
    parser.add_argument('--dataset', type=str, default='CelebA',
                        help='dataset')
    parser.add_argument('--data-path', type=str, default='../data/',
                        help='data_path')
    parser.add_argument('--input-shape', type=str, default="32,32,3",
                        help='comma delimited input shape input')
    parser.add_argument('--defense', type=str, default='No',
                        help='No, AdvTrain, Olympus, and AttriGuard')
    parser.add_argument('--alpha', type=float, default='1.0',
                        help='The coef to balance defense methods')
    # parser.add_argument('--model_save_path', type=str, default='./save/', help='data_path')

    args = parser.parse_args()

    args.input_shape = [int(item) for item in args.input_shape.split(',')]
    args.device = 'cuda:%d' % args.gpu if torch.cuda.is_available() else 'cpu'

    return args


# target/shadow model
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def get_target_model(name="resnet18", num_classes=2):
    if name == "resnet18":
        model = torchvision.models.resnet18()
        model.fc = nn.Sequential(nn.Linear(512, num_classes))
    else:
        raise ValueError("Model not implemented yet :P")
    return model


def evaluate(model, dataloader):
    model.eval()
    correct = 0
    total = 0
    for data in dataloader:
        inputs, labels = data[0], data[1]['Young']
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

    target_model = get_target_model(name="resnet18", num_classes=2)
    alpha = args.alpha
    alpha_name = str(alpha).split('.')

    ''' Adversarially Training Target Model '''
    train = True
    if args.defense == 'No':
        log_path = 'save/CelebA/Normal/target/'
        logx.initialize(logdir=log_path, coolname=False, tensorboard=False)
        target_model_path = "save/CelebA/Normal/target/resnet18_NoAug.pth"
        if train == True:
            target_model.to(args.device)
            target_model.train()
            criterion = nn.CrossEntropyLoss()
            target_optimizer = optim.SGD(
                target_model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4, )
            # loop over the dataset multiple times
            for epoch in range(1, args.epochs + 1):
                running_loss = 0.0
                for i, data in tqdm(enumerate(target_train_loader), desc="Epoch %d" % epoch):
                    # get the inputs; data is a list of [inputs, labels]
                    inputs, labels = data[0], data[1]['Young']
                    inputs, labels = inputs.to(
                        args.device), labels.to(args.device)

                    # zero the parameter gradients
                    target_optimizer.zero_grad()

                    # forward + backward + optimize
                    outputs = target_model(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    target_optimizer.step()

                    # print statistics
                    running_loss += loss.item()

                # if epoch > args.epochs - 5:
                if epoch % 10 == 0:
                    train_acc = evaluate(target_model, target_train_loader)
                    test_acc = evaluate(target_model, target_test_loader)
                    logx.msg("[%d Epoch] Train Acc: %.3f, Test Acc: %.4f, Loss: %.3f" % (
                        epoch, train_acc, test_acc, running_loss))
                    torch.save(target_model.state_dict(), target_model_path)
            print('Finished Training')
    elif args.defense == 'AdvTrain':
        target_model_path = "save/CelebA/AdvTrain/target/resnet18_Fixed1Adv_Alpha_" + \
            alpha_name[0] + '_' + alpha_name[1] + ".pth"
        if train == True:
            total_evaluator = TrainTargetAdvTrain(
                target_model, alpha, epochs=args.epochs, log_path='save/CelebA/AdvTrain/target/')
            total_evaluator.train(target_train_loader, target_test_loader)
            trained_model = total_evaluator.target_model_E
            trained_model.fc = total_evaluator.target_model_C
            torch.save(trained_model.state_dict(), target_model_path)
            print('Finished Training')
    elif args.defense == 'Olympus':
        obfuscator_model_path = "save/CelebA/Olympus/target/obfuscator_Alpha_" + \
            alpha_name[0] + '_' + alpha_name[1] + ".pth"
        if train == True:
            target_model.load_state_dict(torch.load(
                "save/CelebA/Normal/target/resnet18_NoAug.pth"))
            total_evaluator = TrainTargetOlympus(
                target_model, alpha, epochs=args.epochs, log_path='save/CelebA/Olympus/target/')
            total_evaluator.train(target_train_loader, target_test_loader)
            trained_obfuscator = total_evaluator.obfuscator
            torch.save(trained_obfuscator.state_dict(), obfuscator_model_path)
            print('Finished Training')
    elif args.defense == 'AttriGuard':
        target_model.load_state_dict(torch.load(
            "save/CelebA/Normal/target/resnet18_NoAug.pth"))
        total_evaluator = TrainTargetAttriGuard(
            target_model, alpha, epochs=args.epochs, log_path='save/CelebA/AttriGuard/target/')
        total_evaluator.train(target_inference_loader, target_test_loader,
                              real_attack=True, attack_loader=shadow_train_loader)
        print('Finished Training and Testing')
    else:
        print("Not Implemented!")

    '''Load Trained Model'''
    obfuscator = None
    if args.defense == 'No':
        target_model.load_state_dict(torch.load(target_model_path))
        target_model.to(args.device)
        attack_evaluator = AIA_attack(
            target_model, epochs=args.epochs, log_path='save/CelebA/Normal/shadow/')
        attack_evaluator.train(shadow_train_loader, target_test_loader)
    elif args.defense == 'AdvTrain':
        target_model.load_state_dict(torch.load(target_model_path))
        target_model.to(args.device)
        attack_evaluator = AIA_attack(
            target_model, epochs=args.epochs, log_path='save/CelebA/AdvTrain/shadow/')
        attack_evaluator.train(shadow_train_loader, target_test_loader)
    elif args.defense == 'Olympus':
        target_model.load_state_dict(torch.load(
            "save/CelebA/Normal/target/resnet18_NoAug.pth"))
        obfuscator = AutoEncoder(input_dim=512, latent_dim=16)
        obfuscator.load_state_dict(torch.load(obfuscator_model_path))
        target_model.to(args.device)
        obfuscator.to(args.device)
        attack_evaluator = AIA_attack(target_model, epochs=args.epochs,
                                      log_path='save/CelebA/Olympus/shadow/', obfuscator=obfuscator)
        attack_evaluator.train(shadow_train_loader, target_test_loader)
    elif args.defense == 'AttriGuard':
        pass
    else:
        print("Not Implemented!")
