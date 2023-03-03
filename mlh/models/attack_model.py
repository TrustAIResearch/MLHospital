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
import torch.nn as nn
import torch.nn.functional as F


class MLP_BLACKBOX(nn.Module):
    def __init__(self, dim_in):
        super(MLP_BLACKBOX, self).__init__()
        self.dim_in = dim_in
        self.fc1 = nn.Linear(self.dim_in, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 2)

    def forward(self, x):
        x = x.view(-1, self.dim_in)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class MLP_WHITEBOX(nn.Module):
    def __init__(self, dim_in_1, dim_in_2):
        super(MLP_WHITEBOX, self).__init__()

        self.dim_in_1 = dim_in_1
        self.dim_in_2 = dim_in_2

        self.fc1 = nn.Linear(self.dim_in_1, 64)
        self.fc2 = nn.Linear(self.dim_in_2, 64)

        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(64, 32)

        self.fc5 = nn.Linear(64, 32)
        self.fc6 = nn.Linear(32, 2)

    def forward(self, x1, x2):
        x1 = x1.view(-1, self.dim_in_1)
        x1 = F.relu(self.fc1(x1))
        x1 = F.relu(self.fc3(x1))

        x2 = x2.view(-1, self.dim_in_2)
        x2 = F.relu(self.fc2(x2))
        x2 = F.relu(self.fc4(x2))

        x = torch.cat((x1, x2), 1)
        x = F.relu(self.fc5(x))
        x = F.relu(self.fc6(x))

        return x


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


class WhiteBoxAttackModel(nn.Module):
    def __init__(self, class_num, embedding_dim):
        super(WhiteBoxAttackModel, self).__init__()

        self.dropout = nn.Dropout(p=0.2)
        self.output_component = nn.Sequential(
            nn.Linear(class_num, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
        )

        self.loss_component = nn.Sequential(
            nn.Linear(1, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
        )

        self.gradient_component = nn.Sequential(
            nn.Conv2d(1, 5, kernel_size=3, padding=0),
            nn.AdaptiveAvgPool2d(5),  # [batch_size, channel, 5, 5]
            nn.Flatten(),
            nn.Linear(125, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
        )

        self.label_component = nn.Sequential(
            nn.Linear(class_num, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
        )

        self.embedding_component = nn.Sequential(
            nn.Linear(embedding_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
        )

        self.encoder_component = nn.Sequential(
            nn.Linear(64 * 4, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
        )

    def forward(self, original_label, output, gradient, embedding, loss):
        label_component_result = self.label_component(original_label)
        output_component_result = self.output_component(output)
        gradient_component_result = self.gradient_component(gradient)
        # embedding_component_result = self.embedding_component(embedding)

        loss_component_result = self.loss_component(loss)

        # Loss_Component_result = F.softmax(Loss_Component_result, dim=1)
        # Gradient_Component_result = F.softmax(Gradient_Component_result, dim=1)

        # final_inputs = Output_Component_result
        # final_inputs = Loss_Component_result
        # final_inputs = Gradient_Component_result
        # final_inputs = Label_Component_result

        # final_inputs = torch.cat((label_component_result, output_component_result,
        #                           gradient_component_result, embedding_component_result, loss_component_result), 1)
        final_inputs = torch.cat((label_component_result, output_component_result,
                                  gradient_component_result, loss_component_result), 1)
        # final_inputs = torch.cat((label_component_result, output_component_result, embedding_component_result, loss_component_result), 1)
        # final_inputs = output_component_result
        final_result = self.encoder_component(final_inputs)

        return final_result
