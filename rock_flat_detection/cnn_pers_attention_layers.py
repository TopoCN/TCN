import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class PeriodicAttensionTopoLayer(nn.Module):
    def __init__(self, n_elements, point_dimension, dim_out):
        super(PeriodicAttensionTopoLayer, self).__init__()

        self.n_elements = n_elements
        self.point_dimension = point_dimension
        self.mu = nn.Parameter(torch.FloatTensor(1, self.point_dimension))
        self.alpha = nn.Parameter(torch.FloatTensor(1, self.point_dimension))
        self.period = nn.Parameter(torch.FloatTensor(1, self.point_dimension))
        self.length = nn.Parameter(torch.FloatTensor(1, self.point_dimension))
        self.sigma = nn.Parameter(torch.FloatTensor(1))
        self.weights = nn.Parameter(torch.FloatTensor(self.n_elements, dim_out))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.mu, mode='fan_out', a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.alpha, mode='fan_out', a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.period, mode='fan_out', a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.length, mode='fan_out', a=math.sqrt(5))
        nn.init.uniform_(self.sigma, 0, 1)
        nn.init.kaiming_uniform_(self.weights, mode='fan_out', a=math.sqrt(5))

    def forward(self, input):
        center = self.mu.unsqueeze(0)
        centers = center.repeat(input.size(0), input.size(1), 1)

        alpha = self.alpha.unsqueeze(0)
        alphas = alpha.repeat(input.size(0), input.size(1), 1)

        period = self.period.unsqueeze(0)
        periods = period.repeat(input.size(0), input.size(1), 1)

        length = self.length.unsqueeze(0)
        lengths = length.repeat(input.size(0), input.size(1), 1)

        # sin part
        norm_to_alpha = (((input - alphas) ** 2)/periods) * torch.FloatTensor([math.pi]).to(input.device)
        norm_to_sin = (torch.sin(norm_to_alpha) ** 2) * torch.FloatTensor([2.]).to(input.device)

        # centering part
        norm_to_center = ((input - centers) ** 2) * lengths * torch.FloatTensor([0.5]).to(input.device)

        # sum
        output = torch.sum(norm_to_sin, dim = 2) + torch.sum(norm_to_center, dim = 2)
        output = torch.exp(-output) * self.sigma # (B, n_elements)

        # arctan pers @ used for attention
        arctan_pers = torch.atan((input[:, :, 1] ** 2) * (torch.FloatTensor([2.])).to(input.device))  # (B, n_elements)

        h = torch.einsum('bp, po -> bo', output, self.weights)  # (B, dim_out)
        h = F.elu(h)
        return h, arctan_pers


class GraphModel(nn.Module):
    def __init__(self, nu, n_elements, point_dimension, rotate_num_dgms, hks_num_dgms, dim_intermediate, dim_out, final_dropout, num_class):
        super(GraphModel, self).__init__()
        self.rotate_num_dgms = rotate_num_dgms
        self.hks_num_dgms = hks_num_dgms

        self.topo_fcs_rotate = nn.ModuleList()
        for _ in range(rotate_num_dgms):
            self.topo_fcs_rotate.append(PeriodicAttensionTopoLayer(n_elements, point_dimension, dim_intermediate))


        # attention weights
        self.rotate_attention_weight_1 = nn.Parameter(torch.FloatTensor(n_elements, 1))
        self.rotate_attention_weight_2 = nn.Parameter(torch.FloatTensor(n_elements, 1))
        self.rotate_attention_weight_3 = nn.Parameter(torch.FloatTensor(n_elements, 1))
        self.rotate_attention_weight_4 = nn.Parameter(torch.FloatTensor(n_elements, 1))
        nn.init.kaiming_uniform_(self.rotate_attention_weight_1, mode='fan_out', a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.rotate_attention_weight_2, mode='fan_out', a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.rotate_attention_weight_3, mode='fan_out', a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.rotate_attention_weight_4, mode='fan_out', a=math.sqrt(5))

        # MLP
        self.fc1 = nn.Linear(dim_intermediate, dim_out)
        self.dropout = nn.Dropout(final_dropout)
        self.fc2 = nn.Linear(dim_out, num_class)

        # CNN-based model
        self.conv1 = nn.Sequential(
        nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, stride=1, padding=2,),
                  nn.ReLU(),nn.MaxPool2d(kernel_size=3),
                  )
        self.conv2 = nn.Sequential(nn.Conv2d(32, 64, 5, 1, 2), nn.ReLU(), nn.MaxPool2d(2),)
        self.out = nn.Linear(64 * 10 * 10, 3)

    def forward(self, rotate_input, x):

        # rotated
        rotate_topolayer_outputs = []
        for i in range(self.rotate_num_dgms): # default: 4 PDs
            rotate_topolayer_outputs.append(self.topo_fcs_rotate[i](rotate_input[i])[0]) # (B, dim_intermediate)

        # attention in pers -> rotated
        rotate_attention_outputs = []
        for i in range(self.rotate_num_dgms): # default: 4 PDs
            rotate_attention_outputs.append(self.topo_fcs_rotate[i](rotate_input[i])[1]) # use arctan_pers output only; (B, n_elements)

        rotate_attention_weights = [self.rotate_attention_weight_1, self.rotate_attention_weight_2,
                                    self.rotate_attention_weight_3, self.rotate_attention_weight_4]


        rotate_attention_hs = []
        for i in range(self.rotate_num_dgms):
            rotate_attention_hs.append(torch.matmul(rotate_attention_outputs[i], rotate_attention_weights[i]).squeeze(-1)) # (B,) for each of them

        normalized = False

        if normalized:
            rotation_attention_exp = F.softmax(torch.cat(rotate_attention_hs, dim=-1), dim=1)
            attention_topolayer_hs = [torch.einsum('bi, b -> bi', rotate_topolayer_outputs[i], rotation_attention_exp[:, i]) for i in range(self.rotate_num_dgms)]
        else:
            attention_topolayer_hs = [torch.einsum('bi, b -> bi', rotate_topolayer_outputs[i], rotate_attention_hs[i]) for i in range(self.rotate_num_dgms)]

        attention_topolayer_h = attention_topolayer_hs[0] + attention_topolayer_hs[1] + attention_topolayer_hs[2] + attention_topolayer_hs[3] # (B, dim_intermediate)

        output = self.fc1(attention_topolayer_h) # (B, dim_out)
        output = self.dropout(output)
        output_tda = self.fc2(output) # (B, num_class)

        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        output_cnn = self.out(x)
        final_output = output_cnn * 0.99 + output_tda * 0.01

        return final_output
