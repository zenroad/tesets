import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as grad

from utils import *
from transform_net import input_transform_net


from torch.autograd import Variable
from torch.nn.modules.batchnorm import _BatchNorm
import numpy as np
import math
import time


class Swish(nn.Module):
    def __init__(self):
        super(Swish, self).__init__()

    def forward(self, x):
        return 1.78718727865 * (x * torch.sigmoid(x) - 0.20662096414)


class MyBatchNorm1d(_BatchNorm):
    r"""Applies Batch Normalization over a 2d or 3d input that is seen as a
    mini-batch.
    .. math::
        y = \frac{x - mean[x]}{ \sqrt{Var[x] + \epsilon}} * gamma + beta
    The mean and standard-deviation are calculated per-dimension over
    the mini-batches and gamma and beta are learnable parameter vectors
    of size C (where C is the input size).
    During training, this layer keeps a running estimate of its computed mean
    and variance. The running sum is kept with a default momentum of 0.1.
    During evaluation, this running mean/variance is used for normalization.
    Because the BatchNorm is done over the `C` dimension, computing statistics
    on `(N, L)` slices, it's common terminology to call this Temporal BatchNorm
    Args:
        num_features: num_features from an expected input of size
            `batch_size x num_features [x width]`
        eps: a value added to the denominator for numerical stability.
            Default: 1e-5
        momentum: the value used for the running_mean and running_var
            computation. Default: 0.1
        affine: a boolean value that when set to ``True``, gives the layer learnable
            affine parameters. Default: ``True``
    Shape:
        - Input: :math:`(N, C)` or :math:`(N, C, L)`
        - Output: :math:`(N, C)` or :math:`(N, C, L)` (same shape as input)
    """
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True, momentum_decay_step=None, momentum_decay=1):
        super(MyBatchNorm1d, self).__init__(num_features, eps, momentum, affine)
        self.momentum_decay_step = momentum_decay_step
        self.momentum_decay = momentum_decay
        self.momentum_original = self.momentum

    def _check_input_dim(self, input):
        if input.dim() != 2 and input.dim() != 3:
            raise ValueError('expected 2D or 3D input (got {}D input)'
                             .format(input.dim()))
        super(MyBatchNorm1d, self)._check_input_dim(input)

    def forward(self, input, epoch=None):
        if (epoch is not None) and (epoch >= 1) and (self.momentum_decay_step is not None) and (self.momentum_decay_step > 0):
            # perform momentum decay
            self.momentum = self.momentum_original * (self.momentum_decay**(epoch//self.momentum_decay_step))
            if self.momentum < 0.01:
                self.momentum = 0.01


        return F.batch_norm(
            input, self.running_mean, self.running_var, self.weight, self.bias,
            self.training, self.momentum, self.eps)


class MyBatchNorm2d(_BatchNorm):
    r"""Applies Batch Normalization over a 4d input that is seen as a mini-batch
    of 3d inputs
    .. math::
        y = \frac{x - mean[x]}{ \sqrt{Var[x] + \epsilon}} * gamma + beta
    The mean and standard-deviation are calculated per-dimension over
    the mini-batches and gamma and beta are learnable parameter vectors
    of size C (where C is the input size).
    During training, this layer keeps a running estimate of its computed mean
    and variance. The running sum is kept with a default momentum of 0.1.
    During evaluation, this running mean/variance is used for normalization.
    Because the BatchNorm is done over the `C` dimension, computing statistics
    on `(N, H, W)` slices, it's common terminology to call this Spatial BatchNorm
    Args:
        num_features: num_features from an expected input of
            size batch_size x num_features x height x width
        eps: a value added to the denominator for numerical stability.
            Default: 1e-5
        momentum: the value used for the running_mean and running_var
            computation. Default: 0.1
        affine: a boolean value that when set to ``True``, gives the layer learnable
            affine parameters. Default: ``True``
    Shape:
        - Input: :math:`(N, C, H, W)`
        - Output: :math:`(N, C, H, W)` (same shape as input)
    """
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True, momentum_decay_step=None, momentum_decay=1):
        super(MyBatchNorm2d, self).__init__(num_features, eps, momentum, affine)
        self.momentum_decay_step = momentum_decay_step
        self.momentum_decay = momentum_decay
        self.momentum_original = self.momentum

    def _check_input_dim(self, input):
        if input.dim() != 4:
            raise ValueError('expected 4D input (got {}D input)'
                             .format(input.dim()))
        super(MyBatchNorm2d, self)._check_input_dim(input)

    def forward(self, input, epoch=None):
        if (epoch is not None) and (epoch >= 1) and (self.momentum_decay_step is not None) and (self.momentum_decay_step > 0):
            # perform momentum decay
            self.momentum = self.momentum_original * (self.momentum_decay**(epoch//self.momentum_decay_step))
            if self.momentum < 0.01:
                self.momentum = 0.01

        return F.batch_norm(
            input, self.running_mean, self.running_var, self.weight, self.bias,
            self.training, self.momentum, self.eps)


class MyLinear(nn.Module):
    def __init__(self, in_features, out_features, activation=None, normalization=None, momentum=0.1, bn_momentum_decay_step=None, bn_momentum_decay=1):
        super(MyLinear, self).__init__()
        self.activation = activation
        self.normalization = normalization

        self.linear = nn.Linear(in_features, out_features, bias=True)
        if self.normalization == 'batch':
            self.norm = MyBatchNorm1d(out_features, momentum=momentum, affine=True, momentum_decay_step=bn_momentum_decay_step, momentum_decay=bn_momentum_decay)
        elif self.normalization == 'instance':
            self.norm = nn.InstanceNorm1d(out_features, momentum=momentum, affine=True)
        if self.activation == 'relu':
            self.act = nn.ReLU()
        elif 'elu' == activation:
            self.act = nn.ELU(alpha=1.0)
        elif 'swish' == self.activation:
            self.act = Swish()
        elif 'leakyrelu' == self.activation:
            self.act = nn.LeakyReLU(0.1)

        self.weight_init()

    def weight_init(self):
        for m in self.modules():
            if isinstance(m, nn.Linear) :
                n = m.in_features
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.fill_(0)
            elif isinstance(m, MyBatchNorm1d) or isinstance(m, nn.InstanceNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x, epoch=None):
        x = self.linear(x)
        if self.normalization=='batch':
            x = self.norm(x, epoch)
        elif self.normalization is not None:
            x = self.norm(x)

        if self.activation is not None:
            x = self.act(x)

        return x


class MyConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True, activation=None, momentum=0.1, normalization=None, bn_momentum_decay_step=None, bn_momentum_decay=1):
        super(MyConv2d, self).__init__()
        self.activation = activation
        self.normalization = normalization

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        if self.normalization == 'batch':
            self.norm = MyBatchNorm2d(out_channels, momentum=momentum, affine=True, momentum_decay_step=bn_momentum_decay_step, momentum_decay=bn_momentum_decay)
        elif self.normalization == 'instance':
            self.norm = nn.InstanceNorm2d(out_channels, momentum=momentum, affine=True)
        if self.activation == 'relu':
            self.act = nn.ReLU()
        elif self.activation == 'elu':
            self.act = nn.ELU(alpha=1.0)
        elif 'swish' == self.activation:
            self.act = Swish()
        elif 'leakyrelu' == self.activation:
            self.act = nn.LeakyReLU(0.1)

        self.weight_init()

    def weight_init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.fill_(0)
            elif isinstance(m, MyBatchNorm2d) or isinstance(m, nn.InstanceNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x, epoch=None):
        x = self.conv(x)
        if self.normalization=='batch':
            x = self.norm(x, epoch)
        elif self.normalization is not None:
            x = self.norm(x)

        if self.activation is not None:
            x = self.act(x)
        return x





class EquivariantLayer(nn.Module):
    def __init__(self, num_in_channels, num_out_channels, activation='relu', normalization=None, momentum=0.1, bn_momentum_decay_step=None, bn_momentum_decay=1):
        super(EquivariantLayer, self).__init__()

        self.num_in_channels = num_in_channels
        self.num_out_channels = num_out_channels
        self.activation = activation
        self.normalization = normalization

        self.conv = nn.Conv1d(self.num_in_channels, self.num_out_channels, kernel_size=1, stride=1, padding=0)

        if 'batch' == self.normalization:
            self.norm = MyBatchNorm1d(self.num_out_channels, momentum=momentum, affine=True, momentum_decay_step=bn_momentum_decay_step, momentum_decay=bn_momentum_decay)
        elif 'instance' == self.normalization:
            self.norm = nn.InstanceNorm1d(self.num_out_channels, momentum=momentum, affine=True)

        if 'relu' == self.activation:
            self.act = nn.ReLU()
        elif 'elu' == self.activation:
            self.act = nn.ELU(alpha=1.0)
        elif 'swish' == self.activation:
            self.act = Swish()
        elif 'leakyrelu' == self.activation:
            self.act = nn.LeakyReLU(0.1)


        self.weight_init()

    def weight_init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                n = m.kernel_size[0] * m.in_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.fill_(0)
            elif isinstance(m, MyBatchNorm1d) or isinstance(m, nn.InstanceNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x, epoch=None):
        # x is NxK, x_max is 1xK
        # x_max, _ = torch.max(x, 0, keepdim=True)
        # y = self.conv(x - x_max.expand_as(x))
        y = self.conv(x)

        if self.normalization=='batch':
            y = self.norm(y, epoch)
        elif self.normalization is not None:
            y = self.norm(y)

        if self.activation is not None:
            y = self.act(y)

        return y


class PointNet(nn.Module):
    def __init__(self, in_channels, out_channels_list, activation, normalization, momentum=0.1, bn_momentum_decay_step=None, bn_momentum_decay=1):
        super(PointNet, self).__init__()

        self.layers = nn.ModuleList()
        previous_out_channels = in_channels
        for i, c_out in enumerate(out_channels_list):
            if i != len(out_channels_list)-1:
                self.layers.append(EquivariantLayer(previous_out_channels, c_out, activation, normalization,
                                                    momentum, bn_momentum_decay_step, bn_momentum_decay))
            else:
                self.layers.append(EquivariantLayer(previous_out_channels, c_out, None, None))
            previous_out_channels = c_out

    def forward(self, x, epoch=None):
        for layer in self.layers:
            x = layer(x, epoch)
        return x


class PointResNet(nn.Module):
    def __init__(self, in_channels, out_channels_list, activation, normalization, momentum=0.1, bn_momentum_decay_step=None, bn_momentum_decay=1):
        '''
        in -> out[0]
        out[0] -> out[1]             ----
        out[1] -> out[2]                |
             ... ...                    |
        out[k-2]+out[1] -> out[k-1]  <---
        :param in_channels:
        :param out_channels_list:
        :param activation:
        :param normalization:
        :param momentum:
        :param bn_momentum_decay_step:
        :param bn_momentum_decay:
        '''
        super(PointResNet, self).__init__()
        self.out_channels_list = out_channels_list

        self.layers = nn.ModuleList()
        previous_out_channels = in_channels
        for i, c_out in enumerate(out_channels_list):
            if i != len(out_channels_list)-1:
                self.layers.append(EquivariantLayer(previous_out_channels, c_out, activation, normalization,
                                                    momentum, bn_momentum_decay_step, bn_momentum_decay))
            else:
                self.layers.append(EquivariantLayer(previous_out_channels+out_channels_list[0], c_out, None, None))
            previous_out_channels = c_out

    def forward(self, x, epoch=None):
        '''
        :param x: BxCxN
        :param epoch: None or number of epoch, for BN decay.
        :return:
        '''
        layer0_out = self.layers[0](x, epoch)  # BxCxN
        for l in range(1, len(self.out_channels_list)-1):
            if l == 1:
                x_tmp = self.layers[l](layer0_out, epoch)
            else:
                x_tmp = self.layers[l](x_tmp, epoch)
        layer_final_out = self.layers[len(self.out_channels_list)-1](torch.cat((layer0_out, x_tmp), dim=1), epoch)
        return layer_final_out


class classification_net(nn.Module):
     """docstring for edge_conv_model"""
     def __init__(self, k=20):
         super(classification_net, self).__init__()

         self.conv1 = nn.Conv2d(6, 64, kernel_size=1)
         self.conv2 = nn.Conv2d(128, 64, kernel_size=1)
         self.conv3 = nn.Conv2d(128, 64, kernel_size=1)
         self.conv4 = nn.Conv2d(128, 128, kernel_size=1)
         self.conv5 = nn.Conv2d(320, 1024, kernel_size=1)
         
         self.bn1 = nn.BatchNorm2d(64)
         self.bn2 = nn.BatchNorm2d(64)
         self.bn3 = nn.BatchNorm2d(64)
         self.bn4 = nn.BatchNorm2d(128)
         self.bn5 = nn.BatchNorm2d(1024)
         
         self.bn6 = nn.BatchNorm1d(512)
         self.bn7 = nn.BatchNorm1d(256)

         #self.fc1 = nn.Linear(1024, 512)
         self.fc1 = nn.Linear(1408,512)
         self.fc2 = nn.Linear(512, 256)
         self.fc3 = nn.Linear(256, 40)

         self.dropout = nn.Dropout(p=0.3)

         self.input_transform = input_transform_net()
         self.k = k
         self.first_pointnet = PointResNet(3, [64, 128, 256, 384], activation='relu', normalization='batch',
                                              momentum=0.1, bn_momentum_decay_step=None, bn_momentum_decay=0.6)
        
     

     def forward(self, point_cloud):
         print(point_cloud.size())
         point_featrue = self.first_pointnet(point_cloud)
         batch_size, num_point,_ = point_cloud.size()
        
         dist_mat = pairwise_distance(point_cloud)
         nn_idx = knn(dist_mat, k=self.k)
         edge_feat = get_edge_feature(point_cloud, nn_idx=nn_idx, k=self.k)

         edge_feat = edge_feat.permute(0,3,1,2)
         # point_cloud = point_cloud.permute(0,2,1)

         transform_mat = self.input_transform(edge_feat)



         point_cloud_transformed = torch.bmm(point_cloud, transform_mat)
         dist_mat = pairwise_distance(point_cloud_transformed)
         nn_idx = knn(dist_mat, k=self.k)
         edge_feat = get_edge_feature(point_cloud_transformed, nn_idx=nn_idx, k=self.k)

         edge_feat = edge_feat.permute(0,3,1,2)


         net = self.bn1(F.relu(self.conv1(edge_feat)))
         net,_ = torch.max(net, dim=-1, keepdim=True)
         net1 = net


         net = net.permute(0,2,3,1)

         dist_mat = pairwise_distance(net)
         nn_idx = knn(dist_mat, k=self.k)
         edge_feat = get_edge_feature(net, nn_idx=nn_idx, k=self.k)
         edge_feat = edge_feat.permute(0,3,1,2)

         net = self.bn2(F.relu(self.conv2(edge_feat)))
         net,_ = torch.max(net, dim=-1, keepdim=True)
         net2 = net

         net = net.permute(0,2,3,1)

         dist_mat = pairwise_distance(net)
         nn_idx = knn(dist_mat, k=self.k)
         edge_feat = get_edge_feature(net, nn_idx=nn_idx, k=self.k)

         edge_feat = edge_feat.permute(0,3,1,2)


         net = self.bn3(F.relu(self.conv3(edge_feat)))
         net,_ = torch.max(net, dim=-1, keepdim=True)
         net3 = net

         net = net.permute(0,2,3,1)


         dist_mat = pairwise_distance(net)
         nn_idx = knn(dist_mat, k=self.k)
         edge_feat = get_edge_feature(net, nn_idx=nn_idx, k=self.k)

         edge_feat = edge_feat.permute(0,3,1,2)


         net = self.bn4(F.relu(self.conv4(edge_feat)))
         net,_ = torch.max(net, dim=-1, keepdim=True)
         net4 = net
         # import pdb
         # pdb.set_trace()

         net = self.bn5(F.relu(self.conv5(torch.cat((net1, net2, net3, net4), 1))))
         net,_ = torch.max(net, dim=2, keepdim=True)

         net = net.view(batch_size, -1)

         net = torch.cat(net,point_featrue)

         net = self.bn6(F.relu(self.fc1(net)))
         net = self.dropout(net)
         net = self.bn7(F.relu(self.fc2(net)))
         net = self.dropout(net)
         net = self.fc3(net)

         return net