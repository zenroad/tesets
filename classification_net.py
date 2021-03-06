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


class STN3d(nn.Module):
    def __init__(self, num_points = 10000):
        super(STN3d, self).__init__()
        self.num_points = num_points
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.mp1 = torch.nn.MaxPool1d(num_points)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 9)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)


    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.mp1(x)
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = Variable(torch.from_numpy(np.array([1,0,0,0,1,0,0,0,1]).astype(np.float32))).view(1,9).repeat(batchsize,1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, 3, 3)
        return x


class PointNetfeat(nn.Module):
    def __init__(self, num_points = 10000, global_feat = True):
        super(PointNetfeat, self).__init__()
        self.stn = STN3d(num_points = num_points)
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.mp1 = torch.nn.MaxPool1d(num_points)
        self.num_points = num_points
        self.global_feat = global_feat
    def forward(self, x):
        batchsize = x.size()[0]
        trans = self.stn(x)
        x = x.transpose(2,1)
        x = torch.bmm(x, trans)
        x = x.transpose(2,1)
        x = F.relu(self.bn1(self.conv1(x)))
        pointfeat = x
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x = self.mp1(x)
        x = x.view(-1, 1024)
        #if self.global_feat:
        return x
'''
class PointNetfeat(nn.Module):
    def __init__(self, num_points = 2500):
        super(PointNetfeat, self).__init__()
        #self.stn = STN3d(num_points = num_points)
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.mp1 = torch.nn.MaxPool1d(num_points)
        self.num_points = num_points
    def forward(self, x):
        batchsize = x.size()[0]
        #trans = self.stn(x)
        #x = x.transpose(2,1)
        #x = torch.bmm(x, trans)
        #x = x.transpose(2,1)
        x = F.relu(self.bn1(self.conv1(x)))
        #pointfeat = x
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x = self.mp1(x)
        x = x.view(-1, 1024)
        #if self.global_feat:
        return x
'''

class classification_net(nn.Module):
     """docstring for edge_conv_model"""
     def __init__(self, k=30):
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

         self.fc1 = nn.Linear(1024, 512)
         #self.fc1 = nn.Linear(2048,512)
         self.fc2 = nn.Linear(512, 256)
         self.fc3 = nn.Linear(256, 40)

         self.dropout = nn.Dropout(p=0.2)

         self.input_transform = input_transform_net()
         self.k = k

     def forward(self, point_cloud):
         batch_size, num_point,_ = point_cloud.size()
         #pointnetfeat = PointNetfeat(num_points = num_point)
         #pointnetfeat.cuda()
         
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
         
         #point_globle = pointnetfeat(point_cloud_transformed.permute(0,2,1))

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

         net = self.bn6(F.relu(self.fc1(net,)))
         #net = self.bn6(F.relu(self.fc1(torch.cat((net,point_globle),1))))
         net = self.dropout(net)
         net = self.bn7(F.relu(self.fc2(net)))
         #net = self.dropout(net)
         net = self.fc3(net)

         return net