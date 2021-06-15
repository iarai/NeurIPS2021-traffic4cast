"""GCN implementation copied from https://github.com/mie-lab/traffic4cast-
Graph-ResNet/blob/master/models/graph_models.py with permission."""
#  Copyright 2021 Institute of Advanced Research in Artificial Intelligence (IARAI) GmbH.
#  IARAI licenses this file to You under the Apache License, Version 2.0
#  (the "License"); you may not use this file except in compliance with
#  the License. You may obtain a copy of the License at
#
#  http://www.apache.org/licenses/LICENSE-2.0
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import ChebConv  # noqa


class Kipfblock(torch.nn.Module):
    def __init__(self, n_input, n_hidden=64, K=8, p=0.5, bn=False):
        super(Kipfblock, self).__init__()
        self.conv1 = ChebConv(n_input, n_hidden, K=K)
        self.p = p
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.do_bn = bn
        if bn:
            self.bn = torch.nn.BatchNorm1d(n_hidden)

    def forward(self, x, edge_index):
        if self.do_bn:
            x = F.relu(self.bn(self.conv1(x, edge_index)))
        else:
            x = F.relu(self.conv1(x, edge_index))

        x = F.dropout(x, training=self.training, p=self.p)

        return x


class KipfNet(torch.nn.Module):
    def __init__(self, num_features, num_classes, nh1=64, K=8, K_mix=2, inout_skipconn=False):
        super(KipfNet, self).__init__()
        self.inout_skipconn = inout_skipconn
        self.Kipfblock1 = Kipfblock(n_input=num_features, n_hidden=nh1, K=K)

        if inout_skipconn:
            self.conv_mix = ChebConv(nh1 + num_features, num_classes, K=K_mix)
        else:
            self.conv_mix = ChebConv(nh1, num_classes, K=K_mix)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.Kipfblock1(x, edge_index)

        if self.inout_skipconn:
            x = torch.cat((x, data.x), 1)
            x = self.conv_mix(x, edge_index)
        else:
            x = self.conv_mix(x, edge_index)

        return x


class KipfNetd2(torch.nn.Module):
    def __init__(self, num_features, num_classes, nh1=2, nh2=2, K=2, K_mix=1, inout_skipconn=True):
        super(KipfNetd2, self).__init__()
        self.inout_skipconn = inout_skipconn
        self.Kipfblock1 = Kipfblock(n_input=num_features, n_hidden=nh1, K=K)
        self.Kipfblock2 = Kipfblock(n_input=nh1, n_hidden=nh2, K=K)

        if inout_skipconn:
            self.conv_mix = ChebConv(nh2 + num_features, num_classes, K=K_mix)
        else:
            self.conv_mix = ChebConv(nh2, num_classes, K=K_mix)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.Kipfblock1(x, edge_index)
        x = self.Kipfblock2(x, edge_index)

        if self.inout_skipconn:
            x = torch.cat((x, data.x), 1)
            x = self.conv_mix(x, edge_index)
        else:
            x = self.conv_mix(x, edge_index)

        return x


class KipfNet_resd2(torch.nn.Module):
    def __init__(self, num_features, num_classes, nh1=64, nh2=32, K=8, K_mix=2, inout_skipconn=True):
        super(KipfNet_resd2, self).__init__()
        self.inout_skipconn = inout_skipconn

        self.Kipfblock1 = Kipfblock(n_input=num_features, n_hidden=nh1, K=K)
        self.Kipfblock2 = Kipfblock(n_input=nh1, n_hidden=nh2, K=K)

        self.skip_project1 = ChebConv(in_channels=self.Kipfblock1.n_input, out_channels=self.Kipfblock1.n_hidden, K=1)

        self.skip_project2 = ChebConv(in_channels=self.Kipfblock2.n_input, out_channels=self.Kipfblock2.n_hidden, K=1)

        if inout_skipconn:
            self.conv_mix = ChebConv(nh2 + num_features, num_classes, K=K_mix)
        else:
            self.conv_mix = ChebConv(nh2, num_classes, K=K_mix)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.Kipfblock1(x, edge_index) + self.skip_project1(x, edge_index)
        x = self.Kipfblock2(x, edge_index) + self.skip_project2(x, edge_index)

        if self.inout_skipconn:
            x = torch.cat((x, data.x), 1)
            x = self.conv_mix(x, edge_index)
        else:
            x = self.conv_mix(x, edge_index)

        return x


class Graph_resnet(torch.nn.Module):
    def __init__(self, num_features, num_classes, nh=38, K=6, K_mix=2, inout_skipconn=True, depth=3, p=0.5, bn=False):

        super(Graph_resnet, self).__init__()
        self.inout_skipconn = inout_skipconn
        self.depth = depth

        self.Kipfblock_list = nn.ModuleList()
        self.skipproject_list = nn.ModuleList()

        if isinstance(nh, list):
            # if you give every layer a differnt number of channels
            # you need one number of channels for every layer!
            assert len(nh) == depth

        else:
            channels = nh
            nh = []
            for _ in range(depth):
                nh.append(channels)

        for i in range(depth):
            if i == 0:
                self.Kipfblock_list.append(Kipfblock(n_input=num_features, n_hidden=nh[0], K=K, p=p, bn=bn))
                self.skipproject_list.append(ChebConv(num_features, nh[0], K=1))
            else:
                self.Kipfblock_list.append(Kipfblock(n_input=nh[i - 1], n_hidden=nh[i], K=K, p=p, bn=bn))
                self.skipproject_list.append(ChebConv(nh[i - 1], nh[i], K=1))

        if inout_skipconn:
            self.conv_mix = ChebConv(nh[-1] + num_features, num_classes, K=K_mix)
        else:
            self.conv_mix = ChebConv(nh[-1], num_classes, K=K_mix)

    def forward(self, data, **kwargs):

        x, edge_index = data.x, data.edge_index

        for i in range(self.depth):
            x = self.Kipfblock_list[i](x, edge_index) + self.skipproject_list[i](x, edge_index)

        if self.inout_skipconn:
            x = torch.cat((x, data.x), 1)
            x = self.conv_mix(x, edge_index)
        else:
            x = self.conv_mix(x, edge_index)

        return x


class KipfNet_orig(torch.nn.Module):
    def __init__(self, num_features, num_classes, nh1=64, K=8):
        super(KipfNet_orig, self).__init__()
        self.conv1 = ChebConv(num_features, nh1, K=K)
        self.conv2 = ChebConv(nh1, num_classes, K=K)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return x
