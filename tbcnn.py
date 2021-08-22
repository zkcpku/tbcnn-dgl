import torch
import torch.nn as nn
import dgl
import dgl.function as fn
from dgl.nn import GlobalAttentionPooling, MaxPooling, AvgPooling
import math
from copy import deepcopy
from random import randint, choice

class TBCNNCell(torch.nn.Module):
    def __init__(self, x_size, h_size):
        super(TBCNNCell, self).__init__()
        self.W_left=nn.Parameter(torch.rand(x_size, h_size), requires_grad=True)
        self.W_right=nn.Parameter(torch.rand(x_size, h_size), requires_grad=True)
        self.W_top=nn.Parameter(torch.rand(x_size, h_size), requires_grad=True)
        self.W_iou = torch.nn.Linear(x_size, 3 * h_size, bias=False)
        self.U_iou = torch.nn.Linear(h_size, 3 * h_size, bias=False)
        self.b_iou = torch.nn.Parameter(torch.zeros(1, 3 * h_size), requires_grad=True)
        self.b_conv = torch.nn.Parameter(torch.zeros(1, h_size), requires_grad=True)
        self.U_f = torch.nn.Linear(h_size, h_size)

    def message_func(self, edges):
        return {'h': edges.src['h']}

    def reduce_func(self, nodes):
        child_nums=nodes.mailbox['h'].size()[1]
        if child_nums==1:
            W_child=(self.W_left+self.W_right)/2
            c_s=torch.matmul(nodes.mailbox['h'],W_child)
            #c_s=torch.matmul(nodes.mailbox['h'],self.W_left)
            children_state=c_s.squeeze(1)
            h=torch.relu(children_state+torch.matmul(nodes.data['h'],self.W_top)+self.b_conv)
        else:
            W_children=[]
            for i in range(child_nums):
                W_children.append((i/(child_nums-1))*self.W_right+((child_nums-1-i)/(child_nums-1))*self.W_left)
            W_s=torch.stack(W_children,dim=0)
            child_h=nodes.mailbox['h'].unsqueeze(-2) #size(batch, child_nums, 1, x_size)
            # import pdb;pdb.set_trace()
            # print(child_h.shape)
            # print(W_s.shape)
            children_state=torch.matmul(child_h,W_s)
            children_state=children_state.squeeze(-2)
            children_state=children_state.sum(dim=1)
            h=torch.relu(children_state+torch.matmul(nodes.data['h'],self.W_top)+self.b_conv)
        return {'h': h}

    def apply_node_func(self, nodes):
        return {'h': nodes.data['h']}

class TBCNNClassifier(torch.nn.Module):
    def __init__(self,
                 x_size,
                 h_size,
                 dropout,
                 n_classes,
                 vocab_size,
                 num_layers=1):
        super(TBCNNClassifier, self).__init__()
        self.x_size = x_size
        self.dropout = torch.nn.Dropout(dropout)
        self.cell = TBCNNCell(x_size, h_size)
        self.num_layers = num_layers

        self.type_embeddings = nn.Embedding(vocab_size[0], int(x_size/2))
        self.token_embedding = nn.Embedding(vocab_size[1], x_size - int(x_size/2))

        self.classifier=nn.Linear(h_size,n_classes)
        self.pooling=GlobalAttentionPooling(nn.Linear(h_size,1))
        #self.pooling=AvgPooling()

    def forward(self, batch,root_ids=None):
        batch.ndata['h']=torch.cat([self.type_embeddings(batch.ndata['type']),self.token_embedding(batch.ndata['token'])],dim=-1)
        for i in range(self.num_layers):
            batch.update_all(message_func=self.cell.message_func,
                            reduce_func=self.cell.reduce_func,
                            apply_node_func=self.cell.apply_node_func)
        batch_pred=self.pooling(batch,batch.ndata['h'])
        batch_logit=self.classifier(batch_pred)
        batch_softlogit = torch.softmax(batch_logit, dim=-1)
        return batch_softlogit, batch_logit
