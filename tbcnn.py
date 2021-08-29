import torch
import torch.nn as nn
import dgl
import dgl.function as fn
from dgl.nn import GlobalAttentionPooling, MaxPooling, AvgPooling
import math
from copy import deepcopy
from random import randint, choice
import sys
from config import myConfig, my_config

SQRT_MIN_VALUE = 1e-10

def squash(x, dim=-1):
    squared_norm = (x ** 2).sum(dim=dim, keepdim=True)
    scale = squared_norm / (1 + squared_norm) / \
        (torch.sqrt(squared_norm + SQRT_MIN_VALUE) + 1e-8)
    out = scale * x
    return out 


class TreeCapsClassifier(nn.Module):
    def __init__(self, x_size, h_size, dropout, n_classes, vocab_size, num_layers=4, a=my_config.model['a'], b=my_config.model['b'], routing_iter=my_config.model['routing_iter'], device=torch.device('cuda')):
        super(TreeCapsClassifier, self).__init__()
        type_vocabsize, token_vocabsize = vocab_size
        self.x_size = x_size
        self.h_size = h_size
        self.dropout = torch.nn.Dropout(dropout)
        self.layers = nn.ModuleList(TBCNNCell(x_size, h_size)
                                    for _ in range(num_layers))
        self.num_layers = num_layers
        self.token_embeddings = nn.Embedding(token_vocabsize, int(x_size/2))
        self.type_embeddings = nn.Embedding(type_vocabsize, x_size - int(x_size/2))

        self.Dcc = my_config.model['Dcc']
        self.a = a
        self.b = b
        self.routing_iter = routing_iter
        self.device = device
        self.n_classes = n_classes

        self.pvc_caps1_num_caps = int(
            self.num_layers * self.h_size / self.num_layers) * self.a

        self.Wjm = nn.Parameter(torch.Tensor(
            self.pvc_caps1_num_caps, self.Dcc ,self.n_classes, self.num_layers), requires_grad=True)
        self.Wjm.data.normal_(0, 0.01)
        # self.Wjm.data.uniform_(-0.1, 0.1)

        self.classifier = nn.Linear(self.a * self.num_layers, n_classes)



    def forward(self, batch, root_ids=None):
        batch.ndata['h'] = torch.cat([self.type_embeddings(
            batch.ndata['type']), self.token_embeddings(batch.ndata['token'])], dim=1)
        #tbcnn encoding
        layer_feats = []
        for i in range(self.num_layers):
            batch.update_all(message_func=self.layers[i].message_func,
                             reduce_func=self.layers[i].reduce_func,
                             apply_node_func=self.layers[i].apply_node_func)
            layer_feat = batch.ndata['h']
            layer_feats.append(layer_feat)
        # print(batch.ndata['h'].size())  # torch.Size([2465, 256])
        
        # size: batch_nodes, x_size, num_layers
        layer_feats = torch.stack(layer_feats, dim=-1)
        # print(layer_feats.size())  # torch.Size([2465, 256, 1])

        #primary variable capsules
        numnodes = batch.batch_num_nodes()
        numnodes_feats = numnodes*self.x_size
        numnodes_feats = numnodes_feats.tolist()
        numnodes = numnodes.tolist()
        # tensor([236, 163, 181, 438, 769, 203, 205, 270], device='cuda:0') [60416, 41728, 46336, 112128, 196864, 51968, 52480, 69120]
        # print(numnodes, numnodes_feats)
        # all_capsules = layer_feats.view(-1, self.num_layers)
        

        # print(all_capsules.size())  # torch.Size([631040, 1])
        # size: batch_nodes * x_size, num_layers
        # primary_capsules = squash(all_capsules)
        # print(primary_capsules.size())  # torch.Size([631040, 1]) 
        # Npvc * Dpvc
        # Dpvc = num_layers of tbcnn
        # Npvc = node_num * embedding_size

        #graphs_feat=layer_feats.split(numnodes)

        #secondary capsule: Variable-to-Static Routing
        # only used for sorting, sqrt() can be omitted
        # capsule_l2norms = (primary_capsules ** 2).sum(dim=1,
        #                                               keepdim=False).sqrt()
        # print(capsule_l2norms.size())

        primary_capsules = layer_feats
        # (batch_size * num_nodes) * h_size * num_layers
        # print(primary_capsules.shape)

        each_pvc = primary_capsules.split(numnodes) 
        u = each_pvc
        # print(len(u)) # batch_size = 8
        out_SC = []
        for uu in u:
            # num_nodes * h_size * num_layers
            # for each tree-capsule in batch
            SC = self.new_vts_routing(uu)
            out_SC.append(SC)
            # uu_l2 = (uu ** 2).sum(dim=1, keepdim=False).sqrt()
            # print(uu_l2.size())
            # uu_l2_topk_loc = uu_l2.topk(self.a, dim=-1)[1]
            # print(uu_l2_topk_loc)
            # print(uu.shape)
            # # init_uu = uu.gather(dim=0, index=uu_l2_topk_loc.view(-1, 1)) # how to use gather???
            
            # # init the output of SC
            # vj = uu[uu_l2_topk_loc]
            # print(vj.shape)
            # # print(vj)
            # alpha = torch.zero(self.a, self.b)
        out_SC = torch.stack(out_SC, dim=0)
        # size: batch_size * Npvc * m
        # print(out_SC.size())
        # print(out_SC.shape)
        # print(out_SC)

        # batch_logit = self.classifier(out_SC.view(-1, self.a * self.num_layers))
        # batch_soft_logit = torch.softmax(batch_logit, dim=-1)
        # return batch_logit, batch_soft_logit
        
        out_CC = self.new_dynamic_routing(out_SC)
        # print(out_CC.shape)
        # print(out_CC)
        out_CC_l2 = (out_CC ** 2).sum(dim=-1, keepdim=False)
        # print(out_CC_l2)
        out_CC_l2 = torch.sqrt(out_CC_l2 + SQRT_MIN_VALUE)
        

        # print(out_CC.size())
        # print(out_CC[0])
        batch_logit = out_CC_l2
        batch_soft_logit = torch.softmax(batch_logit, dim=-1)
        return batch_logit, batch_logit

    def new_dynamic_routing(self, input):
        # n: Npvc
        # c: Dcc
        # s: n_classes
        # m: num_layers
        # b: batch
        # 
        v_m_j = torch.einsum('ncsm,bnm->bnsc', self.Wjm, input)
        v_m_j_stopped = v_m_j.detach()
        

        delta_IJ = torch.zeros(input.shape[0],self.pvc_caps1_num_caps, self.n_classes).to(self.device)
        for rout in range(self.routing_iter):
            gamma_IJ = torch.softmax(delta_IJ, dim=-1)

            if rout == self.routing_iter -1:
                s_J = torch.einsum('bns,bnsc->bsc', gamma_IJ, v_m_j)
                z_m = squash(s_J)
            else:
                # print(gamma_IJ.shape)
                # print(v_m_j_stopped.shape)
                s_J = torch.einsum('bns,bnsc->bsc', gamma_IJ, v_m_j_stopped)
                z_m = squash(s_J)
                # print("debug")
                # print(z_m.shape)
                # batch_size * n_classes
                # print(v_m_j_stopped.shape)
                # batch_size * a
                delta_IJ = torch.einsum(
                    'bnsc,bsc->bns', v_m_j_stopped, z_m) + delta_IJ
                # b * a * n_classes
                # print(rout)
                # print(z_m.shape)
                # print(z_m)
        return z_m

    def new_vts_routing(self, input):
        # num_nodes * h_size * num_layers
        num_outputs = int(self.num_layers * self.h_size / self.num_layers) * self.a
        alpha_IJ = torch.zeros(int(num_outputs / self.a * self.b), num_outputs).to(self.device)
        # print(input.shape)
        input_l2 = (input.view(input.shape[0],-1) ** 2).sum(dim=1, keepdim=False)
        # print(input_l2.shape)
        # input_l2 = torch.sqrt(input_l2 + 1e-8)
        # input_l2 = (input ** 2).sum(dim=1, keepdim=False).sqrt()
        # print(input_l2.shape)
        # print(self.b)
        input_l2_topb_loc = input_l2.topk(self.b, dim=-1)[1]
        u_i = input[input_l2_topb_loc]
        u_i = u_i.view(-1, self.num_layers)
        u_i = u_i.detach()  # ????

        input_l2_topa_loc = input_l2.topk(self.a, dim=-1)[1]
        v_j = input[input_l2_topa_loc]
        # print(v_j.shape)
        v_j = v_j.view(-1, self.num_layers)
        # print(u_i.shape)
        # print(v_j.shape)
        for rout in range(self.routing_iter):
            u_produce_v = torch.matmul(u_i, v_j.transpose(0, 1))
            # print(u_produce_v.shape)
            # print(alpha_IJ.shape)

            # print(u_produce_v.size())
            alpha_IJ = u_produce_v + alpha_IJ
            beta_IJ = torch.softmax(alpha_IJ, dim=-1)
            # print("beta")
            # print(beta_IJ.shape) # b*a
            v_j = torch.matmul(beta_IJ.transpose(0, 1), u_i)


        # print(v_j.shape)
        v_j = squash(v_j)
        # print(v_j.shape)
        # print(v_j)
        return v_j

    def vts_routing(self, input):
        alpha_IJ = torch.zeros(self.b, self.a).to(self.device)
        input_l2 = (input ** 2).sum(dim=1, keepdim=False)
        input_l2 = torch.sqrt(input_l2 + SQRT_MIN_VALUE)
        # input_l2 = (input ** 2).sum(dim=1, keepdim=False).sqrt()
        # print(input_l2.shape)
        # print(self.b)
        input_l2_topb_loc = input_l2.topk(self.b, dim=-1)[1]
        u_i = input[input_l2_topb_loc]
        u_i = u_i.detach() # ????

        input_l2_topa_loc = input_l2.topk(self.a, dim=-1)[1]
        v_j = input[input_l2_topa_loc]
        # print(u_i.shape)
        # print(v_j.shape)
        for rout in range(self.routing_iter):
            u_produce_v = torch.matmul(u_i, v_j.transpose(0,1))

            # print(u_produce_v.size())
            alpha_IJ = u_produce_v + alpha_IJ
            beta_IJ = torch.softmax(alpha_IJ, dim=-1)
            # print("beta")
            # print(beta_IJ.shape) # b*a
            v_j = torch.matmul(beta_IJ.transpose(0,1), u_i)


        
        v_j = squash(v_j)
        # print(v_j.shape)
        # print(v_j)
        return v_j

    def dynamic_routing(self, input):
        # print("debug")
        # print(input.shape)
        # print(input)
        # batch_size * a * m
        # print(self.Wjm.shape)
        # print(input.shape)
        v_m_j = torch.einsum('amc,bam->bac', self.Wjm, input)
        v_m_j = v_m_j.reshape(v_m_j.shape[0], self.a, self.n_classes, self.Dcc)
        # batch_Size * Dcc
        # v_m_j = torch.matmul(input, self.Wjm)
        # print(v_m_j.shape)

        # v_m_j_stopped = v_m_j
        v_m_j_stopped = v_m_j.detach()
        

        delta_IJ = torch.zeros(input.shape[0],self.a, self.n_classes).to(self.device)
        for rout in range(self.routing_iter):
            gamma_IJ = torch.softmax(delta_IJ, dim=-1)
            # print(gamma_IJ.shape)
            # bs * a * n_classes
            # print(v_m_j.shape)
            # bs * a
            
            # einsum:
            # b: batch_size
            # a: a
            # n: n_classes
            # c: Dcc
            if rout == self.routing_iter -1:
                s_J = torch.einsum('ban,banc->bnc',gamma_IJ, v_m_j)
                z_m = squash(s_J)
            else:
                # print(gamma_IJ.shape)
                # print(v_m_j_stopped.shape)
                s_J = torch.einsum('ban,banc->bnc',gamma_IJ, v_m_j_stopped)
                z_m = squash(s_J)
                # print("debug")
                # print(z_m.shape)
                # batch_size * n_classes
                # print(v_m_j_stopped.shape)
                # batch_size * a
                delta_IJ = torch.einsum('banc,bnc->ban', v_m_j_stopped, z_m) + delta_IJ
                # b * a * n_classes
                # print(rout)
                # print(z_m.shape)
                # print(z_m)
        return z_m
        




class TBCNNCell(torch.nn.Module):
    def __init__(self, x_size, h_size):
        super(TBCNNCell, self).__init__()
        self.W_left=nn.Parameter(torch.rand(x_size, h_size), requires_grad=True)
        self.W_right=nn.Parameter(torch.rand(x_size, h_size), requires_grad=True)
        self.W_top=nn.Parameter(torch.rand(x_size, h_size), requires_grad=True)
        self.b_conv = nn.Parameter(torch.zeros(1, h_size), requires_grad=True)

        self.W_left.data.uniform_(-0.1, 0.1)
        self.W_right.data.uniform_(-0.1, 0.1)
        self.W_top.data.uniform_(-0.1, 0.1)
        self.b_conv.data.uniform_(-0.1, 0.1)

    def message_func(self, edges):
        return {'h': edges.src['h']}

    def reduce_func(self, nodes):
        child_nums=nodes.mailbox['h'].size()[1]
        if child_nums==1:
            #W_child=(self.W_left+self.W_right)/2
            #c_s=torch.matmul(nodes.mailbox['h'],W_child)
            c_s=torch.matmul(nodes.mailbox['h'],self.W_left)
            children_state=c_s.squeeze(1)
            h=torch.relu(children_state+torch.matmul(nodes.data['h'],self.W_top)+self.b_conv)
        else:
            left_weight=[(child_nums-1-i)/(child_nums-1) for i in range(child_nums)]
            right_weight=[i/(child_nums-1) for i in range(child_nums)]
            left_weight=torch.tensor(left_weight).to(self.W_left.device)
            right_weight=torch.tensor(right_weight).to(self.W_left.device)

            child_h=nodes.mailbox['h']
            
            child_h_left=torch.matmul(child_h,self.W_left)
            left_weight=left_weight.unsqueeze(1).unsqueeze(0)
            child_h_left=child_h_left*left_weight

            child_h_right=torch.matmul(child_h,self.W_right)
            right_weight=right_weight.unsqueeze(1).unsqueeze(0)
            child_h_right=child_h_right*right_weight
            
            children_state=child_h_left+child_h_right
            children_state=children_state.sum(dim=1)
            h=torch.relu(children_state+torch.matmul(nodes.data['h'],self.W_top)+self.b_conv)
        '''child_nums=nodes.mailbox['h'].size()[1]
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
            h=torch.relu(children_state+torch.matmul(nodes.data['h'],self.W_top)+self.b_conv)'''
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
        # self.cell = TBCNNCell(x_size, h_size)
        self.num_layers = num_layers

        self.type_embeddings = nn.Embedding(vocab_size[0], int(x_size/2))
        self.token_embedding = nn.Embedding(vocab_size[1], x_size - int(x_size/2))

        self.layers=nn.ModuleList(TBCNNCell(x_size, h_size) for _ in range(num_layers))
        # self.embeddings=nn.Embedding(vocab_size,x_size)
        self.classifier=nn.Linear(h_size,n_classes)
        #self.pooling=GlobalAttentionPooling(nn.Linear(h_size,1))
        self.pooling=MaxPooling()
        #self.pooling=AvgPooling()

    def forward(self, batch,root_ids=None):
        batch.ndata['h']=torch.cat([self.type_embeddings(batch.ndata['type']),self.token_embedding(batch.ndata['token'])],dim=-1)
        for i in range(self.num_layers):
            batch.update_all(message_func=self.layers[i].message_func,
                            reduce_func=self.layers[i].reduce_func,
                            apply_node_func=self.layers[i].apply_node_func)
        batch_pred=self.pooling(batch,batch.ndata['h'])
        batch_logit=self.classifier(batch_pred)
        batch_softlogit = torch.softmax(batch_logit, dim=-1)
        return batch_softlogit, batch_logit

def save_test_data():
    from config import my_config
    from main import CodeNetDataset
    print("start saving test data...")
    batch_size = 8
    train_dataset = CodeNetDataset(my_config.data['train_path'])
    train_dataloader = dgl.dataloading.pytorch.GraphDataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=my_config.data['num_workers'])
    
    for g, labels in train_dataloader:
        inputs, labels = g, labels
        inputs, labels = inputs.to(
            my_config.device), labels.to(my_config.device)
        print(inputs)
        torch.save({'inputs': inputs, 'labels': labels},
                   '/home/zhangkechi/workspace/dgl_tbcnn_edit/for_test.pkl')
        break

if __name__ == '__main__':
    from config import my_config
    all_data = torch.load('/home/zhangkechi/workspace/dgl_tbcnn_edit/for_test.pkl')
    # print(all_data['inputs'])
    # print(all_data['labels'])

    model = TreeCapsClassifier(my_config.model['x_size'], my_config.model['h_size'], my_config.model['dropout'],
                            my_config.task['num_classes'], my_config.task['vocab_size'], my_config.model['num_layers'])

    model.cuda()
    # print(model)
    t = model(all_data['inputs'])
    print(t)
    print(t[0].shape)
    print(t[1].shape)


