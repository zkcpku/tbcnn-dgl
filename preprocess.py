import os
import re
import random
import json
import csv
import torch
import dgl
import dgl.function as fn
from anytree import AnyNode, RenderTree
from anytree.exporter import JsonExporter
from anytree.importer import JsonImporter
from collections import defaultdict, deque, Counter
import pickle
from tqdm import tqdm
# from torchtext.vocab import Vocab

from dgl import save_graphs, load_graphs
from dgl.data.utils import makedirs, save_info, load_info

pythonpath = '/home/zhangkechi/workspace/data/codenet/Project_CodeNet_Python800_spts/'
c1000path = '/home/zhangkechi/workspace/data/codenet/Project_CodeNet_C++1000_spts/'
c1400path = '/home/zhangkechi/workspace/data/codenet/Project_CodeNet_C++1400_spts/'
javapath = '/home/zhangkechi/workspace/data/codenet/Project_CodeNet_Java250_spts/'

datapath = c1000path
edgepath = datapath+'edge.csv'
labelpath = datapath+'graph-label.csv'
nodepath = datapath+'node-feat.csv'
edgenumpath = datapath+'num-edge-list.csv'
nodenumpath = datapath+'num-node-list.csv'
nodedfs = datapath+'node_dfs_order.csv'
nodedepth = datapath+'node_depth.csv'


def get_spt_dataset_dict(bidirection=False, virtual=False):
    numnodes = []
    numedges = []
    nodefeats = []
    token_ids = []
    rule_ids = []
    edges = []
    labels = []
    nodedepths = []
    nodedfss = []
    token_vocabsize = 0
    type_vocabsize = 0
    with open(nodenumpath) as f:
        f_csv = csv.reader(f)
        for row in f_csv:
            numnodes.append(int(row[0]))
    with open(edgenumpath) as f:
        f_csv = csv.reader(f)
        for row in f_csv:
            numedges.append(int(row[0]))
    with open(labelpath) as f:
        f_csv = csv.reader(f)
        for row in f_csv:
            labels.append(int(row[0]))
    print(len(numnodes), len(numedges), len(labels))
    '''nodesum=0
    edgesum=0
    for i in range(len(numnodes)):
        nodesum+=numnodes[i]
        edgesum+=numedges[i]
    print(nodesum,edgesum)'''
    with open(edgepath) as f:
        f_csv = csv.reader(f)
        for row in f_csv:
            source, target = row
            source, target = int(source), int(target)
            edges.append([source, target])
    print(len(edges))

    with open(nodedepth) as f:
        f_csv = csv.reader(f)
        for row in f_csv:
            node_depth = row
            node_depth = int(node_depth[0])
            nodedepths.append(node_depth)
    print(len(nodedepths))

    with open(nodedfs) as f:
        f_csv = csv.reader(f)
        for row in f_csv:
            node_dfs = row
            node_dfs = int(node_dfs[0])
            nodedfss.append(node_dfs)
    print(len(nodedfss))

    with open(nodepath) as f:
        f_csv = csv.reader(f)
        for row in f_csv:
            is_token, token_type, rule_type, is_reserved = row
            token_type, rule_type = int(token_type), int(rule_type)
            if token_type > token_vocabsize:
                token_vocabsize = token_type
            if rule_type > type_vocabsize:
                type_vocabsize = rule_type
            nodefeats.append([token_type, rule_type])
            token_ids.append(token_type)
            rule_ids.append(rule_type)
    print(len(nodefeats))

    all_graphdata = []
    graph_nodestart = 0
    graph_edgestart = 0
    for i in range(len(labels)):
        num_node, num_edge, graph_label = numnodes[i], numedges[i], labels[i]
        #print(num_node,num_edge,graph_label)
        graph_edge = edges[graph_edgestart:graph_edgestart+num_edge]
        #print(graph_edge)
        targets, sources = list(zip(*graph_edge))  # from child to parent
        if bidirection == True:  # bidirectional graph for gnn
            targets, sources = targets+sources, sources+targets
        targets, sources = torch.tensor(targets), torch.tensor(sources)
        #print(targets,sources)
        # g = dgl.graph((sources, targets))
        g = {'ndata': {}}
        graph_tokens = torch.tensor(
            token_ids[graph_nodestart:graph_nodestart+num_node])
        graph_rules = torch.tensor(
            rule_ids[graph_nodestart:graph_nodestart+num_node])

        graph_depths = torch.tensor(
            nodedepths[graph_nodestart:graph_nodestart+num_node])

        graph_dfss = torch.tensor(
            nodedfss[graph_nodestart:graph_nodestart+num_node])

        g['ndata']['token'] = graph_tokens
        g['ndata']['type'] = graph_rules
        g['ndata']['depth'] = graph_depths
        g['ndata']['dfs'] = graph_dfss
        graph_nodestart += num_node
        graph_edgestart += num_edge
        all_graphdata.append([{'graph': g, "edge": graph_edge}, graph_label])

    #simple data split
    print(len(all_graphdata))
    trainset, devset, testset = [[], [], []]
    for i in range(len(all_graphdata)):
        if i % 5 == 3:
            devset.append(all_graphdata[i])
        elif i % 5 == 4:
            testset.append(all_graphdata[i])
        else:
            trainset.append(all_graphdata[i])
    print(len(trainset), len(devset), len(testset))
    token_vocabsize += 2
    type_vocabsize += 2
    print(token_vocabsize, type_vocabsize)
    return trainset, devset, testset, token_vocabsize, type_vocabsize


def get_spt_dataset(bidirection=False, virtual=False):
    numnodes = []
    numedges = []
    nodefeats = []
    token_ids = []
    rule_ids = []
    edges = []
    labels = []
    nodedepths = []
    nodedfss = []
    token_vocabsize = 0
    type_vocabsize = 0
    with open(nodenumpath) as f:
        f_csv = csv.reader(f)
        for row in f_csv:
            numnodes.append(int(row[0]))
    with open(edgenumpath) as f:
        f_csv = csv.reader(f)
        for row in f_csv:
            numedges.append(int(row[0]))
    with open(labelpath) as f:
        f_csv = csv.reader(f)
        for row in f_csv:
            labels.append(int(row[0]))
    print(len(numnodes), len(numedges), len(labels))
    '''nodesum=0
    edgesum=0
    for i in range(len(numnodes)):
        nodesum+=numnodes[i]
        edgesum+=numedges[i]
    print(nodesum,edgesum)'''
    with open(edgepath) as f:
        f_csv = csv.reader(f)
        for row in f_csv:
            source, target = row
            source, target = int(source), int(target)
            edges.append([source, target])
    print(len(edges))

    with open(nodedepth) as f:
        f_csv = csv.reader(f)
        for row in f_csv:
            node_depth = row
            node_depth = int(node_depth[0])
            nodedepths.append(node_depth)
    print(len(nodedepths))

    with open(nodedfs) as f:
        f_csv = csv.reader(f)
        for row in f_csv:
            node_dfs = row
            node_dfs = int(node_dfs[0])
            nodedfss.append(node_dfs)
    print(len(nodedfss))

    with open(nodepath) as f:
        f_csv = csv.reader(f)
        for row in f_csv:
            is_token, token_type, rule_type, is_reserved = row
            token_type, rule_type = int(token_type), int(rule_type)
            if token_type > token_vocabsize:
                token_vocabsize = token_type
            if rule_type > type_vocabsize:
                type_vocabsize = rule_type
            nodefeats.append([token_type, rule_type])
            token_ids.append(token_type)
            rule_ids.append(rule_type)
    print(len(nodefeats))

    all_graphdata = []
    graph_labels = []
    graph_nodestart = 0
    graph_edgestart = 0
    for i in range(len(labels)):
        num_node, num_edge, graph_label = numnodes[i], numedges[i], labels[i]
        #print(num_node,num_edge,graph_label)
        graph_edge = edges[graph_edgestart:graph_edgestart+num_edge]
        #print(graph_edge)
        targets, sources = list(zip(*graph_edge))  # from child to parent
        if bidirection == True:  # bidirectional graph for gnn
            targets, sources = targets+sources, sources+targets
        targets, sources = torch.tensor(targets), torch.tensor(sources)
        #print(targets,sources)
        g = dgl.graph((sources, targets))
        graph_tokens = torch.tensor(
            token_ids[graph_nodestart:graph_nodestart+num_node])
        graph_rules = torch.tensor(
            rule_ids[graph_nodestart:graph_nodestart+num_node])

        graph_depths = torch.tensor(
            nodedepths[graph_nodestart:graph_nodestart+num_node])

        graph_dfss = torch.tensor(
            nodedfss[graph_nodestart:graph_nodestart+num_node])

        g.ndata['token'] = graph_tokens
        g.ndata['type'] = graph_rules
        # g.ndata['depth'] = graph_depths
        # g.ndata['dfs'] = graph_dfss
        graph_nodestart += num_node
        graph_edgestart += num_edge
        all_graphdata.append(g)
        graph_labels.append(graph_label)

    #simple data split
    print(len(all_graphdata))
    trainset, devset, testset = [[], [], []]
    trainlabel, devlabel, testlabel = [[], [], []]
    for i in range(len(all_graphdata)):
        if i % 5 == 3:
            devset.append(all_graphdata[i])
            devlabel.append(graph_labels[i])
        elif i % 5 == 4:
            testset.append(all_graphdata[i])
            testlabel.append(graph_labels[i])
        else:
            trainset.append(all_graphdata[i])
            trainlabel.append(graph_labels[i])

    print(len(trainset), len(devset), len(testset))
    token_vocabsize += 2
    type_vocabsize += 2
    print(token_vocabsize, type_vocabsize)
    return trainset, trainlabel, devset, devlabel, testset, testlabel, token_vocabsize, type_vocabsize

#get_spt_dataset()

def save_dataset(graph_path):
    trainset, trainlabel, devset, devlabel, testset, testlabel, token_vocabsize, type_vocabsize = get_spt_dataset()
    # import pdb;pdb.set_trace()
    save_graphs(graph_path + 'train.pkl', trainset,
                {'labels': torch.tensor(trainlabel)})
    save_graphs(graph_path + 'dev.pkl', devset,
                {'labels': torch.tensor(devlabel)})
    save_graphs(graph_path + 'test.pkl', testset,
                {'labels': torch.tensor(testlabel)})
    print({'token_vocabsize': token_vocabsize, 'type_vocabsize': type_vocabsize})
    save_info(graph_path + 'info.pkl', {'token_vocabsize': token_vocabsize, 'type_vocabsize':type_vocabsize})
    


def graph2tree(edge, token_tensor, type_tensor, depth_tensor, dfs_tensor):
    root = []
    children_dict = {}
    for e in edge:
        if e[0] in children_dict:
            children_dict[e[0]].append(e[1])
        else:
            children_dict[e[0]] = []
            children_dict[e[0]].append(e[1])

    def addNode(tree, this_id):
        tree.append({})
        this_token = token_tensor[this_id]
        this_type = type_tensor[this_id]
        this_depth = depth_tensor[this_id]
        this_dfs = dfs_tensor[this_id]

        tree[-1]['node'] = this_token
        tree[-1]['type'] = this_type
        tree[-1]['depth'] = this_depth
        tree[-1]['dfs'] = this_dfs
        tree[-1]['children'] = []

        if this_id not in children_dict:
            return
        for e in children_dict[this_id]:
            addNode(tree[-1]['children'], e)

    addNode(root, 0)

    return root


def getTree(data):
    edge = data[0]['edge']
    token_tensor = data[0]['graph'].ndata['token'].tolist()
    type_tensor = data[0]['graph'].ndata['type'].tolist()
    depth_tensor = data[0]['graph'].ndata['depth'].tolist()
    dfs_tensor = data[0]['graph'].ndata['dfs'].tolist()
    return graph2tree(edge, token_tensor, type_tensor, depth_tensor, dfs_tensor)
# t = getTree(trainset[0])[0]


def dataset_convert(dataset):
    output_dataset = []
    labels = []
    for e in tqdm(dataset):
        t = getTree(e)[0]
        l = str(e[1] + 1)
        d = {'tree': t, 'label': l}
        labels.append(l)
        output_dataset.append(d)
    return output_dataset, list(set(labels))


# all_children_num = []
# def num_children(tree):
#     all_children_num.append(len(tree['children']))
#     for e in tree['children']:
#         num_children(e)
# num_children(t)

if __name__ == '__main__':
    print(datapath)
    save_path = '/home/zhangkechi/workspace/dgl_tbcnn/data/Project_CodeNet_C++1000_spts/'
    save_dataset(save_path)
