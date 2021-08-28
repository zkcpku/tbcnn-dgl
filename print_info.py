import pickle
import sys
import os


with open('/home/zhangkechi/workspace/dgl_tbcnn/data/Project_CodeNet_C++1000_spts/info.pkl','rb') as f:
    info = pickle.load(f)

print(info)