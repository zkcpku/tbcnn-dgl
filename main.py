import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import random
from tqdm import tqdm
import pickle


from config import *
from tbcnn import *

from dgl.data import DGLDataset
from dgl import save_graphs, load_graphs
from dgl.data.utils import makedirs, save_info, load_info
from transformers import AdamW, get_linear_schedule_with_warmup


class CodeNetDataset(DGLDataset):
    def __init__(self,
                load_path,
                url=None,
                raw_dir=None,
                save_dir=None,
                force_reload=False,
                verbose=False):
        super(CodeNetDataset, self).__init__(name='codenet',
                                        url=url,
                                        raw_dir=raw_dir,
                                        save_dir=save_dir,
                                        force_reload=force_reload,
                                        verbose=verbose)
        self.load_path = load_path
        self.graphs, self.label_dict = load_graphs(load_path)
        self.labels = self.label_dict['labels']


    def download(self):
            # download raw data to local disk
        pass
    def process(self):
        # process raw data to graphs, labels, splitting masks
        pass

    def __getitem__(self, idx):
        # get one example by index
        return self.graphs[idx], self.labels[idx]

    def __len__(self):
        # number of data examples
        return len(self.graphs)


def set_seed(seed=36):
    random.seed(seed)
    os.environ['PYHTONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def train_iter(dataloader, model, criterion, optimizer, scheduler, epoch):
    model.train()
    bar = tqdm(total=len(dataloader))
    running_loss = 0.0
    for g, labels in dataloader:
        bar.update(1)
        inputs, labels = g, labels
        inputs, labels = inputs.to(
            my_config.device), labels.to(my_config.device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        softlogit, logits = model(inputs)
        loss = criterion(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), my_config.optim['max_grad_norm'])

        optimizer.step()
        optimizer.zero_grad()
        scheduler.step()
        running_loss += loss.item()
        bar.set_description("epoch {} loss {}".format(epoch, loss.item()))

    print('Loss: {}'.format(running_loss))


def test_iter(dataloader, model, criterion, epoch):
    model.eval()
    with torch.no_grad():
        bar = tqdm(total=len(dataloader))
        running_loss = 0.0
        all_labels = []
        all_preds = []
        for g, labels in dataloader:
            bar.update(1)
            inputs, labels = g, labels
            inputs, labels = inputs.to(
                my_config.device), labels.to(my_config.device)

            # forward + backward + optimize
            softlogit, logits = model(inputs)
            pred = logits.argmax(dim=1)
            all_labels.extend(labels.cpu().numpy().tolist())
            all_preds.extend(pred.cpu().numpy().tolist())
            loss = criterion(logits, labels)
            running_loss += loss.item()
        # import pdb;pdb.set_trace()
        # print(len(all_preds))
        # print(len(all_labels))
        print("Epoch: ", epoch)
        print("Acc: ", sum(np.array(all_preds) ==
                        np.array(all_labels)) / len(all_preds))
        print('Loss: {}'.format(running_loss))
        return sum(np.array(all_preds) == np.array(all_labels)) / len(all_preds), running_loss


def main():
    set_seed(my_config.seed)

    model = TBCNNClassifier(my_config.model['x_size'],my_config.model['h_size'], my_config.model['dropout'], my_config.task['num_classes'], my_config.task['vocab_size'], my_config.model['num_layers'])
    model.to(my_config.device)

    train_dataset = CodeNetDataset(my_config.data['train_path'])
    train_dataloader = dgl.dataloading.pytorch.GraphDataLoader(
        train_dataset, batch_size=my_config.data['batch_size'], shuffle=True, num_workers=my_config.data['num_workers'])
    
    test_dataset = CodeNetDataset(my_config.data['test_path'])
    test_dataloader = dgl.dataloading.pytorch.GraphDataLoader(
        test_dataset, batch_size=my_config.data['batch_size'], shuffle=False, num_workers=my_config.data['num_workers'])

    dev_dataset = CodeNetDataset(my_config.data['dev_path'])
    dev_dataloader = dgl.dataloading.pytorch.GraphDataLoader(
        dev_dataset, batch_size=my_config.data['batch_size'], shuffle=False, num_workers=my_config.data['num_workers'])


    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': my_config.optim['weight_decay']},
        {'params': [p for n, p in model.named_parameters() if any(
            nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters,
                      lr=my_config.optim['lr'], eps=my_config.optim['adam_epsilon'])

    max_steps = my_config.num_epochs * \
        len(train_dataloader) // my_config.data['batch_size']
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=max_steps*0.1,
                                                num_training_steps=max_steps)

    checkpoint_last = os.path.join(
        my_config.path['save'], 'checkpoint-last')
    scheduler_last = os.path.join(checkpoint_last, 'scheduler.pt')
    optimizer_last = os.path.join(checkpoint_last, 'optimizer.pt')
    if os.path.exists(scheduler_last):
        scheduler.load_state_dict(torch.load(scheduler_last))
    if os.path.exists(optimizer_last):
        optimizer.load_state_dict(torch.load(optimizer_last))

    
    # model.train()
    model.zero_grad()
    criterion = nn.CrossEntropyLoss()

    # loop over the dataset multiple times
    csv_log = open(my_config.path['save'] + '/acc.log', "w")
    csv_log.write('Epoch,Accuracy,Loss,\n')
    for epoch in range(my_config.num_epochs):
        train_iter(train_dataloader, model, criterion, optimizer, scheduler, epoch)
        torch.cuda.empty_cache()
        acc, loss = test_iter(train_dataloader, model, criterion, epoch)
        csv_log.write(str(epoch)+','+str(acc)+','+str(loss)+'\n')
        torch.cuda.empty_cache()
        acc, loss = test_iter(dev_dataloader, model, criterion, epoch)
        csv_log.write(str(epoch)+','+str(acc)+','+str(loss)+'\n')
        torch.cuda.empty_cache()
        acc, loss = test_iter(test_dataloader, model, criterion, epoch)
        csv_log.write(str(epoch)+','+str(acc)+','+str(loss)+'\n')
        torch.cuda.empty_cache()
    print('Finished Training')
    csv_log.close()




if __name__ == '__main__':
    main()
