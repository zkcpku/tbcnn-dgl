class myConfig():
    def __init__(self) -> None:
        self.device = 'cuda'
        self.seed = 36
        self.batch_size = 256
        self.num_workers = 8

        self.model = {'x_size': 256, 'h_size': 256, 'dropout': 0.3, 'num_layers': 4, 'a': 20, 'b': 25, 'routing_iter': 3, 'Dcc': 16}
        # self.model['name'] = 'tbcnn'
        self.model['name'] = 'treecaps'
        
        self.task_name = 'java'
        if self.task_name == 'java':
            # java token_vocabsize, type_vocabsize = (115, 107)
            self.task = {'vocab_size': (107,115), 'num_classes': 250, 'task': 'java250'}
            self.data = {'train_path': '/home/zhangkechi/workspace/dgl_tbcnn/data/Project_CodeNet_Java250_spts/train.pkl',
                        'test_path': '/home/zhangkechi/workspace/dgl_tbcnn/data/Project_CodeNet_Java250_spts/test.pkl',
                        'dev_path': '/home/zhangkechi/workspace/dgl_tbcnn/data/Project_CodeNet_Java250_spts/dev.pkl',
                        'batch_size': self.batch_size,
                        'num_workers': self.num_workers}
            self.path = {
                'save': '/home/zhangkechi/workspace/dgl_tbcnn/save/Project_CodeNet_Java250_spts'}
        # python
        elif self.task_name == 'python':
            # python token_vocabsize, type_vocabsize: (100, 61)
            self.task = {'vocab_size': (61, 100), 'num_classes': 800, 'task': 'python800'}
            self.data = {'train_path': '/home/zhangkechi/workspace/dgl_tbcnn/data/Project_CodeNet_Python800_spts/train.pkl',
                        'test_path': '/home/zhangkechi/workspace/dgl_tbcnn/data/Project_CodeNet_Python800_spts/test.pkl',
                        'dev_path': '/home/zhangkechi/workspace/dgl_tbcnn/data/Project_CodeNet_Python800_spts/dev.pkl',
                        'batch_size': self.batch_size,
                        'num_workers': self.num_workers}
            self.path = {
                'save': '/home/zhangkechi/workspace/dgl_tbcnn/save/Project_CodeNet_Python800_spts'}
        elif self.task_name == 'c1000':
            # c++ 1000
            # {'token_vocabsize': 156, 'type_vocabsize': 190}
            self.task = {'vocab_size': (190, 156), 'num_classes': 1000, 'task': 'c++1000'}
            self.data = {'train_path': '/home/zhangkechi/workspace/dgl_tbcnn/data/Project_CodeNet_C++1000_spts/train.pkl',
                         'test_path': '/home/zhangkechi/workspace/dgl_tbcnn/data/Project_CodeNet_C++1000_spts/test.pkl',
                         'dev_path': '/home/zhangkechi/workspace/dgl_tbcnn/data/Project_CodeNet_C++1000_spts/dev.pkl',
                         'batch_size': self.batch_size,
                         'num_workers': self.num_workers}
            self.path = {
                'save': '/home/zhangkechi/workspace/dgl_tbcnn/save/Project_CodeNet_C++1000_spts'}
        elif self.task_name == 'c1400':
            # c++ 1400
            # {'token_vocabsize': 156, 'type_vocabsize': 190}
            self.task = {'vocab_size': (190, 156), 'num_classes': 1400, 'task': 'c++1000'}
            self.data = {'train_path': '/home/zhangkechi/workspace/dgl_tbcnn/data/Project_CodeNet_C++1400_spts/train.pkl',
                         'test_path': '/home/zhangkechi/workspace/dgl_tbcnn/data/Project_CodeNet_C++1400_spts/test.pkl',
                         'dev_path': '/home/zhangkechi/workspace/dgl_tbcnn/data/Project_CodeNet_C++1400_spts/dev.pkl',
                         'batch_size': self.batch_size,
                         'num_workers': self.num_workers}
            self.path = {
                'save': '/home/zhangkechi/workspace/dgl_tbcnn/save/Project_CodeNet_C++1400_spts'}
        else:
            raise ValueError('task_name not in [java, python, c1000, c1400]')
        

        self.optim = {'weight_decay': 1e-5, 'lr': 0.001, 'adam_epsilon':1e-8,'max_grad_norm': 1.0}
        self.num_epochs = 50

        


my_config = myConfig()
