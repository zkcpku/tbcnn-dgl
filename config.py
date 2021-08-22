class myConfig():
    def __init__(self) -> None:
        self.device = 'cuda'
        self.seed = 36
        self.batch_size = 16
        self.num_workers = 8

        self.model = {'x_size':128, 'h_size':128, 'dropout':0.3, 'num_layers':2}
        
        self.task_name = 'python'
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
        else:
            # python token_vocabsize, type_vocabsize: (100, 61)
            self.task = {'vocab_size': (61, 100), 'num_classes': 800, 'task': 'python800'}
            self.data = {'train_path': '/home/zhangkechi/workspace/dgl_tbcnn/data/Project_CodeNet_Python800_spts/train.pkl',
                        'test_path': '/home/zhangkechi/workspace/dgl_tbcnn/data/Project_CodeNet_Python800_spts/test.pkl',
                        'dev_path': '/home/zhangkechi/workspace/dgl_tbcnn/data/Project_CodeNet_Python800_spts/dev.pkl',
                        'batch_size': self.batch_size,
                        'num_workers': self.num_workers}
            self.path = {
                'save': '/home/zhangkechi/workspace/dgl_tbcnn/save/Project_CodeNet_Python800_spts'}
        

        self.optim = {'weight_decay': 1e-5, 'lr': 0.0001, 'adam_epsilon':1e-8,'max_grad_norm': 1.0}
        self.num_epochs = 50

        


my_config = myConfig()
