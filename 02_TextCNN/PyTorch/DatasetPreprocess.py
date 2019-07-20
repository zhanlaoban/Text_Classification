import re
import logging

import jieba
from torchtext import data
from torchtext import vocab

jieba.setLogLevel(logging.INFO)

def CutWordFunc(text):
    """
    定义TEXT的tokenize规则

    """

    #去掉不在(所有中文、大小写字母、数字)中的非法字符
    regex = re.compile(r'[^\u4e00-\u9fa5A-Za-z0-9]') 
    text = regex.sub(' ', text)

    return [word for word in jieba.cut(text) if word.strip()]


def GetDataset(path, TEXT, LABEL):
    """
    制作数据集

    args:
        path: 自定义数据集存放的目录
        text: 预处理的文本
        TEXT: torchtext.data生成的Field对象
        LABEL: torchtext.data生成的Field对象


    return:
        train: 返回制作的训练数据集
        dev: 返回制作的验证数据集

    """
   
    #定义TEXT的tokenize规则
    TEXT.tokenize = CutWordFunc
    
    #创建表格数据集
    train, dev = data.TabularDataset.splits(
        path = path, format = 'csv', skip_header = True,
        #train='train_new_divided_index.csv', validation='dev_new_divided_index.csv',
        train='train_new_divided_index.csv', validation='dev_new_divided_index.csv',
        #train='dataset_train.csv', validation='dataset_test.csv',
        fields=[
            ('index', None),
            ('label', LABEL),
            ('text', TEXT)
        ]
    )

    return train, dev

def GetIterator(TEXT, LABEL, args, **kwargs):
    """
    生成数据迭代器

    args:
        TEXT: torchtext.data生成的Field对象
        LABEL: torchtext.data生成的Field对象
        args:

    """
    
    #加载数据集
    path = 'data'
    train_dataset, dev_dataset = GetDataset(path, TEXT, LABEL)

    if args.static and args.pretrainedName and args.pretrainedPath:
        #加载预训练的词向量，name:包含词向量的文件名，cache:包含词向量的目录
        vectors = vocab.Vectors(name = args.pretrainedName, cache = args.pretrainedPath)    
        #建立TEXT的词汇表
        TEXT.build_vocab(train_dataset, dev_dataset, vectors=vectors)
    else:
        TEXT.build_vocab(train_dataset, dev_dataset)
    
    #建立LABEL的词汇表
    LABEL.build_vocab(train_dataset, dev_dataset)
    
    train_iter, dev_iter = data.Iterator.splits((train_dataset, dev_dataset), batch_sizes = (args.batchSize, len(dev_dataset)/8), sort_key = lambda x: len(x.text), **kwargs)
    #train_iter, dev_iter = data.Iterator.splits((train_dataset, dev_dataset), batch_sizes = (args.batchSize, len(dev_dataset)), **kwargs)
    
    return train_iter, dev_iter