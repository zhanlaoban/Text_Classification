import re
import logging

import jieba
from torchtext import data
from torchtext import vocab

jieba.setLogLevel(logging.INFO)

def tokenizer(text):
    """
    定义TEXT的tokenize规则
    """

    #去掉不在(所有中文、大小写字母、数字)中的非法字符
    regex = re.compile(r'[^\u4e00-\u9fa5A-Za-z0-9]') 
    text = regex.sub(' ', text)

    #使用jieba分词
    return [word for word in jieba.cut(text) if word.strip()]


def GetIterator(TEXT, LABEL, path, args, **kwargs):
    """
    生成数据迭代器

    args:
        TEXT: torchtext.data生成的Field对象
        LABEL: torchtext.data生成的Field对象
        
    return:
        train_iter： 训练集迭代器
        dev_iter： 验证集迭代器

    """
    
    #定义TEXT的tokenize规则
    TEXT.tokenize = tokenizer
    
    #创建表格数据集
    train_dataset, dev_dataset, test_dataset = data.TabularDataset.splits(
        path = path, format = 'csv', skip_header = True,
        train='cnews.train.csv', validation='cnews.val.csv', test='cnews.test.csv',
        fields=[
            ('label', LABEL),
            ('text', TEXT),
        ]
    )

    if args.static and args.pretrainedEmbeddingName and args.pretrainedEmbeddingPath:
        #加载预训练的词向量，name:包含词向量的文件名，cache:包含词向量的目录
        vectors = vocab.Vectors(name = args.pretrainedEmbeddingName, cache = args.pretrainedEmbeddingPath)    
        #建立TEXT的词汇表
        TEXT.build_vocab(train_dataset, dev_dataset, vectors=vectors)
    else:
        TEXT.build_vocab(train_dataset, dev_dataset)
    
    #建立LABEL的词汇表
    LABEL.build_vocab(train_dataset, dev_dataset)
    
    train_iter, dev_iter, test_iter = data.Iterator.splits((train_dataset, dev_dataset, test_dataset), batch_sizes = (args.batchSize, len(dev_dataset)/8, len(test_dataset)/8), sort_key = lambda x: len(x.text), **kwargs)
    
    return train_iter, dev_iter, test_iter
