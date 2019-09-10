## 1. TextCNN原理
请见我的知乎文章集合：

[深入TextCNN（一）详述CNN及TextCNN原理](https://zhuanlan.zhihu.com/p/77634533)






## 2. PyTorch

#### Files details

DatasetPreprocess.py：数据集的预处理，使用了torchtext生成迭代器

TextCNN_model.py：定义TextCNN模型架构

TextCNN_run.py：定义超参数，定义模型训练的入口

TrainModel.py：定义模型训练、评估、保存

#### Usage

##### For training:

1. 仅限使用预训练的词向量来进行训练模型：[Github](https://github.com/zhanlaoban/NLP_Word2vec_Datasets#100-chinese-word-vectors-%E4%B8%8A%E7%99%BE%E7%A7%8D%E9%A2%84%E8%AE%AD%E7%BB%83%E4%B8%AD%E6%96%87%E8%AF%8D%E5%90%91%E9%87%8F)。新建目录pretrainedW2v，并将上述模型解压得到的文件，放到pretrainedW2v目录中

2. 

##### For Predicting








## 3. TensorFlow: TODO
