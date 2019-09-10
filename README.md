# Text_Classification
Highlights：

- 深度学习**中文文本分类**任务的各种模型实现  
- 以PyTorch和TensorFlow两种形式实现  
- 每个模型均以THUCNews作为benchmark数据集
- 每种模型的实现原理和细节在各个模型文件夹的README.MD中



# Dataset

[THUCNews数据集](http://thuctc.thunlp.org/#中文文本分类数据集THUCNews)

> THUCNews是根据新浪新闻RSS订阅频道2005~2011年间的历史数据筛选过滤生成，包含74万篇新闻文档（2.19 GB），均为UTF-8纯文本格式。我们在原始新浪新闻分类体系的基础上，重新整合划分出14个候选分类类别：财经、彩票、房产、股票、家居、教育、科技、社会、时尚、时政、体育、星座、游戏、娱乐。

原数据集是以一个类别名作为一个文件夹名，在每个文件夹下，单条语料又是以一个单独的txt文件存在的。为了方便模型中对数据集的预处理，减小整体语料数量，预先对该数据集进行了处理，减少了后续的工作量。

**本项目所用benchmark数据集介绍：**

本次训练使用了其中的5个分类，每个分类5000条数据。

- 数据集划分如下：

  训练集: 4000 * 5
  验证集: 500 * 5
  测试集: 500 * 5

- Train/Dev/Test：8/1/1

- classes：5个类别，即：体育, 财经, 房产, 家居, 教育

下载(长期有效)：链接：https://pan.baidu.com/s/1-g2M47lwL9DoZTHCEqfCAA  提取码：ztxt



# Contents

### 01. FastText: TODO

### 02. [TextCNN](https://github.com/zhanlaoban/Text_Classification/tree/master/02_TextCNN)

### 03. TextRNN: TODO

### 04. TextRCNN:TODO

### 05. BERT:TODO

