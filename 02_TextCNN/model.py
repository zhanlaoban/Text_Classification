import torch
import torch.nn as nn
import torch.nn.functional as F


class TextCNN(nn.Module):
    """
    定义TextCNN网络

    网络结构：
    static, singlechannel: embedding(pretrained) + (conv+batchnorm)*3 + maxpool + dropout + fc
    static, multichannel: embedding(pretrained) + embedding2(pretrained) + (conv+batchnorm)*3 + maxpool + dropout + fc

    """
    def __init__(self, args):
        #在子类中调用父类的初始化方法
        super(TextCNN, self).__init__()
        
        self.args = args

        embeddingNum = args.embeddingNum #the number of embedding vectors
        embeddingDim = args.embeddingDim #the dimension of embedding vector
        outChannel = args.filterNum
        filterSizes = args.filterSizes
        classNum = args.classNum
        inChannel = 1
        
        self.embedding = nn.Embedding(embeddingNum, embeddingDim)
        
        if args.static:
            self.embedding = self.embedding.from_pretrained(args.vectors, freeze = not args.fineTuneWordEm)

        if args.multichannel:
            self.embedding2 = nn.Embedding(embeddingNum, embeddingDim).from_pretrained(args.vectors , freeze = not args.fineTuneWordEm)
            inChannel += 1
        else:
            self.embedding2 = None

        self.convs = nn.ModuleList([
            nn.Conv2d(inChannel, outChannel, (size, embeddingDim)) for size in filterSizes])
        #self.convs = convss

        '''
        convs = [nn.Sequential(
                    nn.Conv2d(inChannel, outChannel, (size, embeddingDim)),
                    nn.Conv2d(inChannel, outChannel, (size, embeddingDim))
                    )
                for size in filterSizes]

        self.convs = nn.ModuleList(convs)
        '''
        self.dropout = nn.Dropout(args.dropout)
        self.fc = nn.Linear(len(filterSizes) * outChannel, classNum)

    def forward(self, x):
        if self.embedding2:
            x = torch.stack([self.embedding(x), self.embedding2(x)], dim=1)
        else:
            #print("x:", x.size())   #x: [batchsize, sentence_size], 第一个维度表示batchsize, 第二个维度表示一行文档的长度?
            
            x = self.embedding(x)
            #print("unsqueeze x:", x.size())  #[batchsize, sentence_size, embeddingDim], 第三个维度表示词向量维度
            
        x = x.unsqueeze(1)  #[64, 1, 2827, 300] .unsqueeze(1)表示在tensor指定的第1维度处插入维度大小为1
            #print("unsqueezeed x:", x.size())   
            
        
        
        #这样写的好处是conv(x)中的x,每次for循环开始运行时都是x = x.unsqueeze(1)的x值,而不是进行conv后的x值
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs] 
        #x: [batchsize, filternum, H]
        #上条语句运行后,x为长度为3的list
        #为了学习的目的,将上调语句修改为下面的代码段,也可以运行:
        '''
        y = []
        temp = x #暂存x,保证每次运行for循环时,都是同一个一个x值,或者说句子词向量矩阵
        for conv in self.convs:
            x = conv(temp)            
            x = F.relu(x)
            x = x.squeeze(3)
            print("conv x size:", x.size())
            y.append(x)
        x = y
        '''

        #print("conv x: ", x.size())
        #print("conv x size: ", len(x))
        
        x = [F.max_pool1d(item, item.size(2)).squeeze(2) for item in x]
        #x: [batchsize, filternum]
        '''
        z = []
        for item in x:
            print("item :", item.size())
            item = F.max_pool1d(item, item.size(2))
            item = item.squeeze(2)
            z.append(item)
        x = z
        '''

        x = torch.cat(x, 1) #将x列表中的tensor在维度1处concatenate起来
        #x: [batchsize, filter*3]

        x = self.dropout(x)
        #x: [batchsize, filter*3]

        x = self.fc(x)
        #x: [batchsize, classnum]

        return x
