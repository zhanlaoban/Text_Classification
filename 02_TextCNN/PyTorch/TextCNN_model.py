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
            x = self.embedding(x)
            x = x.unsqueeze(1)
        
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs]
        x = [F.max_pool1d(item, item.size(2)).squeeze(2) for item in x]
        x = torch.cat(x, 1)
        x = self.dropout(x)
        x = self.fc(x)

        return x
