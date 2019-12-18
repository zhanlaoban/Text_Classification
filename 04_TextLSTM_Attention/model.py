import torch
import torch.nn as nn
import torch.nn.functional as F


class TextRNN(nn.Module):
    def __init__(self, args):
        #在子类中调用父类的初始化方法
        super(TextRNN, self).__init__()
         
        if args.static:
            self.embedding = nn.Embedding.from_pretrained(args.vectors, freeze = not args.fineTuneWordEm)
        else:
            self.embedding = nn.Embedding(args.vocab_size, args.embeddingDim)
     
        self.lstm = nn.LSTM(input_size = args.embeddingDim,
                            hidden_size = args.hidden_size,
                            num_layers = args.num_layers,
                            bias = True,
                            batch_first = False,
                            dropout = args.dropout,
                            bidirectional = True)   #num_directions==2

        self.fc = nn.Linear(args.hidden_size, args.classNum)

    def forward(self, x):        
        x = self.embedding(x)
        
        output, (h_n, c_n) = self.lstm(x.permute(1,0,2))

        x = self.fc(h_n[-1])


        return x

        
        #x = [F.relu(conv(x)).squeeze(3) for conv in self.convs]
        #x = [F.max_pool1d(item, item.size(2)).squeeze(2) for item in x]
        #x = torch.cat(x, 1)
        #x = self.fc(x)



