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
    
        self.gru = nn.GRU(input_size = args.embeddingDim,
                            hidden_size = args.hidden_size,
                            num_layers = args.num_layers,
                            bias = True,
                            batch_first = False,
                            dropout = args.dropout,
                            bidirectional = True)   #num_directions==2

        self.fc = nn.Linear(args.hidden_size, args.classNum)

    def forward(self, x):
        #x即同TrainModel.py中的input
        #x:[batchsize, sententce_size]
        
        x = self.embedding(x)
        #print("x shape: ", x.shape)
        #x:[batchsize, sententce_size, embedding_size]
        
        output, h_n = self.gru(x.permute(1,0,2))   #h_0和c_0均初始化为0
        #print("output shape: ", output.permute(1,0,2).shape)   #[batchsize, sententce_size, num_directions*hiddenSize_LSTM]

        #print("h_n shape", h_n.shape)   #[num_layers * num_directions, batch_size, hidden_size]
        #x = F.relu(output.permute(1,0,2))
        #x = F.relu(h_n.permute(1,0,2))
        #print("x shape: ", x.shape)    #[batchsize, sententce_size, num_directions*hiddenSize_LSTM]



        #print("x[:, -1, :] shape: ", x[:, -1, :].shape) #[batch_size, num_directions*hiddenSize_LSTM]
        #print("torch.mean(x, 1) shape: ", torch.mean(x, 1).shape)   #[batch_size, num_directions*hiddenSize_LSTM]        
        #print("h_n[-1] shape: ", h_n[-1].shape) #[batch_size, hidden_size]

        x = self.fc(h_n[-1])
        #x = self.fc(torch.mean(x, 1))
        #x = self.fc(x[:, -1, :])
        #上面的两种写法应该不对, 即不应该获取output,而应该是最后一个时刻的隐状态: h_n
        #x shape: [batch_size, classnum]

        return x

        
        #x = [F.relu(conv(x)).squeeze(3) for conv in self.convs]
        #x = [F.max_pool1d(item, item.size(2)).squeeze(2) for item in x]
        #x = torch.cat(x, 1)
        #x = self.fc(x)



