# TextLSTM

## 1. Principle

好像没有相关论文，所以没有参考论文，仅仅是自己凭感觉实现的简单TextLSTM：

1. 初始隐状态h_0和细胞状态c_0都为0
2. 使用双向LSTM（bidirectional = True）
3. 获取最后一层隐状态h_n，因为是双向LSTM，获取的是反向的第二层LSTM最后的h_n（h_n[1, :, :]）
4. 最后再接一个全连接层（nn.Linear）



## 模型架构

```
TextLSTM(
  (embedding): Embedding(28043, 300)
  (lstm): LSTM(300, 64, dropout=0.5, bidirectional=True)
  (fc): Linear(in_features=64, out_features=6, bias=True)
)
```






