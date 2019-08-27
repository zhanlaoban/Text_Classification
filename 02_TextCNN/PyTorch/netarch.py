import torch
import torch.nn as nn
import torch.nn.functional as F 

class TextCNN(nn.Module):
	def __init__(self, args):
        #在子类中调用父类的初始化方法
        super(TextCNN, self).__init__()

        