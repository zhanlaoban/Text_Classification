import os
import sys
import time
import argparse
import torch
from torchtext import data
from torchtext import vocab
from tensorboardX import SummaryWriter

import TextCNN_model
import TrainModel
import DatasetPreprocess

parser = argparse.ArgumentParser(description='TextCNN text classifier')
# Model hyper parameters
parser.add_argument('-lr', type=float, default=0.1, help='initial learning rate [default: 0.001]')
parser.add_argument('-epochs', type=int, default=256, help='number of epochs for train [default: 256]')
parser.add_argument('-batchSize', type=int, default=128, help='batch size for training [default: 128]')
parser.add_argument('-dropout', type=float, default=0.5, help='the probability for dropout [default: 0.5]')
parser.add_argument('-maxNorm', type=float, default=3.0, help='l2 constraint of parameters [default: 3.0]')
parser.add_argument('-embeddingDim', type=int, default=300, help='number of embedding dimension [default: 128]')
parser.add_argument('-filterNum', type=int, default=10, help='number of each size of filter')
parser.add_argument('-filterSizes', type=str, default='3,4,5', help='comma-separated filter sizes to use for convolution')
parser.add_argument('-earlyStopping', type=int, default=1000, help='iteration numbers to stop without performance increasing')
# Word embedding parameters
parser.add_argument('-static', type=bool, default=True, help='whether to use static pre-trained word vectors')
parser.add_argument('-fineTuneWordEm', type=bool, default=False, help='whether to fine-tune static pre-trained word vectors')
parser.add_argument('-multichannel', type=bool, default=False, help='whether to use 2 channel of word vectors')
parser.add_argument('-logInterval', type=int, default=1, help='how many steps to wait before logging training status [default: 1]')
parser.add_argument('-valInterval', type=int, default=100, help='how many steps to wait before testing [default: 100]')
# Directories
parser.add_argument('-datasetDir', type=str, default='data/cnews/5_100', help='Directory of dataset [default: None]')
parser.add_argument('-pretrainedEmbeddingName', type=str, default='sgns.sogounews.bigram-char', help='filename of pre-trained word vectors')
parser.add_argument('-pretrainedEmbeddingPath', type=str, default='./pretrainedW2v', help='path of pre-trained word vectors')
parser.add_argument('-modelSaveDir', type=str, default='modelSaveDir', help='where to save the modelsavedir')
parser.add_argument('-modelSaveBest', type=bool, default=True, help='whether to save when get best performance')
parser.add_argument('-modelLoadFilename', type=str, default=None, help='filename of model loading [default: None]')
# Device
parser.add_argument('-device', type=int, default=0, help='device to use for iterate data, -1 mean cpu [default: -1]')
args = parser.parse_args()



###process the dataset
print('Processing dataset start...')
TEXT = data.Field()
LABEL = data.Field(sequential=False)
train_iter, dev_iter, test_iter = DatasetPreprocess.GetIterator(TEXT, LABEL, args.datasetDir, args, device = -1, repeat = False, shuffle = True)
print('Processing data done!')

#process the parameters
args.embeddingNum = len(TEXT.vocab)
args.classNum = len(LABEL.vocab)

print('args.embeddingNum = ', args.embeddingNum)
print('args.classNum = ', args.classNum)
print('LABEL.vocab = ', LABEL.vocab)

args.cuda = args.device != -1 and torch.cuda.is_available()
args.filterSizes = [int(size) for size in args.filterSizes.split(',')]

if args.static:
    args.embeddingDim = TEXT.vocab.vectors.size()[-1]
    #TEXT.vocab.vectors: [332909, 300]
    args.vectors = TEXT.vocab.vectors

if args.multichannel:
    args.static = True
    args.nonStatic = True

###print parameters
print('Parameters:')
for attr, value in sorted(args.__dict__.items()):
    if attr in {'vectors'}:
        continue
    print('\t{}={}'.format(attr.upper(), value))



###train
textCNN = TextCNN_model.TextCNN(args)

print('args.vectors ', type(args.vectors))
#with SummaryWriter(log_dir='./visualLog', comment='TextCNN') as writer:
#    writer.add_graph(textCNN, (train_iter,))
print(textCNN)

if args.modelLoadFilename:
    print('\nLoading model from {}...\n'.format(args.modelLoadFilename))
    textCNN.load_state_dict(torch.load(args.modelLoadFilename))

if args.cuda:
    torch.cuda.set_device(args.device)
    textCNN = textCNN.cuda()


try:
    TrainModel.train(train_iter, dev_iter, textCNN, args)
except KeyboardInterrupt:
    print('\nTraining CANCEL! \nExiting from training')