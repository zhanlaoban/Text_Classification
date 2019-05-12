import os
import sys
import time
import torch
import torch.optim as optim
import torch.nn.functional as F

def train(train_iter, dev_iter, model, args):
    '''
    Define the train process. It is the core part of the program.

    args:
    train_iter: iterator of preprocessed trian dataset
    dev_iter: iterator of preprocessed dev dataset
    model: the model of textcnn
    args: get the argments
    
    '''

    if args.cuda:
        model.cuda()
    #optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    #scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer)
    steps = 0
    best_acc = 0
    last_step = 0
    model.train()
    for epoch in range(1, args.epochs + 1):
        #adjust_lr(optimizer, epoch, args)
        for batch in train_iter:
            feature, target = batch.text, batch.label
            feature.data.t_(), target.data.sub_(1)

            if args.cuda:
                feature, target = feature.cuda(), target.cuda()

            optimizer.zero_grad()
            logits = model(feature)
            loss = F.cross_entropy(logits, target)
            loss.backward()
            optimizer.step()
            #scheduler.step(loss)
            steps += 1
            
            if steps % args.logInterval == 0:
                corrects = (torch.max(logits, 1)[1].view(target.size()).data == target.data).sum()
                train_acc = 100.0 * corrects / batch.batch_size
                sys.stdout.write(
                    '\rBatch[{}] Training Set: - loss: {:.6f} - acc: {:.4f}%=({}/{})'.format(steps,
                                                                             loss.item(),
                                                                             train_acc,
                                                                             corrects,
                                                                             batch.batch_size))
            

            #save the best acc
            if steps % args.testInterval == 0:
                dev_acc = eval(dev_iter, model, args)
                if dev_acc > best_acc:
                    best_acc = dev_acc
                    last_step = steps
                    if args.saveBest:
                        print('Saving best model, acc: {:.4f}%\n'.format(best_acc))
                        save(model, args.saveDir, 'best_acc', best_acc)
                else:  #beyond 1000 steps without performance increasing
                    if steps - last_step >= args.earlyStopping:
                        print('\nEarly stop by {} steps, Evaluation Set acc: {:.4f}%'.format(args.earlyStopping, best_acc))
                        raise KeyboardInterrupt


def eval(data_iter, model, args):
    '''
    Evaluate the accuracy of the dev dataset.

    args:
    data_iter: iterator of preprocessed dev dataset
    model: the textcnn model
    args: get the argments

    '''
    model.eval()
    corrects, avg_loss = 0, 0
    for batch in data_iter:
        feature, target = batch.text, batch.label
        feature.data.t_(), target.data.sub_(1)
        if args.cuda:
            feature, target = feature.cuda(), target.cuda()
        logits = model(feature)
        loss = F.cross_entropy(logits, target)
        avg_loss += loss.item()
        corrects += (torch.max(logits, 1)[1].view(target.size()).data == target.data).sum()

    size = len(data_iter.dataset)
    avg_loss /= size
    accuracy = 100.0 * corrects / size
    print('\n           Evaluation Set - loss: {:.6f} - acc: {:.4f}%=({}/{}) \n'.format(avg_loss,
                                                                       accuracy,
                                                                       corrects,
                                                                       size))
    return accuracy


def save(model, saveDir, save_prefix, best_acc):
    '''
    Save the model file.

    args:
    model: the textcnn model
    saveDir: the directory of save file
    save_prefix: prefix of the save file
    best_acc: save the best accuracy to the model file

    '''
    if not os.path.exists(saveDir):
        os.makedirs(saveDir)
    save_prefix = os.path.join(saveDir, save_prefix)
    save_path = '{}_{}_{:.4f}%.pt'.format(save_prefix, time.strftime("%Y-%m-%d %H:%M", time.localtime()), best_acc)

    torch.save(model.state_dict(), save_path)

def adjust_lr(optimizer, epoch, args):
    '''
    Adjust learning rate, but not working well.

    args:
    optimizer: torch.optim生成的optim对象
    epoch: epoch参数
    args: 获得args.lr

    '''
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr