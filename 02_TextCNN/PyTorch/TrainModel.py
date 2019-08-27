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
        model.cuda()    #将模型的所有参数和缓存移步到GPU上，此操作必须在构建optimizer之前执行

    #optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)    
    #scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer)

    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.1)
    model.train()

    steps = 0
    best_acc = 0
    last_step = 0
    
    for epoch in range(1, args.epochs + 1):
        #adjust_lr(optimizer, epoch, args)
        for batch in train_iter:
            input, target = batch.text, batch.label
            input = input.data.t()
            print()
            print(input)
            print(input.data)
            print(input.data.t())                                                                                                                                                       
            target = target.data.sub(1)
            #print(target.data)

            if args.cuda:
                input, target = input.cuda(), target.cuda()

            optimizer.zero_grad()
            output = model(input)

            print("output")
            print(type(output))
            print(output.size)

            loss = F.cross_entropy(output, target)
            loss.backward()                                                                                             
            optimizer.step()                                                                    
            #scheduler.step(loss)
            steps += 1
            
            if steps % args.logInterval == 0:
                corrects = (torch.max(output, 1)[1].view(target.size()).data == target.data).sum()
                train_acc = 100.0 * corrects / batch.batch_size
                sys.stdout.write(
                    '\rBatch[{}] Training Set: - loss: {:.6f} - acc: {:.4f}%=({}/{})'.format(steps,
                                                                                            loss.item(),
                                                                                            train_acc,
                                                                                            corrects,
                                                                                            batch.batch_size))
            

            #save the best acc
            if steps % args.valInterval == 0:
                dev_acc = eval(dev_iter, model, args)
                if dev_acc > best_acc:
                    best_acc = dev_acc
                    last_step = steps
                    if args.modelSaveBest:
                        print('Saving best model, acc: {:.4f}%\n'.format(best_acc))
                        save(model, args.modelSaveDir, 'best_acc', best_acc)
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
        input, target = batch.text, batch.label
        input = input.data.t()
        target = target.data.sub(1)
        if args.cuda:
            input, target = input.cuda(), target.cuda()
        output = model(input)
        loss = F.cross_entropy(output, target)
        avg_loss += loss.item()
        corrects += (torch.max(output, 1)[1].view(target.size()).data == target.data).sum()

    size = len(data_iter.dataset)
    avg_loss /= size
    accuracy = 100.0 * corrects / size
    print('\n           Evaluation Set - loss: {:.6f} - acc: {:.4f}%=({}/{}) \n'.format(avg_loss,
                                                                       accuracy,
                                                                       corrects,
                                                                       size))
    return accuracy


def save(model, modelSaveDir, save_prefix, best_acc):
    '''
    Save the model file.

    args:
    model: the textcnn model
    modelSaveDir: the directory of save file
    save_prefix: prefix of the save file
    best_acc: save the best accuracy to the model file

    '''
    if not os.path.exists(modelSaveDir):
        os.makedirs(modelSaveDir)
    save_prefix = os.path.join(modelSaveDir, save_prefix)
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