import torch
import torch.nn as nn
import torch.optim as optim
import os
import sys
import random
import numpy as np
from Net.Model import Model
from sklearn import metrics
from data_loader import data_loader
from config import config
from utils import AverageMeter

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
#setup_seed(2021)

def train(train_loader, test_loader, opt):
    model = Model(train_loader.dataset.vec_save_dir, train_loader.dataset.rel_num(), opt)
    if torch.cuda.is_available():
        model = model.cuda()
    criterion = nn.CrossEntropyLoss(weight=train_loader.dataset.loss_weight())
    optimizer = optim.SGD(model.parameters(), lr=opt['lr'], weight_decay=1e-5)
    save_dir = os.path.join('result', opt['encoder'])
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    not_best_count = 0
    best_auc = 0
    for epoch in range(opt['epoch']):
        model.train()
        print("\n=== Epoch %d train ===" % epoch)
        avg_loss = AverageMeter()
        avg_acc = AverageMeter()
        avg_pos_acc = AverageMeter()
        for i, data in enumerate(train_loader):
            if torch.cuda.is_available():
                for d in range(len(data)):
                    data[d] = data[d].cuda()
            word, pos1, pos2, mask, length, rel = data
            output = model(word, pos1, pos2, mask，length)
            loss = criterion(output, rel)
            _, pred = torch.max(output, -1)
            acc = (pred == rel).sum().item() / rel.shape[0]
            pos_total = (rel != 0).sum().item()
            pos_correct = ((pred == rel) & (rel != 0)).sum().item()
            if pos_total > 0:
                pos_acc = pos_correct / pos_total
            else:
                pos_acc = 0
            # Log
            avg_loss.update(loss.item(), 1)
            avg_acc.update(acc, 1)
            avg_pos_acc.update(pos_acc, 1)
            sys.stdout.write('\rstep: %d | loss: %f, acc: %f, pos_acc: %f'%(i+1, avg_loss.avg, avg_acc.avg, avg_pos_acc.avg))
            sys.stdout.flush()
            # Optimize
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        if (epoch + 1) % opt['val_iter'] == 0:
            print("\n=== Epoch %d val ===" % epoch)
            y_true, y_pred = valid(test_loader, model)
            auc = metrics.average_precision_score(y_true, y_pred)
            print("\n[TEST] auc: {}".format(auc))
            if auc > best_auc:
                print("Best result!")
                best_auc = auc
                torch.save({'state_dict': model.state_dict()}, os.path.join(save_dir, 'model.pth.tar'))
                not_best_count = 0
            else:
                not_best_count += 1
            if not_best_count >= opt['early_stop']:
                break


def valid(test_loader, model):
    model.eval()
    avg_acc = AverageMeter()
    avg_pos_acc = AverageMeter()
    y_true = []
    y_pred = []
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            if torch.cuda.is_available():
                data = [x.cuda() for x in data]
            word, pos1, pos2, mask, length, scope, rel = data
            sen_output = torch.softmax(model(word, pos1, pos2, mask，length), -1)
            output = []
            for s in scope:
                output.append(sen_output[s[0]:s[1]].mean(0))
            output = torch.stack(output)
            label = rel.argmax(-1)
            _, pred = torch.max(output, -1)
            acc = (pred == label).sum().item() / label.shape[0]
            pos_total = (label != 0).sum().item()
            pos_correct = ((pred == label) & (label != 0)).sum().item()
            if pos_total > 0:
                pos_acc = pos_correct / pos_total
            else:
                pos_acc = 0
            # Log
            avg_acc.update(acc, 1)
            avg_pos_acc.update(pos_acc, 1)
            sys.stdout.write('\rstep: %d | acc: %f, pos_acc: %f'%(i+1, avg_acc.avg, avg_pos_acc.avg))
            sys.stdout.flush()
            y_true.append(rel[:, 1:])
            y_pred.append(output[:, 1:])
    y_true = torch.cat(y_true).reshape(-1).detach().cpu().numpy()
    y_pred = torch.cat(y_pred).reshape(-1).detach().cpu().numpy()
    return y_true, y_pred

def test(test_loader):
    model = load(opt['encoder'], test_loader)
    print("\n=== Test ===")
    y_true, y_pred = valid(test_loader, model)
    auc = metrics.average_precision_score(y_true, y_pred)
    print("\n[TEST] auc: {}".format(auc))
    order = np.argsort(-y_pred)
    p100 = (y_true[order[:100]]).mean()
    p200 = (y_true[order[:200]]).mean()
    p300 = (y_true[order[:300]]).mean()
    print("P@100: {0:.1f}".format(p100*100))
    print("P@200: {0:.1f}".format(p200*100))
    print("P@300: {0:.1f}".format(p300*100))
    print("mean: {0:.1f}".format((p300+p200+p300)/0.03))
    return y_true, y_pred

def load(model_name, test_loader):
    save_dir = os.path.join('result', model_name)
    model = Model(test_loader.dataset.vec_save_dir, test_loader.dataset.rel_num(), opt)
    if torch.cuda.is_available():
        model = model.cuda()
    state_dict = torch.load(os.path.join(save_dir, 'model.pth.tar'))['state_dict']
    own_state = model.state_dict()
    for name, param in state_dict.items():
        if name not in own_state:
            continue
        own_state[name].copy_(param)
    return model

def pr(model_name, y_true, y_pred):
    save_dir = os.path.join('result', model_name)
    order = np.argsort(y_pred)[::-1]
    correct = 0.
    total = y_true.sum()
    print("total triple num:", total)
    precision = []
    recall = []
    for i, o in enumerate(order):
        correct += y_true[o]
        precision.append(float(correct) / (i + 1))
        recall.append(float(correct) / total)
    precision = np.array(precision)
    recall = np.array(recall)
    np.save(os.path.join(save_dir, 'precision.npy'), precision)
    np.save(os.path.join(save_dir, 'recall.npy'), recall)

if __name__ == '__main__':
    opt = vars(config())
    train_loader = data_loader(opt['train'], opt, shuffle=True, training=True)
    test_loader = data_loader(opt['test'], opt, shuffle=False, training=False)
    # best_model = load(opt['encoder'], test_loader)
    train(train_loader, test_loader, opt)
    y_true, y_pred = test(test_loader)
    pr(opt['encoder'], y_true, y_pred)

