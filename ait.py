'''Augmentation Invariant Training
'''
from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.optim.lr_scheduler as lr_scheduler

import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import os
import argparse
import time
import copy
import numpy as np
import random

from models import *
from utils import Log
from utils import summary
from load_ait import AITList

parser = argparse.ArgumentParser(description='Augmentation Invariant Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--batch-size', default=128, type=int, help='batch size')
parser.add_argument('--test-batch-size', default=256, type=int, help='test batch size')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--log', default='logs/tmp.log', type=str, help='log file')
parser.add_argument('--arch', default='resnet32,resnet32', type=str, help='arch')
parser.add_argument('--algo', default='resnet32,resnet32', type=str, help='algo')
parser.add_argument('--kl-w', default=1., type=float, help='AiLoss weight')
parser.add_argument('--parallel', default=2, type=int, help='variants number')
args = parser.parse_args()

Log(args.log, 'w+', 1) # set log file
print("=================FLAGS==================")
for k, v in args.__dict__.items():
    print('{}: {}'.format(k, v))
print("========================================")

date = time.strftime('%m%d%H%M', time.localtime())
device = 'cuda' if torch.cuda.is_available() else 'cpu'
start_epoch = 0  # start from epoch 0 or last checkpoint epoch
epoches = 160 # 160 # train epoches
lr_step = [80, 120] # [80, 120] # step for adjust lr
arches, augs, data = args.arch.split('_')
arches = arches.split(',')
augs = augs.split(',')
net_num = len(arches)
# Data
print('==> Preparing data..')
randomcrop = transforms.RandomCrop(32, padding=4)
randomgray = transforms.RandomGrayscale(p=0.1)
randomcolor = transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1)
if data == 'cifar100':
    normalize = transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276])
elif data == 'cifar10':
    normalize = transforms.Normalize(mean=[0.491, 0.482, 0.447], std=[0.247, 0.243, 0.262])
else:
    print('Wrong data')
    sys.exit(1)
transform_ops = [
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize,
]
transform_trains = []
for aug in augs:
    transform_op = transform_ops[:]
    if aug == 'rp':
        transform_op.insert(0, randomcrop)
    elif aug == 'rg':
        transform_op.insert(0, randomgray)
    elif aug == 'rc':
        transform_op.insert(0, randomcolor)
    transform_train = transforms.Compose(transform_op)
    transform_trains.append(transform_train)

transform_test = transforms.Compose([
    transforms.ToTensor(),
    normalize,
])

if data == 'cifar10':
    num_classes = 10
    trainlist = './data/lists/cifar10_train.txt'
    testlist = './data/lists/cifar10_test.txt'
    trainset = AITList(root='./data/cifar10-pics/train/', fileList=trainlist, net_num=net_num, parallel=args.parallel, train=True, transforms=transform_trains)
    testset = AITList(root='./data/cifar10-pics/test/', fileList=testlist, net_num=1, parallel=1, train=False, transforms=[transform_test])
elif data == 'cifar100':
    num_classes = 100
    trainlist = './data/lists/cifar100_train.txt'
    testlist = './data/lists/cifar100_test.txt'
    trainset = AITList(root='./data/cifar100-pics/train/', fileList=trainlist, net_num=net_num, parallel=args.parallel, train=True, transforms=transform_trains)
    testset = AITList(root='./data/cifar100-pics/test/', fileList=testlist, net_num=1, parallel=1, train=False, transforms=[transform_test])
trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=4)
testloader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=False, num_workers=4)

# Model
print('==> Building model..')
nets = []

for arch in arches:
    if 'cifar' in data:
        size = (3, 32, 32)
        if 'resnet32' in arch:
            net = resnet32_cifar(num_classes=num_classes)
        elif 'mobilenet' in arch:
            net = MobileNet(num_classes=num_classes)
        elif 'mobilev2' in arch:
            net = MobileNetV2(num_classes=num_classes)
        elif 'googlenet' in arch:
            net = GoogLeNet(num_classes=num_classes)
        elif 'wideresnet' in arch:
            net = wideresnet(28, 10, num_classes=num_classes)
        elif 'densenet' in arch:
            net = densenet_cifar(num_classes=num_classes)
    net = net.to(device)
    net = torch.nn.DataParallel(net)
    summary(copy.deepcopy(net.cuda()), size)
    nets.append(net)
cudnn.benchmark = True

best_accs = (net_num) * [0.]  # best test accuracy
best_epoches = net_num * [0]  # best test epoch

if args.resume:
    # Load checkpoint.
    assert os.path.isdir('save'), 'Error: no checkpoint directory found!'
    checkpoint = 'save/tmp.tar'
    print('==> Resuming from checkpoint {} ..', checkpoint)
    checkpoint = torch.load(checkpoint)
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

criterion_kl = nn.KLDivLoss(size_average=False)
criterion = nn.CrossEntropyLoss()
criterions = [criterion_kl, criterion]
schedulers = []
optimizers = []
for i in range(len(criterions)):
    criterions[i] = criterions[i].cuda()
for i in range(net_num):
    optimizer = optim.SGD(nets[i].parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)
    if 'wideresnet' in arches[i]:
        optimizer = optim.SGD(nets[i].parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=lr_step, gamma=0.1)
    schedulers.append(scheduler)
    optimizers.append(optimizer)

# Training
def train(epoch):
    """Train"""
    print('\nEpoch: {} {}'.format(epoch, time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())))
    top1s = []
    top5s = []
    for i in range(net_num):
        nets[i].train()
        schedulers[i].step()
        top1s.append(AverageMeter())
        top5s.append(AverageMeter())
    for batch_idx, (inputs, target) in enumerate(trainloader):
        if not isinstance(inputs, list):
            input = inputs.to(device)
        target = target.to(device)
        bs = target.size(0)
        loss_soft = []
        probs = []
        log_probs = []
        outputs = []
        loss_ais = []
        for i, arch in enumerate(arches):
            if isinstance(inputs, list):
                input = inputs[i]
                input = input.to(device)
            input_i = copy.copy(input)
            input_i = input_i.to(device)
            optimizers[i].zero_grad()
            output = nets[i](input_i)
            prec1, prec5 = accuracy(output.data, target, topk=(1,5))
            top1s[i].update(prec1.item(), input_i.size(0))
            top5s[i].update(prec5.item(), input_i.size(0))
            outputs.append(output)
            loss_soft.append(criterion(output, target))
            probs.append(F.softmax(output, 1))
            log_probs.append(F.log_softmax(output, 1))

        # ai loss
        # multi-AIT
        loss_ai = 0.
        if net_num > 1:
            for i in range(net_num):
                for j in range(net_num):
                    if i != j:
                        loss_ais.append(criterion_kl(log_probs[i], probs[j].detach()) / float(bs))

            loss_ai = sum(loss_ais) / (net_num -1)
        # AIT
        elif args.parallel > 1:
            for i in range(args.parallel):
                probs_i = probs[0][i::args.parallel]
                for j in range(args.parallel):
                    if i != j:
                        log_probs_j = log_probs[0][j::args.parallel]
                        loss_ais.append( \
                                criterion_kl(log_probs_j, \
                                probs_i.detach()) / (float(bs) / 2)
                                )
            loss_ai = sum(loss_ais) / (args.parallel -1)
        kl_w = args.kl_w
        loss = kl_w * loss_ai + np.sum(np.array(loss_soft))
        loss.backward()
        lr = optimizers[0].param_groups[0]['lr']
        for i in range(net_num):
            optimizers[i].step()
            assert lr == optimizers[i].param_groups[0]['lr']
        if batch_idx % 100 == 0:
            date = time.strftime('%m-%d %H:%M:%S', time.localtime())
            top1s_p = [round(x.avg, 2) for x in top1s]
            top5s_p = [round(x.avg, 2) for x in top5s]
            loss_soft_p = [round(lo.data.item(), 3) for lo in loss_soft]
            loss_ai_p = [round(lo.cpu().item(), 2) for lo in loss_ais]
            print('{} Epoch {}: {}/{} Loss_ai: {} Loss_soft: {} Loss: {:.3f} | Lr: {:.4f} | Accs: {}'
                    .format(date, epoch, batch_idx, len(trainloader), loss_ai_p, loss_soft_p, loss, lr, top1s_p))

def test(epoch):
    """Test"""
    global best_acc
    test_loss = 0
    top1s = []
    top5s = []
    for i in range(net_num):
        top1s.append(AverageMeter())
        top5s.append(AverageMeter())
    with torch.no_grad():
        for batch_idx, (input, target) in enumerate(testloader):
            input, target = input.to(device), target.to(device)
            outputs = []
            for i, arch in enumerate(arches):
                nets[i].eval()
                input_i = copy.copy(input)
                input_i = input_i.to(device)
                
                output = nets[i](input_i)
                outputs.append(output)
                loss = criterion(output, target)
                prec1, prec5 = accuracy(output.data, target, topk=(1,5))
                top1s[i].update(prec1.item(), input_i.size(0))
                top5s[i].update(prec5.item(), input_i.size(0))
            test_loss += loss.item()
    # Save checkpoint.
    for i in range(net_num):
        if top1s[i].avg > best_accs[i]:
            best_accs[i] = top1s[i].avg
            state = {
                'net': nets[i].state_dict(),
                'acc': best_accs[i],
                'epoch': epoch,
            }
            if not os.path.isdir('save'):
                os.mkdir('save')
            torch.save(state, './save/{}_{}.tar'.format(args.algo, i))
            best_epoches[i] = epoch
    top1s = [round(x.avg, 2) for x in top1s]
    top5s = [round(x.avg, 2) for x in top5s]
    print('Test: best_accs: {} best_epoches: {} top1s: {} top5s: {} algo: {}'.format(best_accs, best_epoches, top1s, top5s, args.algo))

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

for epoch in range(start_epoch, start_epoch+epoches):
    train(epoch)
    test(epoch)
