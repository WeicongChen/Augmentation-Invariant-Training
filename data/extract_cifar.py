'''Extract CIFAR images
'''
import numpy as np
import os
import random
import cv2
import sys
 
def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        if sys.version_info[0] == 2:
            dict = pickle.load(fo)
        elif sys.version_info[0] == 3:
            dict = pickle.load(fo, encoding='bytes')
    return dict
def get_data(file, num_classes=10):
    absFile = file
    dict = unpickle(absFile)
    X = np.asarray(dict[b'data'].T).astype("uint8")
    number = X.shape[1]
    if num_classes == 10:
        Yraw = np.asarray(dict[b'labels'])
    elif num_classes == 100:
        Yraw = np.asarray(dict[b'fine_labels'])
    Y = np.zeros((num_classes, number))
    for i in range(number):
        Y[Yraw[i],i] = 1
    names = np.asarray(dict[b'filenames'])
    return X,Yraw,names
def visualize_image(X, Y, names, savedir, savefile):
    if not os.path.exists('lists/'):
        os.makedirs('lists/')
    with open(savefile, 'a') as fw:
        for id in range(len(Y)):
            rgb = X[:,id]
            label = Y[id]
            img = rgb.reshape(3,32,32).transpose([1, 2, 0])
            name = names[id].decode('ascii')
            imgname = savedir + name 
            imgdir = imgname.replace(imgname.split('/')[-1], '')
            if not os.path.exists(imgdir):
                os.makedirs(imgdir)
            cv2.imwrite(imgname, img[...,::-1])
            fw.write('{} {}\n'.format(name, label))

def cifar10():
    print('Extract CIFAR-10 data...')
    num_classes = 10
    for i in range(6):
        if i == 5:
            X,Y,names = get_data('cifar-{}-batches-py/test_batch'.format(num_classes), num_classes)
            data = 'test'
        else:
            X,Y,names = get_data('cifar-{}-batches-py/data_batch_{}'.format(num_classes, i+1), num_classes)
            data = 'train'
        savedir = 'cifar{}-pics/{}/'.format(num_classes, data)
        savefile = 'lists/cifar{}_{}.txt'.format(num_classes, data)        
        visualize_image(X, Y, names, savedir, savefile)

def cifar100():
    print('Extract CIFAR-100 data...')
    num_classes = 100
    for data in ['train', 'test']:
        X,Y,names = get_data('cifar-{}-python/{}'.format(num_classes, data), num_classes)
        savedir = 'cifar{}-pics/{}/'.format(num_classes, data)
        savefile = 'lists/cifar{}_{}.txt'.format(num_classes, data)        
        visualize_image(X, Y, names, savedir, savefile)

if __name__ == '__main__':
    cifar10()
    cifar100()

