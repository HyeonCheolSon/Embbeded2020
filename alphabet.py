#-*- coding: utf-8 -*-
import cv2
import numpy as np
import glob
import sys

FNAME = 'alphabets.npz'

def machineLearning():
    img = cv2.imread('alphabets.png')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    cells = [np.hsplit(row,50) for row in np.vsplit(gray,26)]
    x = np.array(cells)
    train = x[:,:].reshape(-1,196).astype(np.float32)
    print("train: ", train)
    print("train shape: ",train.shape)

    k = np.arange(26)
    train_labels = np.repeat(k,50)[:,np.newaxis]

    np.savez(FNAME,train=train,train_labels = train_labels)

def resize20(pimg):
    img = cv2.imread(pimg)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    grayResize = cv2.resize(gray,(14,14))
    ret, thresh = cv2.threshold(grayResize, 125, 255,cv2.THRESH_BINARY_INV)

    cv2.imshow('num',thresh)
    return thresh.reshape(-1,196).astype(np.float32)

def loadTrainData(fname):
    with np.load(fname) as data:
        train = data['train']
        train_labels = data['train_labels']

    return train, train_labels

def checkDigit(test, train, train_labels):
    knn = cv2.ml.KNearest_create()
    knn.train(train, cv2.ml.ROW_SAMPLE, train_labels)

    ret, result, neighbours, dist = knn.findNearest(test, k=5)

    return result

if __name__ == '__main__':
    if len(sys.argv) == 1:
        print ('option : train or test')
        exit(1)
    elif sys.argv[1] == 'train':
        machineLearning()
    elif sys.argv[1] == 'test':
        train, train_labels = loadTrainData(FNAME)

        saveNpz = False
        for fname in glob.glob('ALPHABET_3.png'):
            test = resize20(fname)
            result = checkDigit(test, train, train_labels)

            print (result)

            k = cv2.waitKey(0)

            if k >= 65 and k<=68:
                saveNpz = True
                toss_num = 0
                if k == 65:
                    toss_num=1
                elif k == 66:
                    toss_num = 2
                elif k == 67:
                    toss_num = 3
                elif k == 68:
                    toss_num = 4
                train = np.append(train, test, axis=0)
                newLabel = np.array(toss_num).reshape(-1,1)
                train_labels = np.append(train_labels, newLabel,axis=0)

        cv2.destroyAllWindows()
        if saveNpz:
            np.savez(FNAME,train=train, train_labels=train_labels)
    else:
        print ('unknown option')

        
