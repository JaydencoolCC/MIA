#！-*- coding:utf-8 -*-
"""
Created on 5 Dec 2018

@author: Wentao Liu, Ahmed Salem
"""
import argparse
from sklearn.metrics import roc_auc_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.datasets import fetch_20newsgroups
import os
import random
from sklearn.model_selection import train_test_split
import pickle
import numpy as np
import sys
from dataset import generateAttackData
import torch
from torchvision.datasets import CIFAR10, CIFAR100, MNIST
import torchvision.transforms as transforms
from  collections import  Counter
from train import trainAttackModel
from sklearn.metrics import accuracy_score, classification_report

seed = 21
sys.dont_write_bytecode = True
np.random.seed(seed)

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='CIFAR10',
                    help='Which dataset to use (CIFAR10,CIFAR100,MINST,LFW,Adult,or News)')
parser.add_argument('--classifierType', default='cnn',
                    help='Which classifier cnn or cnn')
parser.add_argument('--dataset2', default='CIFAR10',
                    help='Which second dataset for adversary 2 (CIFAR10 or News)')
parser.add_argument('--classifierType2', default='nn',
                    help='Which classifier cnn or nn')
parser.add_argument('--dataFolderPath', default='./data',
                    help='Path to store data')
parser.add_argument('--pathToLoadData', default='../../data/',# cifar-10-batches-py,cifar-100-python,MINST,lfw,Adult,News
                    help='Path to load dataset from')
parser.add_argument('--num_epoch', type=int, default= 50,
                    help='Number of epochs to train shadow/target models')
parser.add_argument('--preprocessData',  default=True,action='store_true',
                    help='Preprocess the data, if false then load preprocessed data')
parser.add_argument('--trainTargetModel', default=True,action='store_true',
                    help='Train a target model, if false then load an already trained model')
parser.add_argument('--trainShadowModel', default=False,action='store_true',
                    help='Train a shadow model, if false then load an already trained model')
parser.add_argument('--model', default='normal',help='train a target model/shadow model use DP or normal')
opt = parser.parse_args()


def save_attack_data(Dataser1X, Dataset1Y, Dataset2X, Dataset2Y, dataset, cluster, dataFolderPath) :
    attackerModelDataPath = dataFolderPath + dataset + '/' + str(cluster)
    try:
        os.makedirs(attackerModelDataPath)
    except OSError:
        pass
    np.savez(attackerModelDataPath + '/targetModelData.npz',
             Dataser1X, Dataset1Y)
    np.savez(attackerModelDataPath + '/shadowModelData.npz',
             Dataset2X, Dataset2Y)


def top1_threshold_attack(x_, target_model):
    # 为每个数据集生成均匀分布的随机数据点
    #  # mnist为:1*28*28  cifar10和cifar100都为:3*32*32
    # 然后将这些数据点输入到目标模型网络中获取每个数据的最大后验概率值
    #生成的数据满足均匀分布
    nonM_generated = np.random.uniform(0, 255, (1000, 3, 32, 32)) # cifar
    input = torch.tensor(nonM_generated).cuda().type(torch.float32)
    target_model.eval()
    with torch.no_grad():
        outputs = torch.nn.functional.softmax(target_model(input), dim=1)
        pre = torch.max(outputs, 1)[0].data.cpu().numpy()
    # 通过设置百分位数t,来计算阈值的公式:b[(len(b) - 1 ) * q % + 1] 20%
    threshold = np.percentile(pre, 20, interpolation='lower')  # linear, lower, higher, midpoint, nearest
    print('threshold=', threshold)
    # 如果最大后验概率值大于阈值,则标签为1，否则设置为0
    predict = np.where(x_.max(axis=1) > threshold, 1, 0)

    return predict

def attackerThree(dataset='CIFAR10',
                  classifierType='cnn',
                  dataFolderPath='./data/',
                  pathToLoadData='./data/cifar-10-batches-py-official',
                  num_epoch=50,
                  preprocessData=True,
                  trainTargetModel=True,
                  model='normal'):

    targetX, targetY, _, _,trainModel = \
        generateAttackData(dataset,
                           classifierType,
                           dataFolderPath,
                           pathToLoadData,
                           num_epoch,
                           preprocessData,
                           trainTargetModel,
                           topX=3)

    #targetX (n,10) targetY(n)---> 0 or 1
    targetX = top1_threshold_attack(targetX, trainModel)
    print('AUC = {}'.format(roc_auc_score(targetY, targetX)))
    mc = classification_report(targetX, targetX)

if __name__ == '__main__':
    attackerThree(dataset=opt.dataset,
                  classifierType=opt.classifierType,
                  dataFolderPath=opt.dataFolderPath,
                  pathToLoadData=opt.pathToLoadData,
                  num_epoch=opt.num_epoch,
                  preprocessData=opt.preprocessData,
                  trainTargetModel=opt.trainTargetModel)
