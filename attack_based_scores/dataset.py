import numpy as np
import pickle
import os
import random
from model import *
def shuffleAndSplitData(data, label, cluster=10000, partition=False):
   # shuffle data
    c = list(zip(data, label))
    random.shuffle(c)
    data, label = zip(*c)
    if(partition):
        data = np.array(data[:cluster])
        label = np.array(label[:cluster])
    else:
        data = np.array(data)
        label = np.array(label)
        return data, label

def clipDataTopX(dataToClip, top=3):
    res = [sorted(s, reverse=True)[0:top] for s in dataToClip]
    return np.array(res)

def readCIFAR10(data_path):
    for i in range(5):
        f = open(data_path + '/data_batch_' + str(i + 1), 'rb')
        train_data_dict = pickle.load(f, encoding='iso-8859-1')
        f.close()
        if i == 0:
            train_data = train_data_dict["data"]
            train_label = train_data_dict["labels"]
            continue

        train_data = np.concatenate((train_data, train_data_dict["data"]), axis=0)
        train_label = np.concatenate((train_label, train_data_dict["labels"]), axis=0)

    f = open(data_path + '/test_batch', 'rb')
    test_data_dict = pickle.load(f, encoding='iso-8859-1')
    f.close()

    test_data = np.array(test_data_dict["data"])
    test_label = np.array(test_data_dict["labels"])
    return train_data, train_label, test_data, test_label

def preprocessingCIFAR(toTrainData,toTestData):
    def reshape_for_save(raw_data):
        raw_data = np.dstack(
            (raw_data[:, :1024], raw_data[:, 1024:2048], raw_data[:, 2048:]))
        raw_data = raw_data.reshape(
            (raw_data.shape[0], 32, 32, 3)).transpose(0, 3, 1, 2)
        return raw_data.astype(np.float32)

    offset = np.mean(reshape_for_save(toTrainData), 0)
    scale = np.std(reshape_for_save(toTrainData), 0).clip(min=1)
    #标准化
    def rescale(raw_data):
        return (reshape_for_save(raw_data) - offset) / scale

    return rescale(toTrainData),rescale(toTestData)

def getData(dataset,
                   orginialDatasetPath,
                   dataFolderPath='./data/'):
    if dataset == 'CIFAR10':
        print("Loading data")
        data_path = orginialDatasetPath + 'cifar-10-batches-py'
        train_data, train_label, test_data, test_label = readCIFAR10(data_path)  # 读取训练集、测试集)
        dataPath = dataFolderPath + dataset + '/Preprocessed'   #./data/CIFAR10/Preprocessed
        toTrainData, toTrainLabel = shuffleAndSplitData(train_data, train_label,False)
        toTestData, toTestLabel =  shuffleAndSplitData(test_data, test_label,False)
        toTrainDataSave, toTestDataSave = preprocessingCIFAR(toTrainData, toTestData)

    elif dataset == 'CIFAR100':
        print("Loading data")
        data_path = orginialDatasetPath + 'cifar-10-batches-py'
        train_data, train_label, test_data, test_label = readCIFAR100(data_path)  # 读取训练集、测试集)
        dataPath = dataFolderPath + dataset + '/Preprocessed'   #./data/CIFAR10/Preprocessed
        toTrainData, toTrainLabel = shuffleAndSplitData(train_data, train_label,False)
        toTestData, toTestLabel =  shuffleAndSplitData(test_data, test_label,False)
        toTrainDataSave, toTestDataSave = preprocessingCIFAR(toTrainData, toTestData)

    elif dataset == 'MINST':
        print("Loading data")
        data_path = orginialDatasetPath + 'MINST'
        train_data, train_label, test_data, test_label = readMINST(orginialDatasetPath)  # 读取训练集、测试集)
        dataPath = dataFolderPath + dataset + '/Preprocessed'  # ./data/CIFAR10/Preprocessed
        toTrainData, toTrainLabel = shuffleAndSplitData(train_data, train_label, False)
        toTestData, toTestLabel = shuffleAndSplitData(test_data, test_label, False)
        toTrainDataSave, toTestDataSave = preprocessingCIFAR(toTrainData, toTestData)
    try:
        os.makedirs(dataPath)
    except OSError:
        pass

    print("--------------start save data-----------------")
    np.savez(dataPath + '/targetTrain.npz', toTrainDataSave, toTrainLabel)
    np.savez(dataPath + '/targetTest.npz', toTestDataSave, toTestLabel)

    print("Preprocessing finished\n\n")

def generateAttackData(dataset, classifierType, dataFolderPath, pathToLoadData, num_epoch, preprocessData,
                       trainTargetModel, model='normal' , topX=3):
    print(dataset, dataFolderPath, preprocessData)

    attackerModelDataPath = dataFolderPath + dataset +'/attackerModelData' #攻击数据的路径


    if (preprocessData):
        getData(dataset, pathToLoadData)

    if (trainTargetModel):
        targetX, targetY, trainModel= initializeTargetModel(dataset,
                                                 num_epoch,
                                                 classifierType=classifierType,
                                                 model=model)
    else:
        targetX, targetY = load_data(attackerModelDataPath + '/targetModelData.npz')


    targetX = clipDataTopX(targetX, top=topX)
    # shadowX = clipDataTopX(shadowX, top=topX)

    #return targetX, targetY, shadowX, shadowY
    return targetX, targetY, 1,2, trainModel