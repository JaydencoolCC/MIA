import os
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from torch import nn
import torch.optim as optim
from net.CNN import CNN_Model
from net.NN import NN_Model
from net.softMax import Softmax_Model
from net.resnet import ResNet18
#from opacus import PrivacyEngine
from loss import CrossEntropy_L2


def train_model(dataset, n_hidden=50, batch_size=128, epochs=100, learning_rate=0.01, modelType='cnn', model='DP',
                l2_ratio=1e-7):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if modelType == 'cnn' or modelType == 'nn':
        # train_x, train_y,or_trainx,or_trainy, test_x, test_y = dataset
        train_x, train_y, test_x, test_y = dataset
    else:
        train_x, train_y,or_trainx,or_trainy, test_x, test_y = dataset

    input_dim = train_x.shape
    out_dim = len(np.unique(train_y))

    if batch_size > len(train_y):
        batch_size = len(train_y)
    print('Building model with {} training data, {} classes...'.format(len(train_x), out_dim))

    if modelType == 'cnn':
        print('Using a multilayer convolution neural network based model...')
        #net = CNN_Model(input_dim, n_hidden, out_dim)
        net = ResNet18(10)
    elif modelType == 'nn':
        #全连接网络
        print('Using a multilayer neural network based model...')
        net = NN_Model(input_dim, n_hidden, out_dim)
    else:
        print('Using a single layer softmax based model...')
        net = Softmax_Model(input_dim, out_dim)

    # create loss function
    m = input_dim[0]

    # ------------start train----------------------------
    #差分隐私
    if model == 'DP':
        criterion = CrossEntropy_L2(net, m, l2_ratio).to(device)
        # criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.5)
        # optimizer = optim.Adam(net.parameters(), lr=learning_rate)

        visual_batch_size = 300
        visual_batch_rate = int(visual_batch_size / batch_size)

        privacy_engine = PrivacyEngine(
            net,
            batch_size=visual_batch_size,  # batch_size=256    epoch=90  lr=0.001
            sample_size=len(train_x),
            alphas=[1 + x / 10.0 for x in range(1, 100)] + list(range(12, 64)),
            noise_multiplier=1.1,  # 1.1
            max_grad_norm=1.0,  # 1.0
            secure_rng=False,
            target_delta=1e-5
        )
        privacy_engine.attach(optimizer)

        print('Training...')
        net.train()

        temp_loss = 0.0
        top1_acc = []
        losses = []

        for epoch in range(epochs):

            for i, (input_batch, target_batch) in enumerate(iterate_minibatches(train_x, train_y, batch_size)):

                input_batch, target_batch = torch.tensor(input_batch).contiguous(), torch.tensor(target_batch).type(
                    torch.long)
                input_batch, target_batch = input_batch.to(device), target_batch.to(device)

                # empty parameters in optimizer
                # optimizer.zero_grad()

                outputs = net(input_batch)
                # outputs [100, 10]

                # calculate loss value
                loss = criterion(outputs, target_batch)
                preds = np.argmax(outputs.detach().cpu().numpy(), axis=1)
                acc1 = (preds == target_batch.detach().cpu().numpy()).mean()
                losses.append(loss.item())
                top1_acc.append(acc1)

                # back propagation
                loss.backward()

                # update paraeters in optimizer(update weight)
                if ((i + 1) % visual_batch_rate == 0) or ((i + 1) == int(len(train_x) / batch_size)):
                    optimizer.step()
                else:
                    optimizer.virtual_step()

                temp_loss += loss.item()

                if (i + 1) % 5 == 0:
                    epsilon, best_alpha = optimizer.privacy_engine.get_privacy_spent(1e-5)
                    print(
                        f"\tTrain Epoch: {epoch} \t"
                        f"Loss: {np.mean(losses):.6f} "
                        f"Acc@1: {np.mean(top1_acc):.6f} "
                        f"(ε = {epsilon:.2f}, δ = {1e-5}) for α = {best_alpha}"
                    )

            temp_loss = 0.0

        net.eval()
    else:
        # criterion = CrossEntropy_L2(net, m, l2_ratio).to(device)
        criterion = nn.CrossEntropyLoss()
        net.to(device)
        # create optimizer
        optimizer = optim.Adam(net.parameters(), lr=learning_rate)
        # save the mean value of loss in an epoch
        running_loss = []
        running_accuracy = []
        temp_loss = 0.0 # count loss in an epoch
        iteration = 0  # count the iteration number in an epoch

        print('Training is ongoing .......')
        net.train()
        for epoch in range(epochs):

            for input_batch, target_batch in iterate_minibatches(train_x, train_y, batch_size):
                input_batch, target_batch = torch.tensor(input_batch), torch.tensor(target_batch).type(torch.long)
                input_batch, target_batch = input_batch.to(device), target_batch.to(device)

                optimizer.zero_grad()
                outputs = net(input_batch)
                loss = criterion(outputs, target_batch) # calculate loss value
                loss.backward() # back propagation
                optimizer.step() # update paraeters in optimizer(update weight)
                temp_loss += loss.item()

            temp_loss = round(temp_loss, 3)
            #if epoch % 5 == 0:
            print('Epoch:  {}, train loss:  {}'.format(epoch, temp_loss))
            temp_loss = 0.0
            if epoch % 10 == 0:
                net.eval()  # 验证训练集精度
                pred_y = []
                with torch.no_grad():
                    for input_batch, _ in iterate_minibatches(train_x, train_y, batch_size, shuffle=False):
                        input_batch = torch.tensor(input_batch)
                        input_batch = input_batch.to(device)
                        outputs = net(input_batch)
                        pred_y.append(torch.max(outputs, 1)[1].data.cpu().numpy())
                    pred_y = np.concatenate(pred_y)
                print('Training Accuracy: {}'.format(accuracy_score(train_y, pred_y)))
                #验证测试集精度
                if modelType == 'cnn':
                    pred_y = []
                    with torch.no_grad():
                        for input_batch, _ in iterate_minibatches(test_x, test_y, batch_size, shuffle=False):
                            input_batch = torch.tensor(input_batch)
                            input_batch = input_batch.to(device)
                            outputs = net(input_batch)
                            pred_y.append(torch.max(outputs, 1)[1].data.cpu().numpy())
                    pred_y = np.concatenate(pred_y)
                    print('Testing Accuracy: {}'.format(accuracy_score(test_y, pred_y)))
    #训练结束
    net.eval()
    pred_y = []
    if test_x is not None:
        print('Testing...')
        if batch_size > len(test_y):
            batch_size = len(test_y)
        with torch.no_grad():
            for input_batch, _ in iterate_minibatches(test_x, test_y, batch_size, shuffle=False):
                input_batch = torch.tensor(input_batch)
                input_batch = input_batch.to(device)
                outputs = net(input_batch)
                pred_y.append(torch.max(outputs, 1)[1].data.cpu().numpy())
            pred_y = np.concatenate(pred_y)
        print('Testing Accuracy: {}'.format(accuracy_score(test_y, pred_y)))

    print('More detailed results:')
    print(classification_report(test_y, pred_y))
    return net

def iterate_minibatches(inputs, targets, batch_size, shuffle=True):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)

    start_idx = None
    for start_idx in range(0, len(inputs) - batch_size + 1, batch_size):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batch_size]
        else:
            excerpt = slice(start_idx, start_idx + batch_size)
        yield inputs[excerpt], targets[excerpt]

    if start_idx is not None and start_idx + batch_size < len(inputs):
        excerpt = indices[start_idx + batch_size:] if shuffle else slice(start_idx + batch_size, len(inputs))
        yield inputs[excerpt], targets[excerpt]

def load_data(data_name):
    with np.load(data_name) as f:
        train_x, train_y = [f['arr_%d' % i] for i in range(len(f.files))]
    return train_x, train_y

def train_target_model(dataset,
                       epochs=100,
                       batch_size=128,
                       learning_rate=0.01,
                       l2_ratio=1e-7,
                       n_hidden=50,
                       modelType='nn',
                       model='DP'):
    # train_x, train_y, ori_trainx, ori_trainy, test_x, test_y = dataset
    train_x, train_y, test_x, test_y = dataset
    classifier_net = train_model(dataset=dataset,
                                 n_hidden=n_hidden,
                                 epochs=epochs,
                                 learning_rate=learning_rate,
                                 batch_size=batch_size,
                                 modelType=modelType,
                                 model=model,
                                 l2_ratio=l2_ratio)

    print("-----------train is over------------------")
    # test data for attack model
    attack_x, attack_y = [], []
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    classifier_net.eval()
    # data used in training, label is 1
    for batch, _ in iterate_minibatches(train_x,train_y, batch_size, False):
        batch = torch.tensor(batch)
        # batch = Variable(batch)
        batch = batch.to(device)

        output = classifier_net(batch)
        preds_tensor = nn.functional.softmax(output, dim=1)

        attack_x.append(preds_tensor.detach().cpu().numpy())
        attack_y.append(np.ones(len(batch))) #标签为1

    # data not used in training, label is 0
    for batch, _ in iterate_minibatches(test_x, test_y, batch_size, False):
        batch = torch.tensor(batch)
        # batch = Variable(batch)
        batch = batch.to(device)

        output = classifier_net(batch)
        preds_tensor = nn.functional.softmax(output, dim=1)

        attack_x.append(preds_tensor.detach().cpu().numpy())
        attack_y.append(np.zeros(len(batch)))

    attack_x = np.vstack(attack_x)
    attack_y = np.concatenate(attack_y)
    attack_x = attack_x.astype('float32')
    attack_y = attack_y.astype('int32')
    return attack_x, attack_y, classifier_net

def trainTarget(modelType,
                trainData, trainLabel,
                testData=[], testLabel=[],
                splitData=False,
                test_size=0.5,
                inepochs=50,
                batch_size= 300,
                learning_rate=0.001,    #0.001
                model='DP'):
    if splitData:
        data_train, data_test, label_train, label_test = \
            train_test_split(trainData, trainLabel, test_size=test_size, random_state=42)

    else:
        data_train = trainData
        label_train = trainLabel
        data_test = testData
        label_test = testLabel
    dataset = (data_train.astype(np.float32),
               label_train.astype(np.int32),
               data_test.astype(np.float32),
               label_test.astype(np.int32))

    attack_x, attack_y, theModel = train_target_model(dataset=dataset,
                                                      epochs=inepochs,
                                                      batch_size=batch_size,
                                                      learning_rate=learning_rate,
                                                      n_hidden=128,
                                                      l2_ratio=1e-07,
                                                      modelType=modelType,
                                                      model=model
                                                      )

    return attack_x, attack_y, theModel

def initializeTargetModel(dataset,
                          num_epoch,
                          dataFolderPath='./data/',
                          modelFolderPath='./model/',
                          classifierType='cnn',
                          model='DP'):

    dataPath = dataFolderPath + dataset + '/Preprocessed'  #数据路径

    attackerModelDataPath = dataFolderPath + dataset+'/attackerModelData'

    modelPath = modelFolderPath + dataset #模型路径

    try:
        os.makedirs(attackerModelDataPath)
        os.makedirs(modelPath)
    except OSError:
        pass
    print("Training the Target model for {} epoch".format(num_epoch))

    targetTrain, targetTrainLabel = load_data(dataPath + '/targetTrain.npz') #训练数据
    targetTest, targetTestLabel = load_data(dataPath + '/targetTest.npz') #测试数据

    attackModelDataTarget, attackModelLabelsTarget, targetModelToStore = trainTarget(classifierType,
                                                                                     targetTrain,
                                                                                     targetTrainLabel,
                                                                                     # or_Train = or_targetTrain,
                                                                                     # or_TrainLabel =or_targetTrainLabel,
                                                                                     testData=targetTest,
                                                                                     testLabel=targetTestLabel,
                                                                                     splitData=False,
                                                                                     inepochs=num_epoch,
                                                                                     batch_size=256,
                                                                                     model=model)

    np.savez(attackerModelDataPath + '/targetModelData.npz',
             attackModelDataTarget, attackModelLabelsTarget)
    # np.savez('dataPurchase/Purchase50/10005/targetModelData.npz',
    #          attackModelDataTarget, attackModelLabelsTarget)

    torch.save(targetModelToStore, modelPath + '/resnet18_targetModel.pth')

    return attackModelDataTarget, attackModelLabelsTarget, targetModelToStore