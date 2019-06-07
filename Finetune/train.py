import torch
from transition_network import TransitionCNN
from utilities import MovingAverage, RunningAverage, reshapeLabels, reshapeTransitionBatch, desired_labels, save_checkpoint, get_and_write_transition_distribution
from TransitionDataSet import Transitiondataset
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
import time
import os
import cv2
import torchvision.utils as vutils
from tensorboardX import SummaryWriter

import numpy as np
# train model



print('getting data set transition distributions.............')
'''
dataset
'''
#
# for i in range(8):
#     train_set = Transitiondataset('./train_file/train_'+str(i+1)+'.txt','./ground_truths/train_'+str(i+1)+'.txt')
#     train_loader = DataLoader(train_set, batch_size=10, num_workers=2)
#     train_loader_set.append(train_loader)
#     print("train_"+str(i+1)+" finished")
#
#
#
# for i in range(2):
#     test_set = Transitiondataset('./test_file/test_'+str(i+1)+'.txt','./ground_truths/test_'+str(i+1)+'.txt')
#     test_loader=DataLoader(test_set, batch_size=10, num_workers=2)
#     test_loader_set.append(test_loader)
#     print("test_" + str(i + 1) + " finished")

'''
device
'''
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

'''
model
'''
model = TransitionCNN()
'''

'''
model.load_state_dict(torch.load('./models/shot_boundary_detector_even_distrib.pt'))
model.to(device)

#
# total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
# print('total params', total_params)
# total_params = sum(p.numel() for p in model.parameters())
# print('all parameters',total_params )
ignored_params = list(map(id, model.softmaxConv.parameters()))
base_params = filter(lambda  p:id(p) not in ignored_params, model.parameters())

optimizer = optim.Adam([{'params': base_params},{'params': model.softmaxConv.parameters(), 'lr': 0.0001}], lr = 0.00001)

# create directories necessary for storing of checkpoints, csv data and model when ideal validation accuracy is reached

os.makedirs('finetune_models', exist_ok=True)
#

# define the train loop for the CNN
def train(optimizer, model, first_epoch = 1, num_epochs=10):

    criterion = nn.CrossEntropyLoss(weight =torch.Tensor([100, 1]).to(device))
    validation_accuraies = []
    old_accuracy = 0
    train_losses = []
    test_losses = []


    print('----------------------------------------------------------------------')

    for epoch in range(first_epoch, first_epoch + num_epochs):
        start_time_epoch = time.time()
        print('Epoch:', epoch)

        # put the model in train mode
        model.train()

        train_loss = MovingAverage()
        for i in range(8):
            train_set = Transitiondataset('./train_file/train_' + str(i + 1) + '.txt',
                                          './ground_truths/train_' + str(i + 1) + '.txt')
            train_loader = DataLoader(train_set, batch_size=10, num_workers=2)
            for index, batch in enumerate(train_loader):
                transitions = batch[0]
                labels = batch[1]
                inputs = reshapeTransitionBatch(transitions)
                labels = reshapeLabels(labels)
                optimizer.zero_grad()
                predictions = model(inputs.to(device))
                loss = criterion(predictions, labels.to(device))
                loss.backward()

                # update the weights of the model after backward propogation
                optimizer.step()

                #update the loss
                train_loss.update(loss)

                if index%100 == 0:
                    print('trainset_'+str(i+1)+' training loss at the end of batch', index, ':', train_loss)
            print("epoch"+str(epoch)+" trainset_" + str(i + 1) + " finished")


        print('Training Loss after epoch ',epoch, ':',train_loss)
        train_losses.append(train_loss.value)

        #convert the model to its validation phase
        model.eval()
#
        test_loss = MovingAverage()
#
#         # store all the predictions made by the CNN
        Accu = np.ndarray([2])
        for i in range(2):
            pred = []
            all_labels =[]
            test_set = Transitiondataset('./test_file/test_' + str(i + 1) + '.txt',
                                  './ground_truths/test_' + str(i + 1) + '.txt')
            test_loader = DataLoader(test_set, batch_size=10, num_workers=2)
            with torch.no_grad():
                for index, batch in enumerate(test_loader):

                    test_transitions = batch[0]
                    test_labels = batch[1]
                    test_inputs = reshapeTransitionBatch(test_transitions)
                    test_labels = reshapeLabels(test_labels)
                    all_labels.extend(test_labels.cpu().numpy())
                    test_predictions = model(test_inputs.to(device))
                    loss = criterion(test_predictions, test_labels.to(device))
                    # update the weights of the model after backward propogation
                    # update the loss
                    test_loss.update(loss)
                    pred.extend(test_predictions.argmax(dim=1).cpu().numpy())
                    if index % 100 == 0:
                        print('testset_' + str(i + 1) + ' training loss at the end of batch', index, ':', train_loss)

                print("epoch" + str(epoch) + " testset_" + str(i + 1) + " finished")
                all_labels = torch.LongTensor(all_labels)
                pred = torch.LongTensor(pred)
                accuracy = torch.mean((pred == all_labels).float())
                Accu[i] = accuracy.numpy()
                print("Testset_",str(i+1)+" Accuracy:",accuracy)

        if Accu.mean() > old_accuracy:
            torch.save(model.state_dict(), 'finetune_models/shot_boundary_detector_even_distrib.pt')
            old_accuracy = Accu.mean()
            print('save model')

train(model=model, optimizer=optimizer, num_epochs=5)
# end_time_train = time.time()
# total_train_time = end_time_train - start_time_train
#
# print('total train_time:', total_train_time)
