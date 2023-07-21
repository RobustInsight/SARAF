import torch
from simanneal import Annealer
import random
import torch.nn as nn
import time
import pandas as pd

from Utils.Tools import string_to_number

from Networks.GetData import *
import os.path
import torch.optim as optim

from Utils.Tools import write_data_in_file

from Networks.AlexNet_Multiple import AlexNetM

from torchattacks import FGSM, PGD
from Utils.Tools import read_data_from_file

Activation_Functions = [nn.ReLU(), nn.LeakyReLU(), nn.Sigmoid(), nn.SELU(), nn.CELU(), nn.Mish(), nn.GELU()]
operators = [0.25, 0.5, 0.75, 1, 2, 3, 4]
operators_p = [0.0625, 0.0625, 0.0625, 0.0625, 0.25, 0.25, 0.25]


LAYERS = 7

trace = False



def test_model(model, data_loaders, criterion, device, dataset_sizes):
    # after training epochs, test epoch starts
    model.eval()  # set test mode
    running_loss, running_corrects, total_checked = 0.0, 0, 0

    # for each batch
    for indx, (inputs, labels) in enumerate(data_loaders['test']):
        inputs = inputs.to(device)
        labels = labels.to(device)

        # same with the training part.
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * inputs.size(0)  # cumulative sum of loss
        running_corrects += torch.sum(preds == labels.data)  # cumulative sum of corrects count
        total_checked += inputs.size(0)


    # calculating the loss and accuracy
    test_loss = running_loss / dataset_sizes['test']
    test_acc = running_corrects.double() * 100 / total_checked

    return test_loss, test_acc.item()
    # print('<Test Loss: {:.4f} Acc: {:.4f}>'.format(test_loss, test_acc))


def get_attacks():
    return {'FGSM': FGSM,
            'PGD': PGD, }


def atk_by_name(atkname):
    attaks = get_attacks()
    return attaks[atkname]

def energy(model, state, dataloaders, atk, eps, test_acc):
    total_time = time.time()
    """Calculates the length of the route."""

    correct = 0
    total = 0

    eval_time = time.time()
    model.eval()

    attack = atk_by_name(atk)

    a_t_k = attack(model, eps=eps)

    for indx, (images, labels) in enumerate(dataloaders['test']):
        perturb_images = a_t_k(images, labels).to(device)
        # images = images.to(device)
        outputs = model(perturb_images)

        _, predicted = torch.max(outputs.data, 1)

        total += labels.size(0)
        correct += (predicted == labels.to(device)).sum()

    eval_time = time.time() - eval_time
    total_time = time.time() - total_time
    # print(total,correct )
    if trace:
        print('Robust accuracy: %.2f %%' % (100 * float(correct) / total))
        print('_' * 20)

    rob_acc = 100 * float(correct) / total
    write_data_in_file('results_FGSM.csv', (state, atk, eps, rob_acc, test_acc))
    # print(100 * float(correct) / total)
    print(state, atk, eps, test_acc, rob_acc)
    print('total time : ', total_time, ' train time : ', train_time, ' eval time : ', eval_time, 'test time :',
          test_time)



if __name__ == '__main__':
    # initial state, a randomly-ordered itinerary

    pd_data = read_data_from_file('results_AlexNet_Cifar_FGSM_base.csv')
    eps_1_255 = 1 / 255

    filter = pd_data['eps'] == str(eps_1_255)
    filtered_data = pd_data.loc[filter]
    chromosomes = pd.unique(filtered_data['chromosome'])

    eps_s = [1, 2, 3, 4, 5, 6, 7, 8]
    eps_s = [9, 10]

    data_loaders_, dataset_sizes_ = get_cifar10_data_32()
    for state_ in chromosomes:
        state_ = state_.split(',')
        state_ = string_to_number(state_)

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # device = 'cpu'

        model = AlexNetM(num_classes=10, states_=state_).to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        fname = '../save/alx_cf10' + '_'.join(map(str, state_)) + '.pth'
        train_time = time.time()
        if os.path.isfile(fname):
            model.load_state_dict(torch.load(fname))
        else:
            raise Exception('no file ' + state_)

        train_time = time.time() - train_time

        test_time = time.time()
        test_loss_, test_acc_ = test_model(model, data_loaders_, criterion, device, dataset_sizes_)

        test_time = time.time() - test_time

        for eps_ in eps_s:
            energy(model, state_, data_loaders_, 'FGSM', eps_/255, test_acc_)

