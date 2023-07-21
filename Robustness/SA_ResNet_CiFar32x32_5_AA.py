import torch
from simanneal import Annealer
import random
import torch.nn as nn
import time

from autoattack import AutoAttack


from Networks import AF_Tools
from Networks.GetData import *
import os.path
import torch.optim as optim

from Utils.Tools import write_data_in_file

from Networks.ResNet18_Multiple import resnet18



Activation_Functions = AF_Tools.Activation_Functions_
operators = AF_Tools.operators_
operators_p = AF_Tools.operators_p_

EPOCHS = 15


destination_path = '../save/resnet18_cf32x32_APGDE/zoo_res18_cf'
result_file_name = 'results_zoo_ResNet_Cifar32x32_15_APGDE.csv'
LAYERS = 4

trace = False


def train_model(model, data_loaders, dataset_sizes, criterion, optimizer, device):
    model = model.to(device)
    model.train()  # set train mode


    # for each epoch
    for epoch in range(EPOCHS):

        epoch_time = time.time()
        running_loss, running_corrects, total_checked = 0.0, 0, 0
        error_found = False
        # for each batch
        for indx, (inputs, labels) in enumerate(data_loaders['train']):
            #btch_time = time.time()
            # print('indx', indx, len(inputs))

            try:
                inputs, labels = inputs.to(device), labels.to(device)

                # making sure all the gradients of parameter tensors are zero
                optimizer.zero_grad()  # set gradient as 0

                # get the model output
                outputs = model(inputs)



                # get the prediction of model
                _, preds = torch.max(outputs, 1)

                # calculate loss of the output
                loss = criterion(outputs, labels)

                # backpropagation
                loss.backward()

                # update model parameters using optimzier
                optimizer.step()

                batch_loss_total = loss.item() * inputs.size(0)  # total loss of the batch
                running_loss += batch_loss_total  # cumluative sum of loss
                running_corrects += torch.sum(preds == labels.data)  # cumulative sum of correct count
                total_checked += inputs.size(0)

            except Exception as e:
                error_found = True
                print(' ****************************                  error      ****************************')
                print(str(e))
                break
            '''
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, indx * len(inputs), len(data_loaders['train'].dataset),
                       100. * indx / len(data_loaders['train']), loss.item()))
            '''

            #print(indx, 'btch time :', time.time()-btch_time)


        #print('----------------------------------------------epo time :', time.time() - epoch_time)
        if error_found:
            return -1
        # calculating the loss and accuracy for the epoch
        epoch_loss = running_loss / dataset_sizes['train']
        epoch_acc = running_corrects.double() * 100 / total_checked
        print('correct   :  ', running_corrects, '         dataset size  :   ', dataset_sizes['train'])
        # if trace:
        print('Epoch {}/{} -  Train Loss: {:.4f} Acc: {:.4f}'.format(epoch + 1, EPOCHS, epoch_loss, epoch_acc))
        print('time : ', time.time()-epoch_time )

    return 0


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



from robustbench.model_zoo.enums import BenchmarkDataset, ThreatModel
from typing import Callable, Dict, Optional, Sequence, Tuple, Union
from robustbench.data import CORRUPTIONS, get_preprocessing, load_clean_dataset, \
    CORRUPTION_DATASET_LOADERS
from robustbench.utils import clean_accuracy, load_model, parse_args, update_json


def local_benchmark(
    model,
    n_examples: int = 10000,
    dataset: Union[str, BenchmarkDataset] = BenchmarkDataset.cifar_10,
    threat_model: Union[str, ThreatModel] = ThreatModel.Linf,
    to_disk: bool = False,
    model_name: Optional[str] = None,
    data_dir: str = "./data",
    device: Optional[Union[torch.device, Sequence[torch.device]]] = None,
    batch_size: int = 32,
    eps: Optional[float] = None,
    log_path: Optional[str] = None,
    preprocessing: Optional[Union[str,
                                  Callable]] = None) -> Tuple[float, float]:
    """Benchmarks the given model(s).

    It is possible to benchmark on 3 different threat models, and to save the results on disk. In
    the future benchmarking multiple models in parallel is going to be possible.

    :param model: The model to benchmark.
    :param n_examples: The number of examples to use to benchmark the model.
    :param dataset: The dataset to use to benchmark. Must be one of {cifar10, cifar100}
    :param threat_model: The threat model to use to benchmark, must be one of {L2, Linf
    corruptions}
    :param to_disk: Whether the results must be saved on disk as .json.
    :param model_name: The name of the model to use to save the results. Must be specified if
    to_json is True.
    :param data_dir: The directory where the dataset is or where the dataset must be downloaded.
    :param device: The device to run the computations.
    :param batch_size: The batch size to run the computations. The larger, the faster the
    evaluation.
    :param eps: The epsilon to use for L2 and Linf threat models. Must not be specified for
    corruptions threat model.
    :param preprocessing: The preprocessing that should be used for ImageNet benchmarking. Should be
    specified if `dataset` is `imageget`.

    :return: A Tuple with the clean accuracy and the accuracy in the given threat model.
    """

    dataset_: BenchmarkDataset = BenchmarkDataset(dataset)
    threat_model_: ThreatModel = ThreatModel(threat_model)

    device = device or torch.device("cpu")
    model = model.to(device)

    prepr = get_preprocessing(dataset_, threat_model_, model_name,
                              preprocessing)

    clean_x_test, clean_y_test = load_clean_dataset(dataset_, n_examples,
                                                    data_dir, prepr)

    accuracy = clean_accuracy(model,
                              clean_x_test,
                              clean_y_test,
                              batch_size=batch_size,
                              device=device)
    print(f'Clean accuracy: {accuracy:.2%}')

    if threat_model_ in {ThreatModel.Linf, ThreatModel.L2}:
        if eps is None:
            raise ValueError(
                "If the threat model is L2 or Linf, `eps` must be specified.")

        adversary = AutoAttack(model,
                               norm=threat_model_.value,
                               eps=eps,
                               version='custom',
                               attacks_to_run=['apgd-ce'],
                               device=device,
                               log_path=log_path)
        x_adv = adversary.run_standard_evaluation(clean_x_test,
                                                  clean_y_test,
                                                  bs=batch_size)
        adv_accuracy = clean_accuracy(model,
                                      x_adv,
                                      clean_y_test,
                                      batch_size=batch_size,
                                      device=device)

    else:
        raise NotImplementedError
    print(f'Adversarial accuracy: {adv_accuracy:.2%}')

    if to_disk:
        if model_name is None:
            raise ValueError(
                "If `to_disk` is True, `model_name` should be specified.")

        update_json(dataset_, threat_model_, model_name, accuracy,
                    adv_accuracy, eps)

    return accuracy, adv_accuracy



class robust_simannealer(Annealer):
    # pass extra data (the distance matrix) into the constructor
    def __init__(self, state, data_loaders, dataset_sizes, atk_):
        self.dataloaders, self.dataset_sizes, (self.atk, self.eps) = data_loaders, dataset_sizes, atk_
        self.p1 = range(0, LAYERS)
        self.p2 = range(LAYERS, LAYERS*2)
        self.p3 = range(LAYERS*2, LAYERS*3)
        super(robust_simannealer, self).__init__(state)  # important!

    def move(self):
        moves_count = 2
        for i in range(moves_count):
            loc = random.randint(0, len(self.state) - 1)
            if loc in self.p1:  # the first part, which is for the operator
                self.state[loc] = random.choices(operators_p, weights=operators_p)[0]
                if self.state[loc] < 2:  # if operator is a.f(x) then remove the second activation function
                    self.state[loc + LAYERS*2] = 0

            elif loc in self.p2:  # first activation function
                self.state[loc] = random.randint(0, len(Activation_Functions) - 1)
            elif loc in self.p3:  # second activation function
                self.state[loc] = random.randint(0, len(Activation_Functions) - 1)
                if self.state[loc - LAYERS*2] < 2:  # if operator is a.f(x) there is no need to change the chromosome
                    self.state[loc - LAYERS*2] = random.randint(2, 4)
            '''
            for i in range(0, LAYERS):
                if self.state[i] == 2 and self.state[i+LAYERS] == 0 and self.state[i+LAYERS*2] == 2:
                    self.state[i + LAYERS] = 1
            '''

        print(self.state)




        # return self.energy() #- initial_energy

    def energy(self):
        total_time = time.time()
        """Calculates the length of the route."""
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # device = 'cpu'

        if trace:
            print('Individual    : ', self.state)

        #model = resnet18(num_classes=10, states_=self.state).to(device)
        from Networks.ZooResNet import zoo_ResNet, BasicBlock
        model = zoo_ResNet(BasicBlock, [2, 2, 2, 2], num_classes=10).to(device)


        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        fname = destination_path + '_'.join(map(str, self.state)) + '.pth'
        train_time = time.time()
        if os.path.isfile(fname):
            model.load_state_dict(torch.load(fname))
        else:
            res = train_model(model, self.dataloaders, self.dataset_sizes, criterion, optimizer, device)
            if res < 0:
                return 100
            torch.save(model.state_dict(), fname)

        model.eval()
        device = torch.device("cuda:0")
        clean_acc, robust_acc = local_benchmark(model, dataset='cifar10', threat_model='Linf', eps=8.0/255, batch_size=2000, device=device)

        write_data_in_file(result_file_name, (self.state, self.atk, self.eps, robust_acc, clean_acc))

        print(clean_acc, robust_acc)
        '''
        train_time = time.time() - train_time

        test_time = time.time()
        test_loss, test_acc = test_model(model, self.dataloaders, criterion, device, self.dataset_sizes)

        test_time = time.time() - test_time

        correct = 0
        total = 0

        eval_time = time.time()
        model.eval()

        a_t_k = AutoAttack(model, norm="Linf")
        a_t_k.attacks_to_run = ["apgd-ce", "apgd-t"]
        '''



        '''
        images, targets  = self.dataloaders['train']
        adv_images = adversary.run_standard_evaluation(images, targets, bs=len(images))
        logits_adv = model(adv_images)
        loss_adv = criterion(logits_adv, targets)
        '''

        '''

        for indx, (images, labels) in enumerate(self.dataloaders['test']):
            perturb_images = a_t_k.run_standard_evaluation(images, labels, bs=len(images)).to(device)
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
        write_data_in_file(result_file_name, (self.state, self.atk, self.eps, rob_acc, test_acc))
        # print(100 * float(correct) / total)
        print(self.state, self.atk, self.eps, test_acc, rob_acc)
        print('total time : ', total_time, ' train time : ', train_time, ' eval time : ', eval_time, 'test time :', test_time)

        '''
        #return 100 * ( 1 - rob_acc)
        return 100-robust_acc


if __name__ == '__main__':
    # initial state, a randomly-ordered itinerary
    torch.autograd.set_detect_anomaly(True)
    data_loaders_, dataset_sizes_ = get_cifar10_data_32(2000, 2000)
    # eps_s_ = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3]

    atk_eps_s = [("PGD", 8/255)]
    for atk_eps_ in atk_eps_s:

        state_ = [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0] # 79.61

        #state_ = [0.25, 2, 2, 1, 0.25, 1, 3,    1, 1, 4, 5, 0, 0, 6,      0, 2, 2, 0, 0, 0, 4]

        # state_ = [4, 4, 2, 6]
        # np.random.choice(len(Activation_Functions), ch_size_1, replace=True)

        tsp = robust_simannealer(state_, data_loaders_, dataset_sizes_, atk_eps_)
        # tsp.set_schedule(tsp.auto(minutes=0.2, steps=10))
        # since our state is just a list, slice is the fastest way to copy
        tsp.steps = 500
        tsp.copy_strategy = "deepcopy"
        state_, e = tsp.anneal()

        print()
        print("%i mile route:" % e)
        print("".join(state_))
