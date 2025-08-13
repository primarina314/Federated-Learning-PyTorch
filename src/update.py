#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from collections import OrderedDict
import copy
from torch.func import functional_call, vmap, grad

class DatasetSplit(Dataset):
    """An abstract Dataset class wrapped around Pytorch Dataset class.
    """

    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return torch.tensor(image), torch.tensor(label)


class LocalUpdate(object):
    def __init__(self, args, dataset, idxs, logger):
        self.args = args
        self.logger = logger
        self.trainloader, self.validloader, self.testloader = self.train_val_test(
            dataset, list(idxs))
        self.device = 'cuda' if args.gpu else 'cpu'
        # Default criterion set to NLL loss function
        self.criterion = nn.NLLLoss().to(self.device)
        self.model_defference = OrderedDict()
        self.prob = .5

    def train_val_test(self, dataset, idxs):
        """
        Returns train, validation and test dataloaders for a given dataset
        and user indexes.
        """
        # split indexes for train, validation, and test (80, 10, 10)
        idxs_train = idxs[:int(0.8*len(idxs))]
        idxs_val = idxs[int(0.8*len(idxs)):int(0.9*len(idxs))]
        idxs_test = idxs[int(0.9*len(idxs)):]

        trainloader = DataLoader(DatasetSplit(dataset, idxs_train),
                                 batch_size=self.args.local_bs, shuffle=True)
        validloader = DataLoader(DatasetSplit(dataset, idxs_val),
                                 batch_size=int(len(idxs_val)/10), shuffle=False)
        testloader = DataLoader(DatasetSplit(dataset, idxs_test),
                                batch_size=int(len(idxs_test)/10), shuffle=False)
        return trainloader, validloader, testloader

    def calc_grad(self, model:nn.Module, ):
        def compute_loss_stateless(params, buffers, image, label):
            # functional_call을 사용하여 상태 비저장 방식으로 모델 실행
            # image와 label은 배치 차원이 없는 단일 샘플 (예: ,)
            batch = image.unsqueeze(0)
            targets = label.unsqueeze(0)
            
            output = functional_call(model, (params, buffers), (batch,))
            loss = nn.CrossEntropyLoss()(output, targets)
            return loss
        
        compute_grad_stateless = grad(compute_loss_stateless, argnums=0)
        params = copy.deepcopy(model.named_parameters())
        buffers = copy.deepcopy(model.named_buffers())

        per_sample_grads = vmap(compute_grad_stateless, in_dims=(None, None, 0, 0))(params, buffers, images, labels)
        pass
    

    def update_weights(self, model:nn.Module, global_round:int):
        def compute_loss_stateless(params, buffers, image, label):
            # functional_call을 사용하여 상태 비저장 방식으로 모델 실행
            # image와 label은 배치 차원이 없는 단일 샘플 (예: ,)
            batch = image.unsqueeze(0)
            targets = label.unsqueeze(0)
            
            output = functional_call(model, (params, buffers), (batch,))
            # loss = nn.CrossEntropyLoss()(output, targets)
            loss = nn.NLLLoss()(output, targets)
            return loss
        
        initial = copy.deepcopy(model)
        # model.named_parameters()

        # Set mode to train model
        model.train()
        epoch_loss = []

        # Set optimizer for the local updates
        if self.args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=self.args.lr,
                                        momentum=0.5)
        elif self.args.optimizer == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=self.args.lr,
                                         weight_decay=1e-4)

        for iter in range(self.args.local_ep):
            batch_loss = []
            # TODO: 각 레이블마다 grad 혹은 diff

            # batch 처리
            for batch_idx, (images, labels) in enumerate(self.trainloader):
                images, labels = images.to(self.device), labels.to(self.device)

                model.zero_grad()
                params = {name: p for name, p in model.named_parameters()}
                buffers = {name: b for name, b in model.named_buffers()}

                compute_grad_stateless = grad(compute_loss_stateless, argnums=0)
                per_sample_grads = vmap(compute_grad_stateless, in_dims=(None, None, 0, 0))(params, buffers, images, labels)

                log_probs = model(images)
                loss = self.criterion(log_probs, labels)
                loss.backward()
                optimizer.step()

                if self.args.verbose and (batch_idx % 10 == 0):
                    print('| Global Round : {} | Local Epoch : {} | [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        global_round, iter, batch_idx * len(images),
                        len(self.trainloader.dataset),
                        100. * batch_idx / len(self.trainloader), loss.item()))
                self.logger.add_scalar('loss', loss.item())
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))

        for name in model.state_dict():
            # might need to quantization
            self.model_defference[name] = (model.state_dict()[name] - initial.state_dict()[name]) / self.args.lr
        return model.state_dict(), sum(epoch_loss) / len(epoch_loss)

    def inference(self, model):
        """ Returns the inference accuracy and loss.
        """

        model.eval()
        loss, total, correct = 0.0, 0.0, 0.0

        for batch_idx, (images, labels) in enumerate(self.testloader):
            images, labels = images.to(self.device), labels.to(self.device)

            # Inference
            outputs = model(images)
            batch_loss = self.criterion(outputs, labels)
            loss += batch_loss.item()

            # Prediction
            _, pred_labels = torch.max(outputs, 1)
            pred_labels = pred_labels.view(-1)
            correct += torch.sum(torch.eq(pred_labels, labels)).item()
            total += len(labels)

        accuracy = correct/total
        return accuracy, loss


def test_inference(args, model, test_dataset):
    """ Returns the test accuracy and loss.
    """

    model.eval()
    loss, total, correct = 0.0, 0.0, 0.0

    device = 'cuda' if args.gpu else 'cpu'
    criterion = nn.NLLLoss().to(device)
    testloader = DataLoader(test_dataset, batch_size=128,
                            shuffle=False)

    for batch_idx, (images, labels) in enumerate(testloader):
        images, labels = images.to(device), labels.to(device)

        # Inference
        outputs = model(images)
        batch_loss = criterion(outputs, labels)
        loss += batch_loss.item()

        # Prediction
        _, pred_labels = torch.max(outputs, 1)
        pred_labels = pred_labels.view(-1)
        correct += torch.sum(torch.eq(pred_labels, labels)).item()
        total += len(labels)

    accuracy = correct/total
    return accuracy, loss
