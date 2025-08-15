#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from collections import OrderedDict
import copy
from torch.func import functional_call, vmap, grad
import numpy as np
import time

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
        self.trainloader, self.validloader, self.testloader, self.covloader = self.train_val_test_cov(
            dataset, list(idxs))
        self.device = 'cuda' if args.gpu else 'cpu'
        # Default criterion set to NLL loss function
        self.criterion = nn.NLLLoss().to(self.device)
        self.model_defference = OrderedDict()
        self.prob = .5
        self.cov_trace = 0
        self.criterion_cov = nn.NLLLoss(reduction='none').to(self.device)

    def train_val_test_cov(self, dataset, idxs):
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
        covloader = DataLoader(DatasetSplit(dataset, idxs_train),
                                 batch_size=len(idxs_train), shuffle=False)
        
        return trainloader, validloader, testloader, covloader

    def get_trace(self, model:nn.Module):
        
        images_cov = None
        labels_cov = None
        for batch_idx, (_images, _labels) in enumerate(self.covloader):
            images_cov = _images
            labels_cov = _labels

        images_cov, labels_cov = images_cov.to(self.device), labels_cov.to(self.device)
        model.zero_grad()
        model.eval()

        if self.device == 'cuda':
            params = {k: v.detach() for k, v in model.named_parameters()}
            buffers = {k: v.detach() for k, v in model.named_buffers()}

            def compute_loss_per_sample(params, buffers, sample, target):
                prediction = functional_call(model, (params, buffers), (sample.unsqueeze(0),))
                return nn.functional.nll_loss(prediction, target.unsqueeze(0))
            
            compute_grad_per_sample = grad(compute_loss_per_sample, argnums=0)
            compute_batch_grads = vmap(compute_grad_per_sample, in_dims=(None, None, 0, 0), randomness='different')
            per_sample_grads = compute_batch_grads(params, buffers, images_cov, labels_cov)

            flat_grads_list = [v.flatten(start_dim=1) for v in per_sample_grads.values()]
            per_sample_grads_tensor = torch.cat(flat_grads_list, dim=1)

            parameter_variances = torch.var(per_sample_grads_tensor, dim=0, correction=1)

            trace_of_covariance = torch.sum(parameter_variances)
            return trace_of_covariance

        elif self.device == 'cpu':
            log_probs = model(images_cov)
            per_sample_loss = self.criterion_cov(log_probs, labels_cov) # (batch_size,) 크기의 텐서
            
            per_sample_grads_list = []
            for i in range(self.covloader.batch_size):
                model.zero_grad()
                per_sample_loss[i].backward(retain_graph=True)
                current_sample_grad = []
                for param in model.parameters():
                    if param.grad is not None:
                        current_sample_grad.append(param.grad.detach().cpu().numpy().flatten())
                flat_grads = np.concatenate(current_sample_grad)
                per_sample_grads_list.append(flat_grads)
            per_sample_grads_ndarray = np.stack(per_sample_grads_list)

            parameter_variances = np.var(per_sample_grads_ndarray, axis=0, ddof=1)
            trace_of_covariance = np.sum(parameter_variances)
            
            return trace_of_covariance
    
    def set_prob(self, prob):
        self.prob = prob

    def update_weights(self, model:nn.Module, global_round:int):

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
            # batch 처리
            for batch_idx, (images, labels) in enumerate(self.trainloader):
                images, labels = images.to(self.device), labels.to(self.device)

                model.zero_grad()

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

        return model.state_dict(), sum(epoch_loss) / len(epoch_loss)
    
    def update_steps(self, model:nn.Module):
        if np.random.random() > self.prob:
            return None, None
        
        initial = copy.deepcopy(model)

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
            # batch 처리
            for batch_idx, (images, labels) in enumerate(self.trainloader):
                images, labels = images.to(self.device), labels.to(self.device)

                model.zero_grad()

                log_probs = model(images)
                loss = self.criterion(log_probs, labels)
                
                loss.backward()
                optimizer.step()

                self.logger.add_scalar('loss', loss.item())
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))

        model_difference = OrderedDict()
        for name in model.state_dict():
            model_difference[name] = model.state_dict()[name] - initial.state_dict()[name]
        
        print('participated')
        return model_difference, sum(epoch_loss) / len(epoch_loss)

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
