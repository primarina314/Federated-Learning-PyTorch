#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6


import os
import copy
import time
import pickle
import numpy as np
from tqdm import tqdm

import torch
from tensorboardX import SummaryWriter

from options import args_parser
from update import LocalUpdate, test_inference
from models import MLP, CNNMnist, CNNFashion_Mnist, CNNCifar
from utils import get_dataset, average_weights, exp_details
import copy

if __name__ == '__main__':
    start_time = time.time()

    # define paths
    path_project = os.path.abspath('.')
    logger = SummaryWriter('./logs')

    args = args_parser()
    exp_details(args)

    if args.gpu:
        torch.cuda.set_device(args.gpu)
    device = 'cuda' if args.gpu else 'cpu'

    # load dataset and user groups
    train_dataset, test_dataset, user_groups = get_dataset(args)

    # BUILD MODEL
    if args.model == 'cnn':
        # Convolutional neural netork
        if args.dataset == 'mnist':
            global_model = CNNMnist(args=args)
        elif args.dataset == 'fmnist':
            global_model = CNNFashion_Mnist(args=args)
        elif args.dataset == 'cifar':
            global_model = CNNCifar(args=args)

    elif args.model == 'mlp':
        # Multi-layer preceptron
        img_size = train_dataset[0][0].shape
        len_in = 1
        for x in img_size:
            len_in *= x
            global_model = MLP(dim_in=len_in, dim_hidden=64,
                               dim_out=args.num_classes)
    else:
        exit('Error: unrecognized model')

    # Set the model to train and send it to device.
    global_model.to(device)
    global_model.train()
    print(global_model)

    # copy weights
    global_weights = global_model.state_dict()

    # Training
    train_loss, train_accuracy = [], []
    val_acc_list, net_list = [], []
    cv_loss, cv_acc = [], []
    print_every = 2
    val_loss_pre, counter = 0, 0

    """
    TODO: Hyperparams
    Power constraints
    Channel model
    
    """

    for epoch in tqdm(range(args.epochs)):
        local_weights, local_losses = [], []
        print(f'\n | Global Training Round : {epoch+1} |\n')

        global_model.train()
        """
        TODO: Sampling
        아래는 중앙 서버에서 update 에 참여할 user 를 샘플링하는 방식(Centralized Sampling)
        Decentralized sampling 으로 수정해야 함

        TODO: Power constraint, channel
        채널 상태 반영, unbiased - gamma 설정
        probability 설정
        """
        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)
        users_trace = np.zeros(shape=(args.num_users))
        users_prob = np.zeros(shape=(args.num_users))
        

        for idx in range(args.num_users):
            local_model = LocalUpdate(args=args, dataset=train_dataset,
                                      idxs=user_groups[idx], logger=logger)
            users_trace[idx] = local_model.get_trace(global_model)
        max_trace = users_trace.max()
        mean_trace = users_trace.mean()

        for idx in range(args.num_users):
            # decentralized user-sampling
            local_model = LocalUpdate(args=args, dataset=train_dataset,
                                      idxs=user_groups[idx], logger=logger)
            p = users_trace[idx] / max_trace / 5
            if np.random.random() > p:
                continue
            # TODO: difference 신호 선형결합 저장 -> mu 로 나눠서 step 이동
            w, loss = local_model.update_weights(
                model=copy.deepcopy(global_model), global_round=epoch)
            local_weights.append(copy.deepcopy(w))
            local_losses.append(copy.deepcopy(loss))


        for idx in idxs_users:
            local_model = LocalUpdate(args=args, dataset=train_dataset,
                                      idxs=user_groups[idx], logger=logger)
            
            """
            TODO: 다른 상황과 비교
            1. 평균 참여확률 동일하게 맞춘 상태에서 균일 참여 확률 - 같은 mu, 같은 power consumption
            2. 레이블 분포에 의한 확률 결정 방식과 비교
            3. 
            * mean 을 바탕으로 PS 가 threshold 설정
            ** C 값까지 고려한 확률분배 및 스텝 이동 또는 idx_users 에 append. 후자가 나아 보임
            """

            # 모든 usr trace 계산 추가에 의한 소요 시간 차이: 126.4048 -> 134.6846
            """
            TODO: LDP 상황 적용
            w(tensor), loss 가 아니라, diff(tensor) 를 리턴
            """

            w, loss = local_model.update_weights(
                model=copy.deepcopy(global_model), global_round=epoch)
            """
            TODO: Aggregation
            지금은 OTA 환경으로 설정되지 않았고, 각 usr 가 weight 를 직접 보냄. GM 적용도 없음.
            1. weight 가 아니라 grad + noise 에 a_k 곱해서 보내도록 설정
            2. OTA 가정해서 channel gain 및 LC.
            3. AWGN 추가
            """
            local_weights.append(copy.deepcopy(w))
            local_losses.append(copy.deepcopy(loss))


        # update global weights
        global_weights = average_weights(local_weights)

        # update global weights
        global_model.load_state_dict(global_weights)

        loss_avg = sum(local_losses) / len(local_losses)
        train_loss.append(loss_avg)

        # Calculate avg training accuracy over all users at every epoch
        list_acc, list_loss = [], []
        global_model.eval()
        for c in range(args.num_users):
            local_model = LocalUpdate(args=args, dataset=train_dataset,
                                      idxs=user_groups[idx], logger=logger)
            acc, loss = local_model.inference(model=global_model)
            list_acc.append(acc)
            list_loss.append(loss)
        train_accuracy.append(sum(list_acc)/len(list_acc))

        # print global training loss after every 'i' rounds
        if (epoch+1) % print_every == 0:
            print(f' \nAvg Training Stats after {epoch+1} global rounds:')
            print(f'Training Loss : {np.mean(np.array(train_loss))}')
            print('Train Accuracy: {:.2f}% \n'.format(100*train_accuracy[-1]))

    # Test inference after completion of training
    test_acc, test_loss = test_inference(args, global_model, test_dataset)

    print(f' \n Results after {args.epochs} global rounds of training:')
    print("|---- Avg Train Accuracy: {:.2f}%".format(100*train_accuracy[-1]))
    print("|---- Test Accuracy: {:.2f}%".format(100*test_acc))


    from datetime import datetime
    now = datetime.now()
    current_time_str = now.strftime("%Y-%m-%d %H%M%S")

    # Saving the objects train_loss and train_accuracy:
    file_name = './save/objects/[{}]_{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}].pkl'.\
        format(current_time_str, args.dataset, args.model, args.epochs, args.frac, args.iid,
               args.local_ep, args.local_bs)

    with open(file_name, 'wb') as f:
        pickle.dump([train_loss, train_accuracy], f)

    print('\n Total Run Time: {0:0.4f}'.format(time.time()-start_time))

    # PLOTTING (optional)
    import matplotlib
    import matplotlib.pyplot as plt
    matplotlib.use('Agg')

    # Plot Loss curve
    plt.figure()
    plt.title('Training Loss vs Communication rounds')
    plt.plot(range(len(train_loss)), train_loss, color='r')
    plt.ylabel('Training loss')
    plt.xlabel('Communication Rounds')
    plt.savefig('./save/[{}]_fed_{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}]_loss.png'.
                format(current_time_str, args.dataset, args.model, args.epochs, args.frac,
                       args.iid, args.local_ep, args.local_bs))
    
    # Plot Average Accuracy vs Communication rounds
    plt.figure()
    plt.title('Average Accuracy vs Communication rounds')
    plt.plot(range(len(train_accuracy)), train_accuracy, color='k')
    plt.ylabel('Average Accuracy')
    plt.xlabel('Communication Rounds')
    plt.savefig('./save/[{}]_fed_{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}]_acc.png'.
                format(current_time_str, args.dataset, args.model, args.epochs, args.frac,
                       args.iid, args.local_ep, args.local_bs))
