import pandas as pd
import torch
import torch.nn as nn
import numpy as np
import os
import time
import torch.optim as optim
import datetime
import math
import csv
from model import FFNN, CustomFFNN
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--rep-type', default='one_hot', type=str, choices=['one_hot', 'word_emb'])
parser.add_argument('--gpu-num', default=0, type=int)
parser.add_argument('--epochs', default=20, type=int)
parser.add_argument('--batch-size', default=64, type=int)
args = parser.parse_args()

data_folder = './dataset'
train_path = os.path.join(data_folder, 'simple_seq.train.csv')
test_path = os.path.join(data_folder, 'simple_seq.test.csv')
device = torch.device(f'cuda:{args.gpu_num}')
criterion = nn.CrossEntropyLoss()
global net


def load_data(data_path, train=True):
    unrefined_data = pd.read_csv(data_path, sep='\n', header=None)
    unrefined_data_values = unrefined_data.values
    if train == True:
        train_x = []
        train_y = []
        for t_val in unrefined_data_values:
            splited_value = list(filter(None, t_val[0].split(',')))
            train_x.append(splited_value[:-1])
            train_y.append(splited_value[-1])
        return train_x, train_y
    else:
        test_x = []
        for t_val in unrefined_data_values:
            splited_value = list(filter(None, t_val[0].split(',')))
            test_x.append(splited_value)
        return test_x


def output2csv(pred_y, class_dict, file_name='20214031_leesemi_simple_seq.answer.csv'):
    with open(file_name, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['id', 'pred'])
        for i, p in enumerate(pred_y):
            y_id = str(i + 1)
            if len(y_id) < 3:
                y_id = '0' * (3 - len(y_id)) + y_id
            writer.writerow(['S'+y_id, class_dict[p[0].item()]])
    print('file saved.')


def make_word_dict(data):
    word_dict = {}
    word_dict['[UNK]'] = 0
    word_dict['[PAD]'] = 1
    idx = 2
    for d in data:
        for t in d:
            if t not in word_dict.keys():
                word_dict[t] = idx
                idx += 1

    return word_dict


def make_label2vec(label):
    class_dict = {}
    idx = 0
    for l in label:
        if l not in class_dict:
            class_dict[l] = idx
            idx += 1
    label2vec = []
    for l in label:
        label2vec.append(torch.tensor(class_dict[l]))
    label2vec = torch.stack(label2vec, dim=0)
    return class_dict, label2vec


def one_hot_rep(data, word_dict):
    one_hot_seqs = []
    vocab_size = word_dict.__len__()
    for d in data:
        one_hot_seq = []
        for t in d:
            one_hot_vector = torch.zeros(vocab_size, requires_grad=False).to(device)
            if t not in word_dict.keys():
                one_hot_vector[word_dict['[UNK]']] = 1
            else:
                one_hot_vector[word_dict[t]] = 1
            one_hot_seq.append(one_hot_vector)
        one_hot_seq = torch.stack(one_hot_seq, dim=0)
        one_hot_seqs.append(one_hot_seq)
    one_hot_seqs = torch.stack(one_hot_seqs, dim=0)
    return one_hot_seqs


def word_emb(data, word_dict, word_emb_matrix):
    word2vec_seqs = []
    vocab_size = word_dict.__len__()
    for d in data:
        word2vec_seq = []
        for t in d:
            if t not in word_dict.keys():
                word2vec = word_emb_matrix[word_dict['[UNK]']]
            else:
                word2vec = word_emb_matrix[word_dict[t]]
            word2vec_seq.append(word2vec)
        word2vec_seq = torch.stack(word2vec_seq, dim=0)
        word2vec_seqs.append(word2vec_seq)
    word2vec_seqs = torch.stack(word2vec_seqs, dim=0)
    return word2vec_seqs


def pad_sequence(data, max_length):
    padded_data = []
    for d in data:
        if len(d) >= max_length:
            d = d[:max_length]
            padded_data.append(d)
        else:
            for i in range(len(d), 20):
                d.append('[PAD]')
            padded_data.append(d)
    return padded_data


def train(train_x, train_y, epoch):
    tr_loss = 0.
    correct = 0
    net.train()

    optimizer = optim.AdamW(net.parameters(), lr=0.01)
    batch_size = args.batch_size
    iteration = (len(train_x) // batch_size) + 1
    cur_i = 0
    for i in range(1, iteration + 1):
        if cur_i >= len(train_x):
            break
        if i < iteration:
            data, targets = train_x[cur_i:cur_i + i * batch_size].to(device), train_y[cur_i:cur_i + i * batch_size].to(device)
            cur_i += i * batch_size
        if i == iteration:
            data, targets = train_x[cur_i:].to(device), train_y[cur_i:].to(device)
            cur_i = len(train_x)
        optimizer.zero_grad()
        output = net(data)
        loss = criterion(output, targets)
        tr_loss += loss.item()
        loss.backward(retain_graph=True)
        optimizer.step()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(targets.view_as(pred)).sum().item()

        print("\r[epoch {:3d}/{:3d}] loss: {:.6f}".format(epoch, args.epochs, loss), end=' ')

    tr_loss /= iteration
    tr_acc = correct / len(train_x)
    return tr_loss, tr_acc


def test(test_x, targets=None):
    net.eval()
    correct = 0.
    ts_loss = 0.
    with torch.no_grad():
        data = test_x.to(device)
        output = net(data)

        pred = output.argmax(dim=1, keepdim=True)
        if targets != None:
            targets = targets.to(device)
            ts_loss = criterion(output, targets)
            correct += pred.eq(targets.view_as(pred)).sum().item()
            ts_acc = correct / len(test_x)
            return ts_loss, ts_acc
        else:
            return pred


if __name__ == '__main__':
    # data load
    train_x, train_y = load_data(train_path, train=True)
    test_x = load_data(test_path, train=False)

    # make word_dict
    word_dict = make_word_dict(train_x)
    vocab_size = word_dict.__len__()

    # padding
    train_x = pad_sequence(train_x, 20)
    test_x = pad_sequence(test_x, 20)

    # if) using one-hot embedding
    if args.rep_type == 'one_hot':
        train_x = one_hot_rep(train_x, word_dict)
        test_x = one_hot_rep(test_x, word_dict)
        dim = vocab_size

    # if) using word-embedding
    elif args.rep_type == 'word_emb':
        word_emb_matrix = nn.Parameter(torch.Tensor(vocab_size, 1024)).to(device)
        nn.init.xavier_normal_(word_emb_matrix)
        train_x = word_emb(train_x, word_dict, word_emb_matrix)
        test_x = word_emb(test_x, word_dict, word_emb_matrix)
        dim = 1024

    # make label_dict
    label2num, train_y = make_label2vec(train_y)
    num2label = {}
    for k, v in label2num.items():
        num2label[v] = k

    # train
    net = CustomFFNN(dim).to(device)
    for epoch in range(args.epochs):
        tr_loss, tr_acc = train(train_x, train_y, epoch)
        ts_loss, ts_acc = test(train_x, train_y)
        print("loss: {:.4f}, acc: {:.4f} ts_loss: {:.4f}, ts_acc: {:.4f}".format(tr_loss, tr_acc, ts_loss, ts_acc))

    # test & make kaggle file
    pred_y = test(test_x)
    output2csv(pred_y, num2label)