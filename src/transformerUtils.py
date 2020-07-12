import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import random
import math
import time
from transformerDef import ClassificationTransformer

def split_by_char(word):
    new_word = word.replace(",", " ").replace(";", " ").replace(".", " ")
    return new_word.split(" ")

def createDatasetHelper(translations, threshold, word_to_idx, curr_idx, maxLength):
    data = []
    labels = []
    
    for translation in translations:
        currData = []
        hypothesis_arr = translation.hypothesis.strip("\n").split(" ")
        for word in hypothesis_arr:
            separated_words = split_by_char(word)
            for curr_index, sep_word in enumerate(separated_words):
                if len(sep_word) > 0:
                    if sep_word not in word_to_idx:
                        word_to_idx[sep_word] = curr_idx
                        curr_idx += 1
                    currData.append(word_to_idx[sep_word])
                    if curr_index != (len(separated_words) - 1):
                        currData.append(word_to_idx['SEP'])
        if len(currData) < maxLength:
            currData.extend([0 for _ in range(len(currData), maxLength)])
        data.append(currData)
        labels.append(1 if translation.sbleu > threshold else 0)
    
    return data, labels, word_to_idx, curr_idx

def createDataset(trainTranslations, testTranslations, threshold, maxLength):
    word_to_idx = {'CLS': 0, 'SEP': 1}
    curr_idx = 2

    trainData, trainLabels, word_to_idx, curr_idx = createDatasetHelper(trainTranslations, threshold, word_to_idx, curr_idx, maxLength)
    testData, testLabels, word_to_idx, _ = createDatasetHelper(testTranslations, threshold, word_to_idx, curr_idx, maxLength)

    return trainData, trainLabels, testData, testLabels, word_to_idx

def trainTransformer(model, params, train_iter, train_labels, print_every=10):
    lr = params['lr']
    eps = params['eps']
    epochs = params['epochs']

    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    optim = torch.optim.Adam(model.parameters(), lr=lr, eps=eps)

    model.train()
    start = time.time()
    temp = start
    total_loss = 0

    for epoch in range(epochs):

        for i, batch in enumerate(train_iter):
            model_input = torch.LongTensor(batch)
            model_target = torch.LongTensor(train_labels[i])
            preds = model(model_input)
            optim.zero_grad()
            loss = F.cross_entropy(preds, model_target)
            loss.backward()
            optim.step()
            total_loss += loss.data
        
        if epoch % print_every == 0:
            loss_avg = total_loss / print_every
            print(f"time = {(time.time() - start) / 60}, epoch {epoch + 1}, loss = {loss_avg}, {time.time() - temp} seconds per {print_every} epochs")
            # print("time = %dm, epoch %d, loss = %.3f, %ds per %d epochs", (time.time() - start) / 60, epoch + 1, loss_avg, time.time() - temp, print_every)
            total_loss = 0
            temp = time.time()

    return model

def batchify(data, labels, batchSize):
    batched_data = [ data[i : min(i + batchSize, len(data))] for i in range(0, len(data), batchSize) ] 
    batched_labels = [ labels[i : min(i + batchSize, len(labels))] for i in range(0, len(labels), batchSize) ]
    return batched_data, batched_labels


def getClassifierTransformer(trainTranslations, testTranslations, threshold, params):

    hidden_dim = params['hidden_dim']
    num_heads = params['num_heads']
    dim_feedforward = params['feedforward_dim']
    dim_k = params['dim_k']
    dim_v = params['dim_v']
    dim_q = params['dim_q']
    max_length = params['max_length']
    batch_size = params['batch_size']
    print_every = params['verbosity']

    trainData, trainLabels, testData, testLabels, word_to_idx = createDataset(trainTranslations, testTranslations, threshold, max_length)
    model = ClassificationTransformer(word_to_idx, hidden_dim=hidden_dim, num_heads=num_heads, dim_feedforward=dim_feedforward, dim_k=dim_k, 
                                  dim_v=dim_v, dim_q=dim_q, max_length=max_length)
    train_iters, train_labels = batchify(trainData, trainLabels, batch_size)
    model = trainTransformer(model, params, train_iters, train_labels, print_every)
    return model

    