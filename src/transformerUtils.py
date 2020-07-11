import numpy as np
import torch
from torch import nn
import random
import math
from transformerDef import ClassificationTransformer

def split_by_char(word):
    new_word = word.replace(",", " ").replace(";", " ").replace(".", " ")
    return new_word.split(" ")

def createDatasetHelper(translations, threshold, word_to_idx, curr_idx, maxLength):
    data = []
    labels = []
    
    for translation in translations:
        ##remember to split by period, comma and semi-colon as well
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

def createDataset(trainTranslations, testTranslations, threshold, maxLength=100):
    word_to_idx = {'CLS': 0, 'SEP': 1}
    curr_idx = 2

    trainData, trainLabels, word_to_idx, curr_idx = createDatasetHelper(trainTranslations, threshold, word_to_idx, curr_idx, maxLength)
    testData, testLabels, word_to_idx, _ = createDatasetHelper(testTranslations, threshold, word_to_idx, curr_idx, maxLength)

    return np.array(trainData), np.array(trainLabels), np.array(testData), np.array(testLabels), word_to_idx

def trainTransformer(model, params, train_iter):
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
            src = batch.English.transpose(0,1)
            trg = batch.French.transpose(0,1)
            # the French sentence we input has all words except
            # the last, as it is using each word to predict the next
            
            trg_input = trg[:, :-1]
            
            # the words we are trying to predict
            
            targets = trg[:, 1:].contiguous().view(-1)
            
            # create function to make masks using mask code above
            
            src_mask, trg_mask = create_masks(src, trg_input)
            
            preds = model(src, trg_input, src_mask, trg_mask)
            
            optim.zero_grad()
            
            loss = F.cross_entropy(preds.view(-1, preds.size(-1)),
            results, ignore_index=target_pad)
            loss.backward()
            optim.step()
            
            total_loss += loss.data[0]
            if (i + 1) % print_every == 0:
                loss_avg = total_loss / print_every
                print("time = %dm, epoch %d, iter = %d, loss = %.3f,
                %ds per %d iters" % ((time.time() - start) // 60,
                epoch + 1, i + 1, loss_avg, time.time() - temp,
                print_every))
                total_loss = 0
                temp = time.time()

    return model

def batchify(data, batchSize):
    batched_data = [ data[i : min(i + batchSize, len(data))] for i in range(0, len(data), batchSize) ] 
    return np.array(batched_data) 


def getClassifierTransformer(trainTranslations, testTranslations, threshold, params):

    hidden_dim = params['hidden_dim']
    num_heads = params['num_heads']
    dim_feedforward = params['feedforward_dim']
    dim_k = params['dim_k']
    dim_v = params['dim_v']
    dim_q = params['dim_q']
    max_length = params['max_length']
    batch_size = params['batch_size']

    trainData, trainLabels, testData, testLabels, word_to_idx = createDataset(trainTranslations, testTranslations, threshold, max_length)
    model = ClassificationTransformer(word_to_idx, hidden_dim=hidden_dim, num_heads=num_heads, dim_feedforward=dim_feedforward, dim_k=dim_k, 
                                  dim_v=dim_v, dim_q=dim_q, max_length=max_length)
    train_iters = batchify(trainData, batch_size)
    model = trainTransformer(model, params, train_iters)

    