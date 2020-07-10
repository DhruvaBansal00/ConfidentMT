import numpy as np
import torch
from torch import nn
import random
import math
from transformerDef import ClassificationTransformer

def split_by_char(word):
    new_word = word.replace(",", " ").replace(";", " ").replace(".", " ")
    return new_word.split(" ")

def createDataset(translations, threshold):
    word_to_idx = {'CLS': 0, 'SEP': 1}
    curr_idx = 2

    data = []
    labels = []
    
    for translation in translations:
        ##remember to split by period, comma and semi-colon as well
        currData = [0]
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
        data.append(currData)
        labels.append(1 if translation.sbleu > threshold else 0)
    
    return data, labels

