import os
from statistics import mean, median
from translation import Translation
import const
import copy
import subprocess
from itertools import zip_longest
import matplotlib.pyplot as plt
from sklearn.metrics import auc
from numpy import arange
import random
import numpy as np
import string

def addRareWordFeatures(translations, sentence, feature):
    dict_to_freq = {}
    for translation in translations:
        sentence_words = sentence(translation).split(" ")
        for word in sentence_words:
            if len(word) > 1:
                word = word.translate(str.maketrans({a:None for a in string.punctuation}))

                if word in dict_to_freq:
                    dict_to_freq[word] = 1 + dict_to_freq[word]
                else: 
                    dict_to_freq[word] = 1
    
    for translation in translations:
        sentence_words = sentence(translation).split(" ")
        rareWordNum = 0
        for word in sentence_words:
            currWord = word.translate(str.maketrans({a:None for a in string.punctuation}))
            rareWordNum += 1 if dict_to_freq[currWord] <= const.RARE_THREHOLD else 0
        feature(translation, rareWordNum)


def addSentenceBleuStat(translations):
    sentenceBleuStat = open(const.SENTENCE_BLEU, "r")
    next(sentenceBleuStat)
    for index, currLine in enumerate(sentenceBleuStat):
        currBleu = float(currLine.split("BLEU4 = ")[1].split(",")[0])
        translations[index].sbleu = currBleu

def initializeTranslations():

    bleu_res = open(const.FAIRSEQ_GENERATE_FILE, "r")
    NMT_original = open(const.NMT_ORIGINAL, "w")
    NMT_ground_truth = open(const.NMT_GROUND_TRUTH, "w")
    NMT_output = open(const.NMT_OUTPUT, "w")
    Sentence_stats = open(const.SENTENCE_STATS, "w")
    
    translations = []
    translation_id = None
    currTranslation = Translation()
    for line in bleu_res:
        if line.startswith("H-"):
            hypothesis = line.split("\t")[2]
            currTranslation.hypothesis = hypothesis
            NMT_output.write(hypothesis)
        elif line.startswith("T-"):
            reference = line.split("\t")[1]
            currTranslation.reference = reference
            NMT_ground_truth.write(reference)
        elif line.startswith("S-"):
            translation_id = float(line.split("\t")[0].split("-")[1])
            source = line.split("\t")[1]
            currTranslation.source = source
            currTranslation.trnID = translation_id
            NMT_original.write(source)            
        elif line.startswith("P-"):
            scores = [float(i) for i in line.split("\t")[1].split(" ")]
            currTranslation.avgLP, currTranslation.minLP, currTranslation.medianLP, currTranslation.maxLP, currTranslation.sumLP = [mean(scores), min(scores), median(scores), max(scores), sum(scores)]
            Sentence_stats.write(str(mean(scores))+" "+str(min(scores))+" "+str(median(scores))+" "+str(max(scores))+" "+str(sum(scores))+" "+str(translation_id)+"\n")
            translations.append(copy.deepcopy(currTranslation))
            currTranslation = Translation()

    NMT_ground_truth.close()
    NMT_output.close()
    Sentence_stats.close()
    NMT_original.close()
    return translations

def getTranslationFromDataset(dataSet, fwModel, bwModel, lmModel, sourceLang, targetLang, FairseqWrapper, dataFolder="data-bin/wiki_ne_en_bpe5000/"):

    FairseqWrapper.runFairseqGenerate(dataFolder, sourceLang, targetLang, fwModel, 5, 1.2, dataSet, "sentencepiece", const.FAIRSEQ_GENERATE_FILE)
    translations = initializeTranslations()
    FairseqWrapper.runFairseqScore(const.NMT_OUTPUT, const.NMT_GROUND_TRUTH, const.SENTENCE_BLEU)
    addSentenceBleuStat(translations)
    addRareWordFeatures(translations, lambda x: x.translation, lambda x, f: exec("x.rareTrans = f"))
    addRareWordFeatures(translations, lambda x: x.source, lambda x, f: exec("x.rareSource = f"))
    return translations



