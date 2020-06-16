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
import fastBPE
import featureAdditions

def saveData(translations, dataSet):
    sentenceFile = open(const.CLASSIFICATION_DATASET+"/"+dataSet+"/"+const.CLASSIFICATION_SENTENCES, "w")
    featureFile = open(const.CLASSIFICATION_DATASET+"/"+dataSet+"/"+const.CLASSIFICATION_FEATURES, "w")

    for translation in translations:
        sentenceFile.write(translation.source)
        sentenceFile.write(translation.reference)
        sentenceFile.write(translation.hypothesis)
        sentenceFile.write(str(translation.trnID)+"\n")

        featureFile.write(" ".join([str(i) for i in translation.getProperties()])+"\n")
    
    sentenceFile.close()
    featureFile.close()

def graphStatistics(translations, acceptThreshold):
    index_to_label = {0: "Average Logprob", 1: "Min Logprob", 2: "Median Logprob", 
                  3: "Max Logprob", 5: "Number of Rare words in Source", 
                  6: "Number of Rare words in Translation", 7: "Rare in source - rare in translation", 
                  8: "Longest repeated substring length english", 9: "Longest repeated substring length nepali",
                  10: "Token Logprob sum", 11: "Backward Model score", 12: "Language Model score", 
                  13: "Number of . and : in Translation", 14:"Number of | in Source",
                  15:"| in Source - . and : in Translation", 16: "Max unigram count in translation", 
                  17: "Max bigram count in translation", 18: "Max trigram count in translation", 
                  19: "Translation Length", 20: "Source length", 21: "Ratio between source and translation length"}
    indices = [0, 1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]

    for i in indices:
        accepted_set_stats = [[],[]]
        rejected_set_stats = [[],[]]
        for translation in translations:  
            if translation.sbleu > acceptThreshold:
                accepted_set_stats[0].append(translation.features[i])
                accepted_set_stats[1].append(translation.sbleu)
            else:
                rejected_set_stats[0].append(translation.features[i])
                rejected_set_stats[1].append(translation.sbleu)

        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)

        ax.scatter(accepted_set_stats[0], accepted_set_stats[1], alpha=0.8, c="blue", edgecolors='none', s=30, label="Accepted")
        ax.scatter(rejected_set_stats[0], rejected_set_stats[1], alpha=0.8, c="red", edgecolors='none', s=30, label="Rejected")

        plt.xlabel(index_to_label[i]) 
        plt.ylabel('Sentence BLEU') 
        plt.title(index_to_label[i] + ' vs Sentence BLEU')
        plt.legend(loc=1)
        plt.show()

def initializeTranslations(dataFolder, sourceLang, targetLang, fwModel, dataSet, FairseqWrapper):
    FairseqWrapper.runFairseqGenerate(dataFolder, sourceLang, targetLang, fwModel, 5, 1.2, dataSet, "sentencepiece", const.FAIRSEQ_GENERATE_FILE)
    translations = featureAdditions.parseGenerationResult()
    return translations

def getTranslationFromDataset(dataSet, fwModel, bwModel, lmModel, sourceLang, targetLang, FairseqWrapper,
                             dataFolder="data-bin/wiki_ne_en_bpe5000/", produceGraphs=False, saveTranslations=False):

    translations = initializeTranslations(dataFolder, sourceLang, targetLang, fwModel, dataSet, FairseqWrapper)
    featureAdditions.addSentenceBleuStat(translations, FairseqWrapper)
    featureAdditions.addRareWordFeatures(translations, lambda x: x.hypothesis, lambda x, f: exec("x.rareTrans = f"))
    featureAdditions.addRareWordFeatures(translations, lambda x: x.source, lambda x, f: exec("x.rareSource = f"))
    featureAdditions.addRepeatedStringFeatures(translations)
    featureAdditions.addBackwardModelFeatures(translations, FairseqWrapper, dataFolder, targetLang, sourceLang, bwModel, dataSet)
    featureAdditions.addLanguageModelFeatures(translations, FairseqWrapper, dataSet, lmModel)
    featureAdditions.addSentenceEndFeatures(translations)
    featureAdditions.addNgramFeatures(translations)
    featureAdditions.addSentenceLengthFeatures(translations)
    for translation in translations:
        translation.populateFeatures()
    if produceGraphs:
        print(len(translations))
        graphStatistics(translations, 15)
    if saveTranslations:
        saveData(translations, dataSet)
    return translations




