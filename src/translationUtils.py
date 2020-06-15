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

def addSentenceLengthFeatures(translations):
    for translation in translations:
        translation.transLength = len(translation.hypothesis)
        translation.sourceLength = len(translation.source)

def ngramCalculator(string, n):
    max_freq = 0
    ngram_dict = {}
    string_arr = string.split()
    length = n
    while n <= len(string_arr):
        curr_window = str(string_arr[n-length:n])
        if curr_window in ngram_dict:
            ngram_dict[curr_window] += 1
        else:
            ngram_dict[curr_window] = 1
        max_freq = max(max_freq, ngram_dict[curr_window])
        n += 1
    return max_freq

def addNgramFeatures(translations):
    for translation in translations:
        translation.unigram = ngramCalculator(translation.hypothesis, 1)
        translation.bigram = ngramCalculator(translation.hypothesis, 2)
        translation.trigram = ngramCalculator(translation.hypothesis, 3)


def addSentenceEndFeatures(translations):
    for translation in translations:
        translation.sentEndsTrans = translation.hypothesis.count(":") + translation.hypothesis.count(".")
        translation.sentEndsSource = translation.source.count("।")

def addLanguageModelFeatures(translations, FairseqWrapper, dataSet, lmModel):
    bpe = fastBPE.fastBPE(const.BPE_CODE)
    translation_text = [translation.hypothesis for translation in translations]
    bpe_text = bpe.apply(translation_text)

    bpe_translations = open(const.BPE_TRANSLATIONS, "w")
    bpe_translations.writelines(bpe_text)
    bpe_translations.close()

    FairseqWrapper.runFairseqPreprocess(const.BPE_DICTIONARY, dataSet+"pref", const.BPE_TRANSLATIONS, const.BPE_PREPROCESSED_TRNS)
    FairseqWrapper.runFairseqEvalLM(const.BPE_PREPROCESSED_TRNS, lmModel, 128, 1024, dataSet, const.TRANSLATION_LM_SCORE)

    translation_lm_scores = open(const.TRANSLATION_LM_SCORE, 'r')
    for translation in translation_lm_scores:
        index = translation.split(" ")[0]
        if index.isdigit():
            scores = translation.split("[")[1:]
            lmScore = mean([float(i.split("]")[0]) for i in scores])
            translations[int(index)].lmScore = lmScore
    
    translation_lm_scores.close()


def addBackwardModelFeatures(translations, FairseqWrapper):
    id_to_index = {}
    for index, translation in enumerate(translations):
        id_to_index[translation.trnID] = index

    bwModelResults = open(const.FAIRSEQ_GENERATE_FILE, 'r')
    for line in bwModelResults:
        if line.startswith("P-"):
            translation_id = float(line.split("\t")[0].split("-")[1])
            scores = [float(i) for i in line.split("\t")[1].split(" ")]
            translations[id_to_index[translation_id]].backwardAvgLP = mean(scores)

def longestRepeatedSubstring(str): 
    n = len(str) 
    LCSRe = [[0 for x in range(n + 1)]  for y in range(n + 1)] 
    res_length = 0
    index = 0

    for i in range(1, n + 1): 
        for j in range(i + 1, n + 1): 
            if (str[i - 1] == str[j - 1] and
                LCSRe[i - 1][j - 1] < (j - i)): 
                LCSRe[i][j] = LCSRe[i - 1][j - 1] + 1
                if (LCSRe[i][j] > res_length): 
                    res_length = LCSRe[i][j] 
                    index = max(i, index)                 
            else: 
                LCSRe[i][j] = 0
    return res_length

def addRepeatedStringFeatures(translations):
    for translation in translations:
        translation.repeatSource = longestRepeatedSubstring(translation.source)
        translation.repeatTrans = longestRepeatedSubstring(translation.hypothesis)

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
            if len(word) > 1:
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

def getTranslationFromDataset(dataSet, fwModel, bwModel, lmModel, sourceLang, targetLang, FairseqWrapper,
                             dataFolder="data-bin/wiki_ne_en_bpe5000/", produceGraphs=False, saveTranslations=False):

    FairseqWrapper.runFairseqGenerate(dataFolder, sourceLang, targetLang, fwModel, 5, 1.2, dataSet, "sentencepiece", const.FAIRSEQ_GENERATE_FILE)
    translations = initializeTranslations()
    FairseqWrapper.runFairseqScore(const.NMT_OUTPUT, const.NMT_GROUND_TRUTH, const.SENTENCE_BLEU, "sentence-bleu")
    addSentenceBleuStat(translations)
    addRareWordFeatures(translations, lambda x: x.hypothesis, lambda x, f: exec("x.rareTrans = f"))
    addRareWordFeatures(translations, lambda x: x.source, lambda x, f: exec("x.rareSource = f"))
    addRepeatedStringFeatures(translations)
    FairseqWrapper.runFairseqGenerate(dataFolder, targetLang, sourceLang, bwModel, 5, 1.2, dataSet, "sentencepiece", const.FAIRSEQ_GENERATE_FILE)
    addBackwardModelFeatures(translations, FairseqWrapper)
    addLanguageModelFeatures(translations, FairseqWrapper, dataSet, lmModel)
    addSentenceEndFeatures(translations)
    addNgramFeatures(translations)
    addSentenceLengthFeatures(translations)
    for translation in translations:
        translation.populateFeatures()
    if produceGraphs:
        print(len(translations))
        graphStatistics(translations, 15)
    if saveTranslations:
        saveData(translations, dataSet)
    return translations




