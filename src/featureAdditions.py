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
from spm_encode import encode

def parseGenerationResult():
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
            currTranslation.trnID = int(translation_id)
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

def addSentenceBleuStat(translations, FairseqWrapper):
    FairseqWrapper.runFairseqScore(const.NMT_OUTPUT, const.NMT_GROUND_TRUTH, const.SENTENCE_BLEU, "sentence-bleu")
    sentenceBleuStat = open(const.SENTENCE_BLEU, "r")
    next(sentenceBleuStat)
    for index, currLine in enumerate(sentenceBleuStat):
        currBleu = float(currLine.split("BLEU4 = ")[1].split(",")[0])
        translations[index].sbleu = currBleu

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

    FairseqWrapper.runFairseqPreprocessLM(const.BPE_DICTIONARY, dataSet+"pref", const.BPE_TRANSLATIONS, const.BPE_PREPROCESSED_TRNS)
    FairseqWrapper.runFairseqEvalLM(const.BPE_PREPROCESSED_TRNS, lmModel, 128, 1024, dataSet, const.TRANSLATION_LM_SCORE)

    translation_lm_scores = open(const.TRANSLATION_LM_SCORE, 'r')
    for translation in translation_lm_scores:
        index = translation.split(" ")[0]
        if index.isdigit():
            scores = translation.split("[")[1:]
            lmScore = mean([float(i.split("]")[0]) for i in scores])
            translations[int(index)].lmScore = lmScore
    
    translation_lm_scores.close()


def addBackwardModelFeatures(translations, FairseqWrapper, dataFolder, targetLang, sourceLang, bwModel, dataSet):
    FairseqWrapper.runFairseqGenerate(dataFolder, targetLang, sourceLang, bwModel, 5, 1.2, dataSet, "sentencepiece", const.FAIRSEQ_GENERATE_FILE)
    
    id_to_index = {}
    for index, translation in enumerate(translations):
        id_to_index[translation.trnID] = index

    backwardTranslations = parseGenerationResult()
    addSentenceBleuStat(backwardTranslations, FairseqWrapper)


    english_translation_file = open(const.BACKWARD_DATASET+dataSet+"."+targetLang, 'w')
    nepali_original_file = open(const.BACKWARD_DATASET+dataSet+"."+sourceLang, 'w')

    for btrans in backwardTranslations:
        translations[id_to_index[btrans.trnID]].backwardRefAvgLP, translations[id_to_index[btrans.trnID]].backwardRefSBleu = btrans.avgLP, btrans.sbleu

    for translation in translations:
        english_translation_file.write(translation.hypothesis)
        nepali_original_file.write(translation.source) 
    
    english_translation_file.close()
    nepali_original_file.close() 
    
    encode(const.SENTENCEPIECE_MODEL_DIR+sourceLang+targetLang+".bpe.model", inputs =[const.BACKWARD_DATASET+dataSet+"."+targetLang, const.BACKWARD_DATASET+dataSet+"."+sourceLang],
            outputs =[const.BACKWARD_DATASET+dataSet+".bpe."+targetLang, const.BACKWARD_DATASET+dataSet+".bpe."+sourceLang], output_format="piece")
    FairseqWrapper.runFairseqPreprocessBinarize(targetLang, sourceLang, const.SENTENCEPIECE_MODEL_DIR+sourceLang+targetLang+"Dict.data", dataSet+"pref", const.BACKWARD_DATASET+dataSet+".bpe", const.BACKWARD_DATASET)
    FairseqWrapper.runFairseqGenerate(const.BACKWARD_DATASET, targetLang, sourceLang, bwModel, 5, 1.2, dataSet, "sentencepiece", const.FAIRSEQ_GENERATE_FILE)
    
    backwardTranslations = parseGenerationResult()
    addSentenceBleuStat(backwardTranslations, FairseqWrapper)
    for btrans in backwardTranslations:
        translations[btrans.trnID].backwardsAvgLP, translations[btrans.trnID].backwardSBleu = btrans.avgLP, btrans.sbleu

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