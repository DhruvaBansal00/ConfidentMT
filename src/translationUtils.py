import os
from statistics import mean, median
from translation import Translation
import const
import copy
import subprocess

def storeTranslationStatistics():

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
            NMT_original.write(hypothesis)
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
            currTranslation.avgLP, currTranslation.minLP, currTranslation.medianLP, currTranslation.maxLP, currTranslation.sumLP = mean(scores), min(scores), median(scores), max(scores), sum(scores)
            Sentence_stats.write(str(mean(scores))+" "+str(min(scores))+" "+str(median(scores))+" "+str(max(scores))+" "+str(sum(scores))+" "+str(translation_id)+"\n")
            translations.append(copy.deepcopy(currTranslation))
            currTranslation = Translation()

    NMT_ground_truth.close()
    NMT_output.close()
    Sentence_stats.close()
    NMT_original.close()
    return translations

def getTranslationFromDataset(dataSet, fwModel, bwModel, lmModel, sourceLang, targetLang, runFairSeqGenerate, dataFolder="data-bin/wiki_ne_en_bpe5000/"):

    runFairSeqGenerate(dataFolder, sourceLang, targetLang, fwModel, 5, 1.2, dataSet, "sentencepiece", const.FAIRSEQ_GENERATE_FILE)
    translations = storeTranslationStatistics()
    return translations



