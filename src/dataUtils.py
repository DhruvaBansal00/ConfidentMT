import numpy as np
from itertools import zip_longest
import matplotlib.pyplot as plt
import random
from sklearn.metrics import auc
import statistics
import os
from translation import Translation
import const
import copy

def createObjectsFromFile(dataSet):
    sentenceFile = open(const.CLASSIFICATION_DATASET+"/"+dataSet+"/"+const.CLASSIFICATION_SENTENCES, "r")
    featureFile = open(const.CLASSIFICATION_DATASET+"/"+dataSet+"/"+const.CLASSIFICATION_FEATURES, "r")
    translations = []
    sentenceData = sentenceFile.readlines()
    featureData = featureFile.readlines()
    currIndex = 0
    while currIndex < len(sentenceData):
        currTranslation = Translation()
        currTranslation.source, currTranslation.reference, currTranslation.hypothesis, currTranslation.trnID = [sentenceData[currIndex], sentenceData[currIndex + 1], sentenceData[currIndex + 2], float(sentenceData[currIndex + 3])]
        currTranslation.loadProperties([float(i.strip('\n')) for i in featureData[int(currIndex/4)].split(" ")])
        currIndex += 4
        translations.append(copy.deepcopy(currTranslation))
    
    return translations

def compute_exclued_included_sentenceBleuScore(acceptedTranslations, rejectedTranslations):
    acceptedScore = 0 if len(acceptedTranslations) == 0 else sum([translation.sbleu for translation in acceptedTranslations])/len(acceptedTranslations)
    
    rejectedScore = 0 if len(rejectedTranslations) == 0 else sum([translation.sbleu for translation in rejectedTranslations])/len(rejectedTranslations)

    return rejectedScore, acceptedScore

def compute_excluded_included_score(acceptedTranslations, rejectedTranslations, FairseqWrapper):
    if len(acceptedTranslations) != 0:
        temporary_reference_inclusion = open(const.INCLUSION_REFERENCE, "w")
        temporary_output_inclusion = open(const.INCLUSION_OUTPUT, "w")
        for translation in acceptedTranslations:
            temporary_reference_inclusion.write(translation.reference)
            temporary_output_inclusion.write(translation.hypothesis)
        temporary_reference_inclusion.close()
        temporary_output_inclusion.close()

        FairseqWrapper.runFairseqScore(const.INCLUSION_OUTPUT, const.INCLUSION_REFERENCE, const.INCLUSION_RESULT, "sacrebleu")
        temporary_inclusion_result = open(const.INCLUSION_RESULT, 'r')
        inclusion_result_string = [line for line in temporary_inclusion_result][1].split(" ")[2]
        temporary_reference_inclusion.close()
        temporary_output_inclusion.close()
        temporary_inclusion_result.close()
    else:
        inclusion_result_string = "0"

    if len(rejectedTranslations) != 0:
        temporary_reference_exclusion = open(const.EXCLUSION_REFERENCE, "w")
        temporary_output_exclusion = open(const.EXCLUSION_OUTPUT, "w")
        for translation in rejectedTranslations:
            temporary_reference_exclusion.write(translation.reference)
            temporary_output_exclusion.write(translation.hypothesis)
        temporary_reference_exclusion.close()
        temporary_output_exclusion.close()

        FairseqWrapper.runFairseqScore(const.EXCLUSION_OUTPUT, const.EXCLUSION_REFERENCE, const.EXCLUSION_RESULT, "sacrebleu")
        temporary_exclusion_result = open(const.EXCLUSION_RESULT, 'r')
        exclusion_result_string = "0" if len(rejectedTranslations) == 0 else [line for line in temporary_exclusion_result][1].split(" ")[2]
        temporary_reference_exclusion.close()
        temporary_output_exclusion.close()
        temporary_exclusion_result.close() 
    else:
        exclusion_result_string = "0"

    return float(exclusion_result_string), float(inclusion_result_string)


def getClassifierTrainTestSets(trainTranslations, testTranslations, threshold, featureIndices):
    trainFeatures = []
    trainY = []
    testFeatures = []
    testY = []

    for translation in trainTranslations:
        trainFeatures.append([translation.features[i] for i in featureIndices])
        if translation.sbleu < threshold:
            trainY.append(0)
        else:
            trainY.append(1)
    
    for translation in testTranslations:
        testFeatures.append([translation.features[i] for i in featureIndices])
        if translation.sbleu < threshold:
            testY.append(0)
        else:
            testY.append(1)

    return trainFeatures, trainY, testFeatures, testY


def getRegressionTrainTestSets(trainTranslations, testTranslations, featureIndices):
    trainFeatures = []
    trainY = []
    testFeatures = []
    testY = []
    
    for translation in trainTranslations:
        trainFeatures.append([translation.features[i] for i in featureIndices])
        trainY.append(translation.sbleu)
    
    for translation in testTranslations:
        testFeatures.append([translation.features[i] for i in featureIndices])
        testY.append(translation.sbleu)
    
    return trainFeatures, trainY, testFeatures, testY


def printDatasetClassProp(Y): 
    classes = {}
    total = len(Y)
    for i in Y:
        if i in classes:
            classes[i] += 1
        else:
            classes[i] = 1
    
    for cls in classes:
        print("Proportion in class " + str(cls) + " = " + str(classes[cls]/total))


def normalizeFeatures(trainX, testX, features):
    means = {}
    stdDv = {}
    for feature in features:
        currData = trainX[:][feature]
        means[feature] = statistics.mean(currData)
        stdDv[feature] = statistics.stdev(currData)

    sets = [trainX, testX]
    for dataSet in sets: 
        for currData in dataSet:
            for feature in features:
                currData[feature] = (currData[feature] - means[feature])/stdDv[feature]

    return trainX, testX


def computeSimilarity(o1, o2):
    total = len(o1)
    same = 0
    for i in range(len(o1)):
        if o1[i] == o2[i]:
            same += 1
    print(same/total)

def calculateAccuracy(predictedClasses, groundTruth):
    correct_accepted = 0
    total_accepted = 0

    correct_rejected = 0
    total_rejected = 0

    for i in range(len(predictedClasses)):
        if groundTruth[i] == 1:
            total_accepted += 1
            if predictedClasses[i] == groundTruth[i]:
                correct_accepted += 1
        else:
            total_rejected += 1
            if predictedClasses[i] == groundTruth[i]:
                correct_rejected += 1


    print("Correctly accepted = " + str(correct_accepted/total_accepted))
    print("Incorrectly rejected = " + str(1 - correct_accepted/total_accepted))
    print("Correctly rejected = " + str(correct_rejected/total_rejected))
    print("Incorrectly accepted = " + str(1 - correct_rejected/total_rejected))

    print("Total Accuracy = " + str((correct_accepted + correct_rejected)/(total_accepted + total_rejected)))

def calculatedAcceptedFraction(predictedClasses):
    return len([i for i in predictedClasses if i > 0])/len(predictedClasses)