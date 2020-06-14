import numpy as np
from itertools import zip_longest
import matplotlib.pyplot as plt
import random
from sklearn.metrics import auc
import statistics
import os
from translation import Translation

def compute_exclued_included_sentenceBleuScore(acceptedTranslations, rejectedTranslations):
    acceptedScore = 0 if len(acceptedTranslations) == 0 else sum([translation.score for translation in acceptedTranslations])/len(acceptedTranslations)
    
    rejectedScore = 0 if len(rejectedTranslations) == 0 else sum([translation.score for translation in rejectedTranslations])/len(rejectedTranslations)

    return rejectedScore, acceptedScore

def compute_excluded_included_score (acceptedTranslations, rejectedTranslations):
    if len(acceptedTranslations) != 0:
        temporary_reference_inclusion = open("analysis/temporary_reference_inclusion.data", "w")
        temporary_output_inclusion = open("analysis/temporary_output_inclusion.data", "w")

    
        for translation in acceptedTranslations:
            temporary_reference_inclusion.write(translation.reference)
            temporary_output_inclusion.write(translation.translation)

        temporary_reference_inclusion.close()
        temporary_output_inclusion.close()


        fairseq_command = (f'fairseq-score --sys analysis/temporary_output_inclusion.data'
                           f'--ref analysis/temporary_reference_inclusion.data'
                           f' --sacrebleu > analysis/inclusion_result.data')
        os.system(fairseq_command)

        temporary_inclusion_result = open("analysis/inclusion_result.data")
        inclusion_result_string = [line for line in temporary_inclusion_result][1].split(" ")[2]

        temporary_reference_inclusion.close()
        temporary_output_inclusion.close()
        temporary_inclusion_result.close()

    else:
        inclusion_result_string = "0"

    if len(rejectedTranslations) != 0:

        temporary_reference_exclusion = open("analysis/temporary_reference_exclusion.data", "w")
        temporary_output_exclusion = open("analysis/temporary_output_exclusion.data", "w")
        
        for translation in rejectedTranslations:
            temporary_reference_exclusion.write(translation.reference)
            temporary_output_exclusion.write(translation.translation)

        
        temporary_reference_exclusion.close()
        temporary_output_exclusion.close()

        fairseq_command  = (f'fairseq-score --sys analysis/temporary_output_exclusion.data'
                            f'--ref analysis/temporary_reference_exclusion.data'
                            f' --sacrebleu > analysis/exclusion_result.data')
        os.system(fairseq_command)

        temporary_exclusion_result = open("analysis/exclusion_result.data")
        exclusion_result_string = "0" if len(rejectedTranslations) == 0 else [line for line in temporary_exclusion_result][1].split(" ")[2]

        temporary_reference_exclusion.close()
        temporary_output_exclusion.close()
        temporary_exclusion_result.close()
    
    else:
        exclusion_result_string = "0"

    return float(exclusion_result_string), float(inclusion_result_string)

def readTranslations(sentenceFile, featureArray):
    translations = []
    temp = []
    index = 0
    for line in sentenceFile:
        if len(temp) < 3:
            temp.append(line)
        else:
            score = float(line.strip("\n"))
            translations.append(Translation(temp[0], temp[1], temp[2], score, featureArray[index]))
            index += 1
            temp = []
    
    return translations

def getTrainTestSets(trainTranslations, testTranslations, threshold_train, threshold_test, avgLogProb):
    trainFeatures = []
    trainY = []
    testFeatures = []
    testY = []

    for translation in trainTranslations:
        trainFeatures.append(translation.features)
        if avgLogProb:
            if translation.features[0] < threshold_train:
                trainY.append(0)
            else:
                trainY.append(1)
        else:
            if translation.score < threshold_train:
                trainY.append(0)
            else:
                trainY.append(1)
    
    for translation in testTranslations:
        testFeatures.append(translation.features)
        if translation.score < threshold_test:
            testY.append(0)
        else:
            testY.append(1)

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

def datasetReader(featureFile, labelFile):
    files = [featureFile, labelFile]

    X = []
    Y = []

    for lines in zip_longest(*files, fillvalue=''):
        currX, currY = lines[0], float(lines[1].strip("\n"))
        Xarr = []
        features = currX.split()
        for feature in features:
            Xarr.append(float(feature.strip(",").strip("\n")))
        X.append(Xarr)
        Y.append(currY)
    
    return np.array(X), np.array(Y)

def normalizeFeatures(trainX, testX):
    featureLists = [trainX[:, i] for i in range(trainX.shape[1])]

    #means and stddv calculated using training features only. 
    means = [statistics.mean(feature) for feature in featureLists]
    stdDv = [statistics.stdev(feature) for feature in featureLists]

    trainX = np.array([[(row[i] - means[i]) / stdDv[i] for i in range(len(row))] for row in trainX])
    testX = np.array([[(row[i] - means[i]) / stdDv[i] for i in range(len(row))] for row in testX])

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