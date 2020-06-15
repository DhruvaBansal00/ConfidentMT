import classifierTrainers
import dataUtils
from tqdm import tqdm
import numpy as np

def precisionCurveFromClassification(trainTranslations, testTranslations, classifier, FairseqWrapper, thresholds, featureIndices):
    acceptedFractions = []
    acceptedScores = []

    for threshold in tqdm(thresholds):
        trainX, trainY, testX, testY = dataUtils.getTrainTestSets(trainTranslations, testTranslations, threshold, featureIndices)
        currClassifier = classifierTrainers.getTrainerFromClassifier(classifier)(trainX, trainY)
        predictions = currClassifier.predict(testX)

        acceptedTranslations = np.array(testTranslations)[np.array(predictions) > 0]
        rejectedTranslations = np.array(testTranslations)[np.array(predictions) < 1]
        _, acceptedScore = dataUtils.compute_excluded_included_score(acceptedTranslations, rejectedTranslations, FairseqWrapper)
        acceptedScores.append(acceptedScore)
        acceptedFractions.append(float(len(acceptedTranslations))/float(len(predictions)))
    
    acceptedScores = [x for _,x in sorted(zip(acceptedFractions,acceptedScores))]
    acceptedFractions.sort()

    return acceptedFractions, acceptedScores

def verboseTraining(trainTranslations, testTranslations, classifier, FairseqWrapper, threshold, featureIndices):
    
    print("#################################################")
    trainX, trainY, testX, testY = dataUtils.getTrainTestSets(trainTranslations, testTranslations, threshold, featureIndices)
    currClassifier = classifierTrainers.getTrainerFromClassifier(classifier)(trainX, trainY, verbose=True)

    print("Train Accuracy")
    predictions = currClassifier.predict(trainX)
    dataUtils.calculateAccuracy(predictions, trainY)
    print("Percent acepted = " + str(100 * dataUtils.calculatedAcceptedFraction(predictions)))
    acceptedTranslations = np.array(trainTranslations)[np.array(predictions) > 0]
    rejectedTranslations = np.array(trainTranslations)[np.array(predictions) < 1]
    _, acceptedScore = dataUtils.compute_excluded_included_score(acceptedTranslations, rejectedTranslations, FairseqWrapper)
    print("Corpus BLEU score of accepted translations = " + str(acceptedScore))

    print("Test Accuracy")
    predictions = currClassifier.predict(testX)
    dataUtils.calculateAccuracy(predictions, testY)
    print("Percent acepted = " + str(100 * dataUtils.calculatedAcceptedFraction(predictions)))
    acceptedTranslations = np.array(testTranslations)[np.array(predictions) > 0]
    rejectedTranslations = np.array(testTranslations)[np.array(predictions) < 1]
    _, acceptedScore = dataUtils.compute_excluded_included_score(acceptedTranslations, rejectedTranslations, FairseqWrapper)
    print("Corpus BLEU score of accepted translations = " + str(acceptedScore))
    print("#################################################")






