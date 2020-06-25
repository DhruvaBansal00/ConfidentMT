import regressorTrainers
import dataUtils
from tqdm.notebook import tqdm
import numpy as np

def precisionCurveFromRegressor(trainTranslations, testTranslations, regressor, FairseqWrapper, thresholds, featureIndices, normalizeFeatures=[]):
    acceptedFractions = []
    acceptedScores = []

    trainX, trainY, testX, testY = dataUtils.getRegressionTrainTestSets(trainTranslations, testTranslations, featureIndices)
    trainX, testX = dataUtils.normalizeFeatures(trainX, testX, normalizeFeatures)
    currRegressor = regressorTrainers.getTrainerFromRegressor(regressor)(trainX, trainY)
    testPredictions = currRegressor.predict(testX)

    for threshold in thresholds:

        acceptedTranslations = np.array(testTranslations)[np.array(testPredictions) >= threshold]
        rejectedTranslations = np.array(testTranslations)[np.array(testPredictions) < threshold]
        _, acceptedScore = dataUtils.compute_excluded_included_score(acceptedTranslations, rejectedTranslations, FairseqWrapper)
        acceptedScores.append(acceptedScore)
        acceptedFractions.append(float(len(acceptedTranslations))/float(len(testPredictions)))
    
    acceptedScores = [x for _,x in sorted(zip(acceptedFractions,acceptedScores))]
    acceptedFractions.sort()

    return acceptedFractions, acceptedScores

def verboseTraining(trainTranslations, testTranslations, regressor, FairseqWrapper, threshold, featureIndices, normalizeFeatures=[]):
    
    print("#################################################")
    trainX, trainY, testX, testY = dataUtils.getRegressionTrainTestSets(trainTranslations, testTranslations, featureIndices)
    trainX, testX = dataUtils.normalizeFeatures(trainX, testX, normalizeFeatures)
    currRegressor = regressorTrainers.getTrainerFromRegressor(regressor)(trainX, trainY, verbose=True)

    print("Train Accuracy")
    predictions = [int(i > threshold) for i in currRegressor.predict(trainX)]
    dataUtils.calculateAccuracy(predictions, trainY)
    print("Percent acepted = " + str(100 * dataUtils.calculatedAcceptedFraction(predictions)))
    acceptedTranslations = np.array(trainTranslations)[np.array(predictions) > 0]
    rejectedTranslations = np.array(trainTranslations)[np.array(predictions) < 1]
    _, acceptedScore = dataUtils.compute_excluded_included_score(acceptedTranslations, rejectedTranslations, FairseqWrapper)
    print("Corpus BLEU score of accepted translations = " + str(acceptedScore))

    print("Test Accuracy")
    predictions = [int(i > threshold) for i in currRegressor.predict(trainX)]
    dataUtils.calculateAccuracy(predictions, testY)
    print("Percent acepted = " + str(100 * dataUtils.calculatedAcceptedFraction(predictions)))
    acceptedTranslations = np.array(testTranslations)[np.array(predictions) > 0]
    rejectedTranslations = np.array(testTranslations)[np.array(predictions) < 1]
    _, acceptedScore = dataUtils.compute_excluded_included_score(acceptedTranslations, rejectedTranslations, FairseqWrapper)
    print("Corpus BLEU score of accepted translations = " + str(acceptedScore))
    print("#################################################")
