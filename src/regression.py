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