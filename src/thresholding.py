from translation import Translation
import translationUtils
import dataUtils
from tqdm import tqdm

def splitByThreshold(translations, threshold, featureIndex):
    acceptedTranslations = []
    rejectedTranslations = []

    for translation in translations:
        if translation.features[featureIndex] > threshold:
            acceptedTranslations.append(translation)
        else:
            rejectedTranslations.append(translation)
    
    return acceptedTranslations, rejectedTranslations

def precisionCurveFromThresholding(translations, thresholds, featureIndex, FairseqWrapper):
    acceptedFractions = []
    acceptedScores = []

    for threshold in tqdm(thresholds):
        acceptedTranslations, rejectedTranslations = splitByThreshold(translations, threshold, featureIndex)
        _, inclusionScore = dataUtils.compute_excluded_included_score(acceptedTranslations, rejectedTranslations, FairseqWrapper)
        acceptedScores.append(float(inclusionScore))
        acceptedFractions.append(len(acceptedTranslations)/len(translations))
    
    acceptedScores = [x for _,x in sorted(zip(acceptedFractions,acceptedScores))]
    acceptedFractions.sort()

    return acceptedFractions, acceptedScores

