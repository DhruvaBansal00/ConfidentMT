NUM_FEATURES = 22
FAIRSEQ_GENERATE_FILE = "analysis/fairseqGenerate.data"
NMT_ORIGINAL = "analysis/original_sentences.data"
NMT_GROUND_TRUTH = "analysis/ground_truth_sentences.data"
NMT_OUTPUT = "analysis/translated_sentences.data"
SENTENCE_STATS = "analysis/logprob_statistics.data"
SENTENCE_BLEU = "analysis/scores.data"
RARE_THREHOLD = 2
BPE_CODE = "language_models/lm/bpe32k.code"
BPE_TRANSLATIONS = "analysis/bpe_translations.data"
BPE_DICTIONARY = "language_models/lm/dict.data"
BPE_PREPROCESSED_TRNS = "analysis/preprocessed_translations"
TRANSLATION_LM_SCORE = "analysis/translation_lm_score.data"
CLASSIFICATION_DATASET = "ClassificationDataset"
CLASSIFICATION_SENTENCES = "sentences.data"
CLASSIFICATION_FEATURES = "features.data"
INCLUSION_OUTPUT = "analysis/temporary_output_inclusion.data"
INCLUSION_REFERENCE = "analysis/temporary_reference_inclusion.data"
INCLUSION_RESULT = "analysis/inclusion_result.data"
EXCLUSION_OUTPUT = "analysis/temporary_output_exclusion.data"
EXCLUSION_REFERENCE = "analysis/temporary_reference_exclusion.data"
EXCLUSION_RESULT = "analysis/exclusion_result.data"
BACKWARD_DATASET = CLASSIFICATION_DATASET+"/translationData/"