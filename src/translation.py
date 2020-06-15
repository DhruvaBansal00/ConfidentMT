class Translation:
    def __init__(self):
        self.source = None
        self.reference = None
        self.hypothesis = None
        self.features = None
        self.trnID = None

        #individualFeatures
        self.avgLP = None
        self.minLP = None
        self.medianLP = None 
        self.maxLP = None
        self.sbleu = None ##sentence bleu
        self.rareSource = None 
        self.rareTrans = None
        self.repeatSource = None
        self.repeatTrans = None
        self.sumLP = None 
        self.backwardAvgLP = None 
        self.lmScore = None 
        self.sentEndsTrans = None
        self.sentEndsSource = None 
        self.unigram = None 
        self.bigram = None 
        self.trigram = None
        self.transLength = None 
        self.sourceLength = None
    
    def populateFeatures(self):
        self.features = [self.avgLP, self.minLP, self.medianLP, self.maxLP, self.sbleu, self.rareSource, self.rareTrans,
                      self.rareSource - self.rareTrans, self.repeatTrans, self.repeatSource, self.sumLP, self.backwardAvgLP, self.lmScore, self.sentEndsTrans,
                      self.sentEndsSource, self.sentEndsSource - self.sentEndsTrans, self.unigram, self.bigram, self.trigram,
                      self.transLength, self.sourceLength, self.sourceLength/(float(self.transLength))]
    def __repr__(self):
        return str(self.sbleu)