__author__ = 'Harsh'
import numpy as np
import pandas as pd
import csv
import json
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn import feature_extraction,preprocessing

class OttoClassifier:

    training = ''
    trainingLabels = ''
    test = ''
    rd =  ''
    final = ''
    svm = ''
    tfidf = ''

    def __init__(self):
        self.self = self;
        self.loadData()

    def loadData(self):
        self.training = pd.read_csv('E:\\Kaggle\\Otto Classification\\train.csv', index_col ='id')
        self.prepareForTraining()

    def printTrain(self):
        print self.training
        print self.trainingLabels

    def prepareForTraining(self):
        self.trainingLabels = self.training.target
        self.training = self.training.drop(['target'], axis= 1)

    def tfidfFeatures(self):
        print 'Tf_Idf feature Extraction'
        self.tfidf = feature_extraction.text.TfidfTransformer()
        self.training = self.tfidf.fit_transform(self.training).toarray()

    def processLabels(self):
        print 'Encoding Labels'
        lenc = preprocessing.LabelEncoder()
        self.trainingLabels = lenc.fit_transform(self.trainingLabels)

    def Classifier(self):
        self.rd = RandomForestClassifier(n_estimators=400,max_depth=5)
        self.rd.fit(self.training,self.trainingLabels)
        print 'Fitting Complete'

    def loadTest(self):
        self.test = pd.read_csv('E:\\Kaggle\\Otto Classification\\test.csv', index_col ='id')
        #self.test = self.test.drop(['id'],axis=1)
        self.test = self.tfidf.fit_transform(self.test).toarray()
        print self.test

    def predictionOnTest(self):
        print 'Start Predicting'
        self.final = self.rd.predict_proba(self.test)
        print self.final

    def prepareAndSaveasCSV(self):
        ind = [i for i in range(1,144369)]
        print ind
        cols = ['Class_1','Class_2','Class_3','Class_4','Class_5','Class_6','Class_7','Class_8','Class_9']
        finalFrame = pd.DataFrame(self.final, index = ind, columns=cols )
        finalFrame['id'] = pd.Series(ind , index = finalFrame.index)
        finalFrameCols = finalFrame.columns.tolist()
        finalFrameCols = finalFrameCols[-1:] + finalFrameCols[:-1]
        print finalFrameCols
        finalFrame = finalFrame[finalFrameCols]
        finalFrame = finalFrame.set_index('id')
        finalFrame.to_csv('E:\\Kaggle\\Otto Classification\\OuputWithOnlyRandomForest.csv')

############
## Main Program

ot = OttoClassifier()
ot.printTrain()
ot.tfidfFeatures()
ot.processLabels()
ot.Classifier()
ot.loadTest()
ot.predictionOnTest()
ot.prepareAndSaveasCSV()