__author__ = 'Harsh'
import pandas as pd
import numpy as np
from sklearn import ensemble, feature_extraction, preprocessing
from sklearn.ensemble import ExtraTreesClassifier

# import data
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
sample = pd.read_csv('sampleSubmission.csv')

# drop ids and get labels
labels = train.target.values
train = train.drop('id', axis=1)
train = train.drop('target', axis=1)
test = test.drop('id', axis=1)

# transform counts to TFIDF features
tfidf = feature_extraction.text.TfidfTransformer()
train = tfidf.fit_transform(train).toarray()
test = tfidf.transform(test).toarray()

# encode labels
lbl_enc = preprocessing.LabelEncoder()
labels = lbl_enc.fit_transform(labels)

# train a ExtraTree classifier
clf = ensemble.ExtraTreesClassifier(n_estimators=700,n_jobs=-1)
clf.fit(train, labels)

# predict on test set
preds = clf.predict_proba(test)

# create submission file
ind = [i for i in range(1,144369)]
print ind
cols = ['Class_1','Class_2','Class_3','Class_4','Class_5','Class_6','Class_7','Class_8','Class_9']
finalFrame = pd.DataFrame(preds, index = ind, columns=cols )
finalFrame['id'] = pd.Series(ind , index = finalFrame.index)
finalFrameCols = finalFrame.columns.tolist()
finalFrameCols = finalFrameCols[-1:] + finalFrameCols[:-1]
print finalFrameCols
finalFrame = finalFrame[finalFrameCols]
finalFrame = finalFrame.set_index('id')
finalFrame.to_csv('extraTree.csv')
