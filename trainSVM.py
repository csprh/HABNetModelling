""" Train a model """
from models import ResearchModels
from dataHAB import DataSet
import time
import os.path
import xml.etree.ElementTree as ET
import sys
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
import numpy

# Train the model
def train(inDir, dataDir,data_type, seqName, seq_length, model, image_shape,
          batch_size, nb_epoch, featureLength):


    data = DataSet(seqName, seq_length, inDir, dataDir)

    # Get samples per epoch.
    # Multiply by 0.7 to attempt to guess how much of data.data is the train set.
    steps_per_epoch = (len(data.data) * 0.7) // batch_size

    X, y = data.get_all_sequences_in_memory('train', data_type)
    X_test, y_test = data.get_all_sequences_in_memory('test', data_type)

    tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-2, 1e-3, 1e-4, 1e-5],
                     'C': [0.001, 0.10, 0.1, 10, 25, 50, 100, 1000]}]

    clf = GridSearchCV(SVC(C=1), tuned_parameters, cv=3)
                      # scoring='%s_macro' % score)
    fX = X.reshape(X.shape[0], seq_length*featureLength)
    clf.fit(fX, y[:,1])
    fX_test = X_test.reshape(X_test.shape[0], seq_length*featureLength)
    svmScore = clf.score(fX_test, y_test[:,1])
    print("SVM score =  %f ." % svmScore)


"""Main Thread"""
def main(argv):
    """Settings Loaded from Xml Configuration"""
    # model can be one of lstm, mlp, svm
    #import pudb; pu.db

    if (len(argv)==0):
        xmlName = 'classifyHAB1.xml'
    else:
        xmlName = argv[0]

    tree = ET.parse(xmlName)
    root = tree.getroot()

    for child in root:
        thisTag = child.tag
        thisText = child.text
        if thisTag == 'inDir':
            inDir = thisText
        elif thisTag == 'dataDir':
            dataDir = thisText
        elif thisTag == 'seqName':
            seqName = thisText
        elif thisTag == 'model':
            model = thisText
        elif thisTag == 'cnnModel':
            cnnModel = thisText
        elif thisTag == 'featureLength':
            featureLength = int(thisText)
        elif thisTag == 'seqLength':
            seqLength = int(thisText)
        elif thisTag == 'batchSize':
            batchSize = int(thisText)
        elif thisTag == 'epochNumber':
            epochNumber = int(thisText)
    trainCV(inDir, dataDir, 'features', seqName, seqLength, model, None,
          batchSize, epochNumber, featureLength)
    #train(inDir, dataDir, 'features', seqName, seqLength, model, None,
    #      batchSize, epochNumber, featureLength)

if __name__ == '__main__':
    main(sys.argv[1:])
