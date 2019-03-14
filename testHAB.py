# A file to perform machine learning on HAB data
#
# Copyright: (c) 2019 Paul Hill

"""
This file contains a script that inputs the configuration file and then
a single datapoint whole dataset of HAB datacube data.  An available list of ML
classifiers are available to classify the outputs previously extracted
bottleneck features.

By default it loads the configuration file classifyHAB1.xml.  However it can
take one argument that specifies the config file to use i.e.
python3 trainHAB.py ./cnfgXMLs/NASNet22_lstm0.xml
"""

from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, CSVLogger
from models import ResearchModels
from keras.models import Sequential, load_model
from dataHAB import DataSet
import time
import os.path
import sys
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from inputXMLConfig import inputXMLConfig
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Train the model
def test(inDir, dataDir, seqName, seq_length, model, featureLength, SVDFeatLen):


    modelNameInt = dataDir + seqName + '_' + model
    modelName = modelNameInt + '.h5'

    model = load_model(modelName)

    data = DataSet(seqName, seq_length,  inDir, dataDir, SVDFeatLen)
    X_test = data.get_extracted_sequenceAllMods(inDir)
    Y_new = model.predict_classes(X_test)
    Y_prob = model.predict_proba(X_test)

    sys.stdout.write(str(Y_prob))

"""Main Thread"""
def main(argv):
    """Settings Loaded from Xml Configuration"""

    import pudb; pu.db

    if (len(argv)==0):
        xmlName = './cnfgXMLs/NASNet11_lstm0.xml'
    else:
        xmlName = argv[0]

    cnfg = inputXMLConfig(xmlName)
    if (len(argv)==2):
        cnfg.inDir = argv[2]

    train(cnfg.inDir, cnfg.seqName, cnfg.seqLength, cnfg.model, cnfg.featureLength, cnfg.SVDFeatLen)

if __name__ == '__main__':
    main(sys.argv[1:])
