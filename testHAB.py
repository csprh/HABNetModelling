# A file to perform machine learning on HAB data
#
# Copyright: (c) 2019 Paul Hill

"""
This file contains a script that inputs an XML configuration file (containing
the definition of the trained models) and the directory containing a directory
structure of the bottleneck features extracted from the datacube images by
the defined CNNs.

The detection and probablity of a HAB is then output to stdout

By default it loads the configuration file SNet11_lstm0.xml.  However it can
take one argument that specifies the config file to use and the directory where
the datacube is to be found i.e. dirIn
i.e. python3 trainHAB.py ./cnfgXMLs/NASNet11_lstm0.xml dirIn
"""

from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, CSVLogger
from keras.models import load_model
from dataHAB import DataSet
import sys
from inputXMLConfig import inputXMLConfig
import numpy as np
# Train the model
def test(inDir, dataDir, seqName, seq_length, model, featureLength, SVDFeatLen):

    resultFile = inDir + 'classesProbs.txt' 
    open(resultFile,'w')
    modelNameInt = dataDir + seqName + '_' + model
    modelName = modelNameInt + '.h5'

    model = load_model(modelName)

    data, INDS = DataSet(seqName, seq_length, inDir, dataDir, SVDFeatLen)
    X_test = data.get_extracted_sequenceAllMods(inDir)
    #Y_new =  model.predict_classes(np.array( [X_test,]))
    #Y_prob = model.predict_proba(np.array( [X_test,]))
    Y_new =  model.predict_classes(np.array(X_test))
    Y_prob = model.predict_proba(np.array(X_test))



    for thisInd in range(len(Y_prob)):
        thisIND = INDS[thisInd]
        Y = Y_new[thisInd]
        P = Y_prob[thisInd,1]
        outString = "Index = %s, Class = %d, Probability = %f " %(thisIND, Y, P)
        write() 
    close() 


"""Main Thread"""
def main(argv):
    """Settings Loaded from Xml Configuration"""

    #import pudb; pu.db

    if (len(argv)==0):
        xmlName = './cnfgXMLs/NASNet11_lstm0.xml'
    else:
        xmlName = argv[0]

    cnfg = inputXMLConfig(xmlName)
    if (len(argv)==2):
        cnfg.inDir = argv[1]

    test(cnfg.inDir, cnfg.dataDir, cnfg.seqName, cnfg.seqLength, cnfg.model, cnfg.featureLength, cnfg.SVDFeatLen)

if __name__ == '__main__':
    main(sys.argv[1:])
