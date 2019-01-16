"""
This script generates extracted features for each video, which other
models make use of.
"""
import glob
import numpy as np
import os.path
from dataHAB import DataSet
from extractor import Extractor
import sys
from inputXMLConfig import inputXMLConfig



def extract(inDir, seqName, dataDir, seqLength, cnnModel):

    # Get the dataset.

    data = DataSet(seqName, seqLength, inDir, dataDir)
    # get the model.
    model = Extractor(cnnModel)

    # Loop through data.


    max_depth = 0
    bottom_most_dirs = []


    # data = listOfDirectories;
    for thisDir in data.dataLowest:

        # Get the path to the sequence for this video.
        npypath = os.path.join(thisDir, seqName)


        #frames = sorted(glob.glob(os.path.join(thisDir, '*png')))
        frames = sorted(glob.glob(os.path.join(thisDir, '*jpg')))
        sequence = []
        for image in frames:
            features = model.extract(image)
            sequence.append(features)

        # Save the sequence.
        np.save(npypath, sequence)


    """Main Thread"""
def main(argv):
    """Settings Loaded from Xml Configuration"""
    # model can be one of lstm, mlp, svm
    #import pudb; pu.db

    if (len(argv)==0):
        xmlName = 'classifyHAB1.xml'
    else:
        xmlName = argv[0]

    cnfg = inputXMLConfig(xmlName)
    extract(cnfg.inDir, cnfg.seqName, cnfg.dataDir, cnfg.seqLength, cnfg.cnnModel)


if __name__ == '__main__':
    main(sys.argv[1:])

