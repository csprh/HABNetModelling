# This file contains a script to extract bottleneck features from each image
# within a set of datacubes.
#
# Copyright: (c) 2019 Paul Hill

"""
This file contains a function that loops through all of the input datacube
images and extracts and contatenates bottleneck features from them using a
specified CNN.

By default it loads the configuration file NASNet11_lstm0.xml.  However it can
take one argument that specifies the config file to use and also the directory where
datacubes are to be found.
i.e. python3 extract_features.py ./cnfgXMLs/NASNet11_lstm0.xml dirIn
"""

import glob
import numpy as np
import os.path
from dataHAB import DataSet
from extractor import Extractor
import sys
from inputXMLConfig import inputXMLConfig



def extract(inDir, seqName, dataDir, seqLength):

    # Get the dataset.
    data = DataSet(seqName, seqLength, inDir, dataDir)
    # get the model.
    model = Extractor(seqName)

    # Loop through data.
    max_depth = 0
    bottom_most_dirs = []


    # data = listOfDirectories;
    for thisDir in data.dataLowest:

        # Get the path to the sequence for this video.
        npypath = os.path.join(thisDir, seqName)

        frames = sorted(glob.glob(os.path.join(thisDir, '*png')))
        sequence = []
        sequence2 = []
        sequence4 = []
        sequence6 = []
        sequence8 = []
        sequence2_b = []
        sequence4_b = []
        sequence6_b = []
        sequence8_b = []
        ind = 0
        for image in frames:
            ind = ind + 1

            features = model.extract(image)
            if ind > 2:
                sequence2.append(features)
            elif ind > 4:
                sequence4.append(features)
            elif ind > 6:
                sequence6.append(features)
            elif ind > 8:
                sequence8.append(features)
            if ind < 3:
                sequence2_b.append(features)
            elif ind < 5:
                sequence4_b.append(features)
            elif ind < 7:
                sequence6_b.append(features)
            elif ind < 9:
                sequence8_b.append(features)
            sequence.append(features)
        # Save the sequence.
        np.save(npypath, sequence)
        np.save(npypath +'_2', sequence2)
        np.save(npypath +'_4', sequence4)
        np.save(npypath +'_6', sequence6)
        np.save(npypath +'_8', sequence8)
        np.save(npypath +'_2_b', sequence2_b)
        np.save(npypath +'_4_b', sequence4_b)
        np.save(npypath +'_6_b', sequence6_b)
        np.save(npypath +'_8_b', sequence8_b)
    """Main Thread"""
def main(argv):
    """Settings Loaded from Xml Configuration"""
    # model can be one of lstm, mlp, svm
    # import pudb; pu.db

    if (len(argv)==0):
        xmlName = 'cnfgXMLs/NASNet33_lstm0.xml'
    else:
        xmlName = argv[0]

    cnfg = inputXMLConfig(xmlName)
    if (len(argv)==2):
        cnfg.inDir = argv[1]

    extract(cnfg.inDir, cnfg.seqName, cnfg.dataDir, cnfg.seqLength)

if __name__ == '__main__':
    main(sys.argv[1:])

