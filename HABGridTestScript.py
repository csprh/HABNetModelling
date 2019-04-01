# This script demonstrates how to generate a classification from a
# lat, lon and date (or a pre downloaded dataCube )
# Firstly, the lat, lon and date is defined and a working directory (output directory)
#
# The following scripts are called

# genSingleH5sWrapper.m: A Matlab script that generates a datacube (H5 format)

# outputImagesFromDataCubeScript: A Matlab script that generates quantised
# images that are put into outputDirectory (from the datacube)

# extract_features: A python script that extracts bottle neck features using
# CNNs defined in the config file xml.  The features are stored in
# outputDirectory

# testHAB: A python script that uses the model defined in the xml file and
# generates a classification based on the datacube extracted images
# The classification probablity is printed to stdout.
#
# THE UNIVERSITY OF BRISTOL: HAB PROJECT
# Author Dr Paul Hill March 2019

import sys
import os
import extract_features
import testHAB

import xml.etree.ElementTree as ET
from xml.etree.ElementTree import Element



import pudb; pu.db

sample_date = 737174;

#mstringApp = '/Applications/MATLAB_R2016a.app/bin/matlab'
xmlName = '/home/cosc/csprh/linux/HABCODE/code/HAB/extractData/configHABunderDesk.xml'

mstringApp = 'matlab'

root = ET.parse(xmlName)
elem = Element("confgData")
elem.attrib["testDate"] = "737174"
root.write(xmlName)
outputDirectory = elem.attrib["testImsDir"]


# GENERATE DATACUBE FROM LAT, LON, DATE (not necessary if you already have datacube).
mstring = mstringApp + ' -nosplash -r \"test_getAllH5s; quit;"'
os.system(mstring)

# GENERATE IMAGES FROM DATA CUBE
mstring = mstringApp + ' -nosplash -r \"test_cubeSequence; quit;"'
os.system(mstring)

# EXTRACT BOTTLENECK FEATURES FROM IMAGES
extract_features.main(['cnfgXMLs/NASNet11_lstm0.xml', outputDirectory])

# GENERATE CLASSIFICATION FROM BOTTLENECK FEATURES AND TRAINED MODEL
testHAB.main(['cnfgXMLs/NASNet11_lstm0.xml', outputDirectory])


