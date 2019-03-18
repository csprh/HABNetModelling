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
import pudb; pu.db

import scipy.io as sio

mat_fname = '/home/cosc/csprh/linux/HABCODE/code/HAB/extractData/work/florida_2003-2018-50K.mat'
mat_contents = sio.loadmat(mat_fname)

lonArray = mat_contents['longitude']
latArray = mat_contents['latitude']
datArray = mat_contents['sample_date']
cntArray = mat_contents['count2']

for i in range(0, 10):

    ii = i *100
    lat = latArray[ii,0]
    lon = lonArray[ii,0]
    sample_date = datArray[ii,0]
    cnt = cntArray[ii,0]
    print('Count = ' + str(cnt))
    #h5name = '/Users/csprh/Dlaptop/MATLAB/MYCODE/HAB/WORK/HAB/florida2/Cube_09073_09081_737173.h5'
    #outputDirectory = '/Users/csprh/Dlaptop/MATLAB/MYCODE/HAB/WORK/HAB/CNNIms'
    #h5name = '/home/cosc/csprh/linux/HABCODE/scratch/HAB/tmpTest/testCubes/Cube_09073_09081_737173.h5'
    #mstringApp = '/Applications/MATLAB_R2016a.app/bin/matlab'


    h5name = '/home/cosc/csprh/linux/HABCODE/scratch/HAB/tmpTest/testCubes/Cube_Test.h5'
    outputDirectory = '/home/cosc/csprh/linux/HABCODE/scratch/HAB/tmpTest/CNNIms'
    mstringApp = 'matlab'

    # GENERATE DATACUBE FROM LAT, LON, DATE (not necessary if you already have datacube).
    mstring = mstringApp + ' -nosplash -r \"genSingleH5sWrapper ' + str(lat) + ' ' + str(lon) + ' ' + str(sample_date) + ' ' +  h5name  + '\;quit;"'
    os.system(mstring)

    # GENERATE IMAGES FROM DATA CUBE
    mstring = mstringApp + ' -nosplash -r \"outputImagesFromDataCubeScript ' +  h5name + ' ' + outputDirectory + '\;quit;"'
    os.system(mstring)

    # EXTRACT BOTTLENECK FEATURES FROM IMAGES
    extract_features.main(['cnfgXMLs/NASNet11_lstm0.xml', outputDirectory])

    # GENERATE CLASSIFICATION FROM BOTTLENECK FEATURES AND TRAINED MODEL
    testHAB.main(['cnfgXMLs/NASNet11_lstm0.xml', outputDirectory])



