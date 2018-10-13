"""
"""
import os
import glob
import numpy as np
import os.path
from dataHAB import DataSet
from extractor import Extractor
import sys
from inputXMLConfig import inputXMLConfig

from keras.preprocessing import image
from PIL import Image



try:
    import scipy
    # scipy.linalg cannot be accessed until explicitly imported
    from scipy import linalg
    # scipy.ndimage cannot be accessed until explicitly imported
    from scipy import ndimage
except ImportError:
    scipy = None


from scipy.ndimage import rotate

#from keras.preprocessing.image import ImageDataGenerator
#from keras.preprocessing.image import ImageDataGenerator


def augment_data(inDir, seqName, dataDir, seqLength, cnnModel):

    # Get the dataset.

    data = DataSet(seqName, seqLength, inDir, dataDir)

    # Loop through data.

    max_depth = 0
    bottom_most_dirs = []

    #thisOutputDir = '/Users/csprh/tmp/CNNIms/florida3Aug'
    thisOutputDir = '/mnt/storage/home/csprh/scratch/HAB/CNNIms/florida3Aug'
    # data = listOfDirectories;
    for thisDir in data.data:



        #for ii in [1,2,3,4,5,6,7,8]:
        for augNumber in range(0,8):
            createAug(augNumber, thisDir, thisOutputDir)


def createAug(augNumber, thisDir, thisOutputDir):


        head1, tail1 = os.path.split(thisDir)
        head2, tail2 = os.path.split(head1)
        head3, tail3 = os.path.split(head2)

        outputDir = thisOutputDir + os.path.sep + tail3 + os.path.sep + tail2 + os.path.sep +  tail1 + chr(augNumber+65)

        for modNumber in range(1,11):
            dirpath1 = thisDir + os.path.sep + str(modNumber)
            dirpath2 = outputDir + os.path.sep + str(modNumber)
            os.system("mkdir -p " + dirpath2)

            frames = sorted(glob.glob(os.path.join(dirpath1, '*jpg')))
            sequence = []
            for images in frames:
                thisImage = images
                img = image.load_img(thisImage)
                x = image.img_to_array(img)
                head4, tail4 = os.path.split(thisImage)

                rotImage = rotateImage(x,augNumber)
                aRotImage = image.array_to_img(rotImage)

                outname = dirpath2 +  os.path.sep + tail4

                aRotImage.save(outname)

def rotateImage(x,augNumber):

    flip = 0
    if augNumber == 0:
        theta = 0
    elif augNumber == 1:
        theta = 90
    elif augNumber == 2:
        theta = 180
    elif augNumber == 3:
        theta = 270
    elif augNumber == 4:
        theta = 0
        flip = 1
    elif augNumber == 5:
        theta = 90
        flip = 1
    elif augNumber == 6:
        theta = 180
        flip = 1
    elif augNumber == 7:
        theta = 270
        flip = 1


    if flip == 1:
        x = x[::-1, ...]

    transform_parameters = {'theta': theta}
                                #'tx': tx,
                                #'ty': ty,
                                #'shear': shear,
                                #'zx': zx,
                                #'zy': zy,
                                #'flip_horizontal': flip_horizontal,
                                #'flip_vertical': flip_vertical,
                                #'channel_shift_intensity': channel_shift_intensity,
                                #'brightness': brightness}

    #img_gen = ImageDataGenerator()
    #x = img_gen.apply_transform(x, transform_parameters)
    x = rotate(x, theta, reshape=False, mode="nearest")
    #x = image.apply_transform(x,transform_parameters)
    #x2 = image.apply_affine_transform(x, theta=theta)

    #x2 = apply_affine_transform(x, theta=theta)
    return x



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
    augment_data(cnfg.inDir, cnfg.seqName, cnfg.dataDir, cnfg.seqLength, cnfg.cnnModel)


if __name__ == '__main__':
    main(sys.argv[1:])

