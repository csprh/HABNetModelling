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
import tensorflow as tf
from tensorflow import keras
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

from keras.preprocessing.image import ImageDataGenerator
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
    x = image.apply_transform(x,transform_parameters)
    #x2 = image.apply_affine_transform(x, theta=theta)

    #x2 = apply_affine_transform(x, theta=theta)
    return x

def transform_matrix_offset_center(matrix, x, y):
    o_x = float(x) / 2 + 0.5
    o_y = float(y) / 2 + 0.5
    offset_matrix = np.array([[1, 0, o_x], [0, 1, o_y], [0, 0, 1]])
    reset_matrix = np.array([[1, 0, -o_x], [0, 1, -o_y], [0, 0, 1]])
    transform_matrix = np.dot(np.dot(offset_matrix, matrix), reset_matrix)
    return transform_matrix

def apply_affine_transform(x, theta=0, tx=0, ty=0, shear=0, zx=1, zy=1,
                           row_axis=0, col_axis=1, channel_axis=2,
                           fill_mode='nearest', cval=0.):
    """Applies an affine transformation specified by the parameters given.
    # Arguments
        x: 2D numpy array, single image.
        theta: Rotation angle in degrees.
        tx: Width shift.
        ty: Heigh shift.
        shear: Shear angle in degrees.
        zx: Zoom in x direction.
        zy: Zoom in y direction
        row_axis: Index of axis for rows in the input image.
        col_axis: Index of axis for columns in the input image.
        channel_axis: Index of axis for channels in the input image.
        fill_mode: Points outside the boundaries of the input
            are filled according to the given mode
            (one of `{'constant', 'nearest', 'reflect', 'wrap'}`).
        cval: Value used for points outside the boundaries
            of the input if `mode='constant'`.
    # Returns
        The transformed version of the input.
    """
    if scipy is None:
        raise ImportError('Image transformations require SciPy. '
                          'Install SciPy.')
    transform_matrix = None
    if theta != 0:
        theta = np.deg2rad(theta)
        rotation_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],
                                    [np.sin(theta), np.cos(theta), 0],
                                    [0, 0, 1]])
        transform_matrix = rotation_matrix

    if tx != 0 or ty != 0:
        shift_matrix = np.array([[1, 0, tx],
                                 [0, 1, ty],
                                 [0, 0, 1]])
        if transform_matrix is None:
            transform_matrix = shift_matrix
        else:
            transform_matrix = np.dot(transform_matrix, shift_matrix)

    if shear != 0:
        shear = np.deg2rad(shear)
        shear_matrix = np.array([[1, -np.sin(shear), 0],
                                 [0, np.cos(shear), 0],
                                 [0, 0, 1]])
        if transform_matrix is None:
            transform_matrix = shear_matrix
        else:
            transform_matrix = np.dot(transform_matrix, shear_matrix)

    if zx != 1 or zy != 1:
        zoom_matrix = np.array([[zx, 0, 0],
                                [0, zy, 0],
                                [0, 0, 1]])
        if transform_matrix is None:
            transform_matrix = zoom_matrix
        else:
            transform_matrix = np.dot(transform_matrix, zoom_matrix)

    if transform_matrix is not None:
        h, w = x.shape[row_axis], x.shape[col_axis]
        transform_matrix = transform_matrix_offset_center(
            transform_matrix, h, w)
        x = np.rollaxis(x, channel_axis, 0)
        final_affine_matrix = transform_matrix[:2, :2]
        final_offset = transform_matrix[:2, 2]

        channel_images = [scipy.ndimage.interpolation.affine_transform(
            x_channel,
            final_affine_matrix,
            final_offset,
            order=1,
            mode=fill_mode,
            cval=cval) for x_channel in x]
        x = np.stack(channel_images, axis=0)
        x = np.rollaxis(x, 0, channel_axis + 1)
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

