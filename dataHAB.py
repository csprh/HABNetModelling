"""
Class for managing our data.
"""
import numpy as np
import random
import os.path
import threading
from keras.utils import to_categorical


class threadsafe_iterator:
    def __init__(self, iterator):
        self.iterator = iterator
        self.lock = threading.Lock()

    def __iter__(self):
        return self

    def __next__(self):
        with self.lock:
            return next(self.iterator)


def threadsafe_generator(func):
    """Decorator"""
    def gen(*a, **kw):
        return threadsafe_iterator(func(*a, **kw))
    return gen


class DataSet():

    def __init__(self, seqName, seq_length, inDir, dataDir,  modNumber=11,image_shape=(224, 224, 3)):

        self.seq_length = seq_length
        self.inDir = inDir
        self.dataDir = dataDir
        self.seqName = seqName

        # Get the data.
        self.dataLowest = self.get_data(self.inDir)
        self.data = self.extract_data(self.dataLowest)
        self.image_shape = image_shape
        self.modNumber = modNumber

    @staticmethod
    def get_data(inDir):
        """Load our data from file."""

        max_depth = 0
        bottom_most_dirs = []
        for dirpath, dirnames, filenames in os.walk(inDir):
            depth = len(dirpath.split(os.sep))
            if max_depth < depth:
                max_depth = depth
                bottom_most_dirs = [dirpath]
            elif max_depth == depth:
                bottom_most_dirs.append(dirpath)

        return bottom_most_dirs

    @staticmethod
    def extract_data(dataLowest):
        """ Get rid of last layer of dataLowest and put into data """
        output = []
        bottom_most_dirs = []
        for x in dataLowest:
                head, tail = os.path.split(x)
                bottom_most_dirs.append(head)

        for x in bottom_most_dirs:
            if x not in output:
                output.append(x)

        return output

    def get_class_one_hot(self, path_str):
        """Given a class as a string, return its number in the classes
        list. This lets us encode and one-hot it for training."""
        # Encode it first.
        parts = path_str.split(os.path.sep)

        # Now one-hot it.
        label_hot = to_categorical(int(parts[-2]), 2)

        assert len(label_hot) == 2

        return label_hot

    def split_train_test_prop(self, prop):
        """Split the data into train and test groups (and return them and all)."""
        train = []
        test = []

        inde = 0
        dataLen = len(self.data)
        np.random.seed(0)
        rc = np.random.choice([0, 1], size=(dataLen,), p=[1-prop, prop])

        for item in self.data:
            if  rc[inde] == 0:
                train.append(item)
            else:
                test.append(item)
            inde = inde + 1
        return train, test

    def split_train_test(self):
        """Split the data into train and test groups (and return them and all)."""
        train = []
        test = []
        thisall = []

        for item in self.data:
            parts = item.split(os.path.sep)
            if parts[-3] == 'Train':
                train.append(item)
            else:
                test.append(item)
            thisall.append(item)
        return train, test, thisall


    def get_all_sequences_in_memory(self, train_test):
        """
        Load all the sequences into memory for speed (train, test or all)
        """
        # Get the right dataset.
        train, test, thisall = self.split_train_test()
        if train_test == 'train':
            data = train
            print("Loading %d samples into memory for training." % len(data))
        elif train_test == 'test':
            data = test
            print("Loading %d samples into memory for training." % len(data))
        elif train_test == 'all':
            data = thisall
            print("Loading all %d samples into memory" % len(thisall))

        X, Y = [], []
        for sample in data:

            sequence = self.get_extracted_sequenceAllMods(sample)

            X.append(sequence)
            Y.append(self.get_class_one_hot(sample))

        return np.array(X), np.array(Y)

    def get_all_sequences_in_memory_prop(self,  prop):
        """
        Load all the sequences into memory (in proportion) for speed (train, test)
        """
        # Get the right dataset.
        train, test = self.split_train_test_prop(prop)
        self.train = train
        self.test = test
        X1, Y1 = [], []
        X2, Y2 = [], []
        for sample in train:

            sequence = self.get_extracted_sequenceAllMods(sample)

            X1.append(sequence)
            Y1.append(self.get_class_one_hot(sample))

        for sample in test:

            sequence = self.get_extracted_sequenceAllMods(sample)

            X2.append(sequence)
            Y2.append(self.get_class_one_hot(sample))

        return np.array(X1), np.array(Y1), np.array(X2), np.array(Y2)


    def get_extracted_sequenceAllMods(self, filename):
        """Get the saved extracted features.  Concatenate all mods"""

        thisreturn = []
        for i in range(1,self.modNumber):

            thispath = filename + '/' + str(i) + '/' +  self.seqName + '.npy'
            thisfeats = np.load(thispath)

            if i == 1 :
                thisreturn = thisfeats
            else:
                thisreturn = np.concatenate((thisreturn, thisfeats), axis=1)
        return thisreturn

    def get_extracted_sequence(self,  filename):
        """Get a single image from the eighth modality."""

        path = filename + '/8/' + self.seqName + '.npy'
        if os.path.isfile(path):
            return np.load(path)
        else:
            return None

