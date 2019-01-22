""" Train a model """
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, CSVLogger
from models import ResearchModels
from dataHAB import DataSet
import time
import os.path
import sys
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from inputXMLConfig import inputXMLConfig


# Train the model
def train(inDir, dataDir,data_type, seqName, seq_length, model, image_shape,
          batch_size, nb_epoch, featureLength):

    checkpointer = ModelCheckpoint(
        filepath=os.path.join(dataDir, 'checkpoints', model + '-' + data_type + \
            '.{epoch:03d}-{val_loss:.3f}.hdf5'), verbose=1, save_best_only=True)
    # Helper: TensorBoard
    tb = TensorBoard(log_dir=os.path.join(dataDir, 'logs', model))

    # Helper: Stop when we stop learning.
    #early_stopper = EarlyStopping(patience=2)
    early_stopper = EarlyStopping(monitor='val_acc', patience=2,  mode='auto')

    # Helper: Save results.
    timestamp = time.time()
    csv_logger = CSVLogger(os.path.join(dataDir, 'logs', model + '-' + 'training-' + \
        str(timestamp) + '.log'))

    data = DataSet(seqName, seq_length, inDir, dataDir)

    # Get samples per epoch.
    # Multiply by 0.7 to attempt to guess how much of data.data is the train set.
    steps_per_epoch = (len(data.data) * 0.7) // batch_size

    #X, Y = data.get_all_sequences_in_memory('train', data_type)
    X, Y, X_test, Y_test = data.get_all_sequences_in_memory2( data_type, 0.05)

    if model == 'svm':
        #tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-2, 1e-3, 1e-4, 1e-5],
        #             'C': [0.001, 0.10, 0.1, 10, 25, 50, 100, 1000]}]

        tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-2,  1e-4],
                     'C': [0.10,  10, 50, 1000]}]

        YI = np.int64(Y)
        Y_testI = np.int64(Y_test)
        #Cs = [0.001, 0.01, 0.1, 1, 10]
        Cs = [0.01, 0.1]
        #gammas = [0.001, 0.01, 0.1, 1]
        gammas = [0.01, 0.1]
        param_grid = {'C': Cs, 'gamma' : gammas}
        clf = GridSearchCV(SVC(kernel='rbf'), param_grid, cv=2)


#       clf = GridSearchCV(SVC(C=1), tuned_parameters, cv=3)
        #clf = SVC(C=1)
        # scoring='%s_macro' % score)
        fX = X.reshape(X.shape[0], seq_length*featureLength)
        clf.fit(fX, YI[:,1])
        fX_test = X_test.reshape(X_test.shape[0], seq_length*featureLength)
        svmScore = clf.score(fX_test, Y_testI[:,1])
        print("SVM score =  %f ." % svmScore)
    else:
        # Get the model.
        rm = ResearchModels(model, seq_length, None, features_length=featureLength)

        rm.model.fit(
                X,
                Y,
                batch_size=batch_size,
                #validation_data=(X_test, Y_test),
                validation_split=0.1,
                verbose=1,
                callbacks=[tb, early_stopper, csv_logger],
                epochs=nb_epoch)

        scores = rm.model.evaluate(X_test, Y_test, verbose=1)
        print("%s: %.2f%%" % (rm.model.metrics_names[1], scores[1]*100))

# Cross Validated Version of Train
def trainCV(inDir, dataDir,data_type, seqName, seq_length, model, image_shape,
          batch_size, nb_epoch, featureLength):
    """Set up training"""
    seed = 7
    checkpointer = ModelCheckpoint(
        filepath=os.path.join(dataDir, 'checkpoints', model + '-' + data_type + \
            '.{epoch:03d}-{val_loss:.3f}.hdf5'), verbose=1, save_best_only=True)
    # Helper: TensorBoard
    tb = TensorBoard(log_dir=os.path.join(dataDir, 'logs', model))

    # Helper: Stop when we stop learning.
    early_stopper = EarlyStopping(monitor='val_loss', patience=5, verbose =1)

    # Helper: Save results.
    timestamp = time.time()
    csv_logger = CSVLogger(os.path.join(dataDir, 'logs', model + '-' + 'training-' + \
        str(timestamp) + '.log'))

    data = DataSet(seqName, seq_length, inDir, dataDir)

    # Get samples per epoch.
    # Multiply by 0.7 to attempt to guess how much of data.data is the train set.
    steps_per_epoch = (len(data.data) * 0.7) // batch_size

    X, Yhot = data.get_all_sequences_in_memory('all', data_type)

    Y = Yhot[:,1]

    kfold = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
    cvscores = []

    """Loop through Train and Test CV Datasets"""
    for train, test in kfold.split(X, Y):

        X_train =     X[train]
        X_test =      X[test]

        """Choose between SVM and other Models"""
        if model == 'svm':
            Y_train  =    Y[train]
            Y_test  =     Y[test]

            tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-2, 1e-3, 1e-4, 1e-5],
                     'C': [0.001, 0.10, 0.1, 10, 25, 50, 100, 1000]}]

            clf = GridSearchCV(SVC(C=1), tuned_parameters, cv=5, verbose=2)
                      # scoring='%s_macro' % score)
            fX_train = X_train.reshape(X_train.shape[0], seq_length*featureLength)

            clf.fit(fX_train, Y_train)
            fX_test =  X_test.reshape(X_test.shape[0], seq_length*featureLength)
            svmScore = clf.score(fX_test, Y_test)
            print("SVM score =  %f ." % svmScore)
            cvscores.append(svmScore * 100)
        else:

            Y_train  =    Yhot[train]
            Y_test  =     Yhot[test]
            rm = ResearchModels(model, seq_length, None, features_length=featureLength)

            rm.model.fit(
                X_train,
                Y_train,
                batch_size=batch_size,
                validation_split=0.33,
                verbose=1,
                callbacks=[tb, early_stopper, csv_logger],
                epochs=nb_epoch)
            scores = rm.model.evaluate(X_test, Y_test, verbose=1)
            print("%s: %.2f%%" % (rm.model.metrics_names[1], scores[1]*100))
            cvscores.append(scores[1] * 100)
    print("%.2f%% (+/- %.2f%%)" % (numpy.mean(cvscores), numpy.std(cvscores)))
    print(cvscores)


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
    train(cnfg.inDir, cnfg.dataDir, 'features', cnfg.seqName, cnfg.seqLength, cnfg.model, None,
          cnfg.batchSize, cnfg.epochNumber, cnfg.featureLength)
    #train(inDir, dataDir, 'features', seqName, seqLength, model, None,
    #      batchSize, epochNumber, featureLength)

if __name__ == '__main__':
    main(sys.argv[1:])
