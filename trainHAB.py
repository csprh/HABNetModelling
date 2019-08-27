# A file to perform machine learning on HAB data
#
# Copyright: (c) 2019 Paul Hill

"""
This file contains a script that reads the XML configuration file and then
extracts a whole dataset of HAB datacube data.  An available list of ML
classifiers are available to classify the outputs previously extracted
bottleneck features.

By default it loads the configuration file NASNet11_lstm0.xml.  However it can
take one argument that specifies the config file to use i.e.
python3 trainHAB.py ./cnfgXMLs/NASNet11_lstm0.xml

The models can be chosen from the following

Keras based models
model = lstm0: BEST PERFORMING lstm (Batch normalisation and dropout)
model = lstm1: Dropout
model = lstm2: Batch Normalisation
model = mlp1:  Batch Normalisation
model = mlp2:  Dropout

These models save the model files in [seqName + model] H5 file

Non Keras based models
model = svm: Support Vector Machines (currently not working)
model = xbg: State of the art boosting method
model = RF:  Random Forest

Non Keras currently does not save model as they are not the best performing

"""

from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, CSVLogger
from sklearn.feature_selection import RFE
from models import ResearchModels
from dataHAB import DataSet
import time
import os.path
import sys
import numpy as np
import xgboost as xgb
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from inputXMLConfig import inputXMLConfig
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Train the model
def train(inDir, dataDir, seqName, seq_length, model,
          batch_size, nb_epoch, featureLength, SVDFeatLen, modNumber):

    modelNameInt = dataDir + seqName + '_' + model
    data = DataSet(seqName, seq_length,  inDir, dataDir, SVDFeatLen, modNumber)

    if SVDFeatLen == -1 :
       X_train, Y_train, X_test, Y_test = data.get_all_sequences_in_memory_prop(0.2)
    else :
       X_train, Y_train, X_test, Y_test = data.get_all_sequences_in_memory_svd(0.2)

    # Non Keras models 'Random Forest: RF....xgboost: xgb.....svm' are treated
    # separately here.  None are currently out performing keras based models
    if model == 'RF':
        Y_trainI = np.int64(Y_train)
        Y_testI = np.int64(Y_test)
        fX_train = X_train.reshape(X_train.shape[0], seq_length*featureLength)
        fX_test = X_test.reshape(X_test.shape[0], seq_length*featureLength)

        #scaling = MinMaxScaler(feature_range=(-1,1)).fit(fX)
        #fX = scaling.transform(fX)
        #fX_test = scaling.transform(fX_test)s
        rf=RandomForestClassifier(n_estimators=1000,
                                              criterion='entropy',
                                              max_depth=14,
                                              max_features='auto',
                                              random_state=42)

        ## This line instantiates the model.
        #param_grid = {'n_estimators': [900, 1100],'max_features': ['auto', 'sqrt', 'log2'],
        #    'max_depth' : [16,18,20,22],    'criterion' :['gini', 'entropy'] }
        #rf = GridSearchCV(estimator=rf, param_grid=param_grid, cv= 5)
        ## Fit the model on your training data.

        rf.fit(fX_train, Y_trainI[:,1])


        ## And score it on your testing data.
        rfScore = rf.score(fX_test, Y_testI[:,1])
        np.savetxt('rfImports.txt', rf.feature_importances_);
        print("RF Score = %f ." % rfScore)

        rfe = RFE(rf, n_features_to_select=1000, verbose =3 )

        rfe.fit(fX_train, Y_trainI[:,1])

        ## And score it on your testing data.
        rfeScore = rfe.score(fX_test, Y_testI[:,1])
        np.savetxt('rfe.txt', rfe.ranking_);
        print("RFE Score = %f ." % rfeScore)


    elif model == 'xgb':
        # Train xgboost
        Y_trainI = np.int64(Y_train)
        Y_testI = np.int64(Y_test)
        fX_train = X_train.reshape(X_train.shape[0], seq_length*featureLength)
        fX_test = X_test.reshape(X_test.shape[0], seq_length*featureLength)

        dtrain = xgb.DMatrix(fX_train, Y_trainI)
        dtest = xgb.DMatrix(fX_test, Y_testI)
        param = {'max_depth' : 3, 'eta' : 0.1, 'objective' : 'binary:logistic', 'seed' : 42}
        num_round = 50
        bst = xgb.train(param, dtrain, num_round, [(dtest, 'test'), (dtrain, 'train')])

        preds = bst.predict(dtest)
        preds[preds > 0.5] = 1
        preds[preds <= 0.5] = 0
        print("XGB score =  %f ." % accuracy_score(preds, Y_testI))

    elif model == 'svm':
        #Currently, SVMs do not work for very large bottleneck features.
        #tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-2, 1e-3, 1e-4, 1e-5],
        #             'C': [0.001, 0.10, 0.1, 10, 25, 50, 100, 1000]}]

        tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-2,  1e-4],
                     'C': [0.10,  10, 50, 1000]}]

        Y_trainI = np.int64(Y_train)
        Y_testI = np.int64(Y_test)

        Cs = [0.01, 0.1]
        gammas = [0.01, 0.1]
        param_grid = {'C': Cs, 'gamma' : gammas}
        clf = GridSearchCV(SVC(kernel='rbf'), param_grid, cv=2)

        fX_train = X_train.reshape(X_train.shape[0], seq_length*featureLength)
        scaling = MinMaxScaler(feature_range=(-1,1)).fit(fX)
        fX_train = scaling.transform(fX_train)
        fX_test = X_test.reshape(X_test.shape[0], seq_length*featureLength)
        fX_test = scaling.transform(fX_test)
        clf.fit(fX_train, Y_trainI[:,1])
        svmScore = clf.score(fX_test, Y_testI[:,1])
        print("SVM score =  %f ." % svmScore)
    else:

        modelName = modelNameInt + '.h5'
        modelNameBest = modelNameInt + '_best.h5'

        checkpointer = ModelCheckpoint(
        filepath=os.path.join(dataDir, 'checkpoints', model + \
            '.{epoch:03d}-{val_loss:.3f}.hdf5'), verbose=1, save_best_only=True)
        # Helper: TensorBoard
        tb = TensorBoard(log_dir=os.path.join(dataDir, 'logs', model))

        # Helper: Stop when we stop learning.
        early_stopper = EarlyStopping(monitor='val_acc', patience=500,  mode='auto')

        # Helper: Save results.
        timestamp = time.time()
        csv_logger = CSVLogger(os.path.join(dataDir, 'logs', model + '-' + 'training-' + \
           str(timestamp) + '.log'))

        # Get the model.
        rm = ResearchModels(model, seq_length, None,features_length=featureLength)

        filepath=dataDir + "weightsbest.hdf5"
        checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
        rm.model.fit(
                X_train,
                Y_train,
                batch_size=batch_size,
                validation_split=0.1,
                verbose=1,
                callbacks=[tb, early_stopper, csv_logger, checkpoint],
                epochs=nb_epoch)

        rm.model.save(modelName)
        rm.model.load_weights(filepath)
        rm.model.save(modelNameBest)

        scores = rm.model.evaluate(X_test, Y_test, verbose=1)
        print("%s: %.2f%%" % (rm.model.metrics_names[1], scores[1]*100))

"""Main Thread"""
def main(argv):
    """Settings Loaded from Xml Configuration"""

    import pudb; pu.db

    if (len(argv)==0):
        xmlName = './cnfgXMLs/NASNet11_lstm0.xml'
    else:
        xmlName = argv[0]

    cnfg = inputXMLConfig(xmlName)
    train(cnfg.inDir, cnfg.dataDir, cnfg.seqName, cnfg.seqLength, cnfg.model,
          cnfg.batchSize, cnfg.epochNumber, cnfg.featureLength, cnfg.SVDFeatLen, cnfg.modNumber)

if __name__ == '__main__':
    main(sys.argv[1:])
