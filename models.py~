# A collection of Keras temporal models to classify temporal stage of HAB
# classifier
#
# Copyright: (c) 2019 Paul Hill



from keras.layers import (Dense, Flatten, Dropout, ZeroPadding3D, Activation,
    BatchNormalization)
import keras
from keras_self_attention import SeqSelfAttention
from keras.layers.recurrent import LSTM
from keras.models import Sequential, load_model
from keras.optimizers import Adam, RMSprop
from keras.layers.wrappers import TimeDistributed
from keras.layers.convolutional import (Conv1D, Conv2D, MaxPooling3D, Conv3D, MaxPooling1D,
    MaxPooling2D)
from keras.layers import Bidirectional
from keras import regularizers
from keras.regularizers import l2
from collections import deque
import sys

class ResearchModels():
    def __init__(self, model, seq_length,
                 saved_model=None, features_length=20480):
        """
        `model` = one of:
            lstm0
            lstm1
            lstm2
            mlp1
            mlp2
        `seq_length` = the length of our video sequences
        `saved_model` = the path to a saved Keras model to load
        """

        # Set defaults.
        self.seq_length = seq_length
        self.load_model = load_model
        self.saved_model = saved_model
        self.feature_queue = deque()

        # Set the metrics. Only use top k if there's a need.
        metrics = ['accuracy']

        # Get the appropriate model.
        if self.saved_model is not None:
            print("Loading model %s" % self.saved_model)
            self.model = load_model(self.saved_model)
        elif model == 'lstm0':
            print("Loading LSTM0 model.")
            self.input_shape = (seq_length, features_length)
            self.model = self.lstm0()
        elif model == 'lstm1':
            print("Loading LSTM1 model.")
            self.input_shape = (seq_length, features_length)
            self.model = self.lstm1()
        elif model == 'lstm2':
            print("Loading LSTM2 model.")
            self.input_shape = (seq_length, features_length)
            self.model = self.lstm2()
        elif model == 'lstm3':
            print("Loading LSTM3 model.")
            self.input_shape = (seq_length, features_length)
            self.model = self.lstm3()
        elif model == 'lstm0Attention':
            print("Loading lstm0Attention model.")
            self.input_shape = (seq_length, features_length)
            self.model = self.lstm0Attention()
        elif model == 'mlp1':
            print("Loading simple MLP1.")
            self.input_shape = (seq_length, features_length)
            self.model = self.mlp1()
        elif model == 'mlp2':
            print("Loading simple MLP2.")
            self.input_shape = (seq_length, features_length)
            self.model = self.mlp2()
        else:
            print("Unknown network.")
            sys.exit()

        # Now compile the network.
        optimizer = Adam(lr=1e-5, decay=1e-5)
        self.model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=metrics)
        #self.model.compile(loss='hinge',  optimizer='adadelta',  metrics=['accuracy'])

        self.model.build(self.input_shape)
        print(self.model.summary())

    def lstm0(self):
        """Build a simple LSTM network. We pass the extracted features from
        our CNN to this model"""
        # Model.
        model = Sequential()
        #model.add(Bidirectional(LSTM(2048, return_sequences=False,
        model.add(LSTM(512, return_sequences=False,
                       input_shape=self.input_shape))
        model.add(Dropout(0.5))
        model.add(BatchNormalization())
        model.add(Dense(512, activation='relu'))
        model.add(Dropout(0.5))
        model.add(BatchNormalization())
        model.add(Dense(2, activation='softmax'))
        return model

    def lstm1(self):
        """Build a simple LSTM network. We pass the extracted features from
        our CNN to this model"""
        # Model.
        model = Sequential()
        model.add(LSTM(128, return_sequences=False,
                       input_shape=self.input_shape,
                       dropout=0.5))
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(2, activation='softmax'))
        return model

    def lstm2(self):
        """Build a simple LSTM network. We pass the extracted features from
        our CNN to this model."""
        # Model.
        model = Sequential()
        model.add(LSTM(512, return_sequences=False,
                       input_shape=self.input_shape))
        model.add(BatchNormalization())
        model.add(Dense(512, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dense(2, activation='softmax'))
        return model

    def lstm3(self):
        """Build a simple LSTM network. We pass the extracted features from
        our CNN to this model"""
        # Model.
        model = Sequential()
        #model.add(TimeDistributed(Conv1D(filters=8, kernel_size=5, strides=2,  activation='relu'), input_shape=(self.input_shape[0], self.input_shape[1],1)))
        #model.add(TimeDistributed(MaxPooling1D(pool_size=16)))
        #model.add(TimeDistributed(Flatten()))
        #model.add(TimeDistributed(MaxPooling1D(pool_size=4), input_shape=(None, self.input_shape[0], self.input_shape[1])))
        #model.add(TimeDistributed(Dropout(0.5)))
        #model.add(TimeDistributed(MaxPooling1D(pool_size=2)))
        #model.add(TimeDistributed(Conv1D(filters=64, kernel_size=10, strides = 10, activation='relu') ), input_shape=self.input_shape)
        #model.add(MaxPooling1D(pool_size=2, input_shape = (self.input_shape[0], self.input_shape[1])))
        #model.add(Bidirectional(LSTM(2048, return_sequences=False,
        model.add(LSTM(units=512, return_sequences=True,input_shape=self.input_shape))
        model.add(Dropout(0.5))
        model.add(BatchNormalization())
        model.add(LSTM(units=256, return_sequences=False,input_shape=self.input_shape))
        #model.add(SeqSelfAttention(attention_activation='sigmoid'))
        model.add(Dropout(0.5))
        model.add(BatchNormalization())
        #model.add(LSTM(512, return_sequences=True))
        #model.add(Dropout(0.5))
        #model.add(LSTM(256, return_sequences=False))
        #model.add(Dropout(0.5))
        #model.add(TimeDistributed(Conv1D(filters=32, kernel_size=3,  activation='relu')))
        #model.add(TimeDistributed(MaxPooling1D(pool_size=16)))
        #model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.5))
        model.add(BatchNormalization())
        model.add(Dense(2, activation='softmax'))
        return model

    def lstm0Attention(self):
        """Build a simple LSTM network. We pass the extracted features from
        our CNN to this model"""
        # Model.
        model = Sequential()
        model.add(LSTM(256, return_sequences=True, input_shape=self.input_shape))
        model.add(SeqSelfAttention(attention_activation='sigmoid'))
        #model.add(LSTM(512, return_sequences=False, input_shape=self.input_shape))
        #model.add(SeqSelfAttention(attention_activation='sigmoid'))
        #model.add(Dropout(0.5))
        #model.add(BatchNormalization())
        model.add(Dense(512, activation='relu'))
        model.add(Dropout(0.5))
        model.add(BatchNormalization())
        model.add(Dense(2, activation='softmax'))
        return model

    def mlp1(self):
        """Build a simple MLP. It uses extracted features as the input
        because of the otherwise too-high dimensionality."""
        # Model.
        model = Sequential()
        model.add(Flatten(input_shape=self.input_shape))
        model.add(Dense(512, use_bias=False))
        model.add(BatchNormalization())
        model.add(Dense(512, use_bias=False))
        model.add(BatchNormalization())

        model.add(Dense(2, activation='softmax'))

        return model

    def mlp2(self):
        """Build a simple MLP. It uses extracted features as the input
        because of the otherwise too-high dimensionality."""
        # Model.
        model = Sequential()
        model.add(Flatten(input_shape=self.input_shape))
        model.add(Dense(512))
        model.add(Dropout(0.5))
        model.add(Dense(512))
        model.add(Dropout(0.5))
        model.add(Dense(2, activation='softmax'))

        return model

    def mlp3(self):
        """Build a simple MLP. It uses extracted features as the input
        because of the otherwise too-high dimensionality."""
        # Model.
        model = Sequential()
        model.add(Flatten(input_shape=self.input_shape))
        model.add(Dense(256, use_bias=False, kernel_regularizer=regularizers.l2(0.003)))
        model.add(BatchNormalization())
        model.add(Dense(256, use_bias=False, kernel_regularizer=regularizers.l2(0.003)))
        model.add(BatchNormalization())
        model.add(Dense(2, activation='softmax'))

        return model

