from keras.preprocessing import image
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import preprocess_input as inception_v3_preprocessor
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.applications.inception_resnet_v2 import preprocess_input as  inception_resnet_v2_preprocessor
from keras.applications.vgg19 import VGG19
from keras.applications.vgg19 import preprocess_input as vgg19_preprocessor
from keras.applications.nasnet import NASNetLarge, NASNetMobile
from keras.applications.nasnet import preprocess_input as nasnet_preprocessor

#from keras.applications.inception_v3 import preprocess_input as inception_v3_preprocessor
from keras.models import Model, load_model, Sequential
from keras.layers import Input, Flatten
from cifar10vgg import cifar10vgg

import numpy as np



class Extractor():
    def __init__(self, cnnModel, weights=None):
        """Either load pretrained from imagenet, or load our saved
        weights from our own training."""

        self.cnnModel = cnnModel
        self.weights = weights  # so we can check elsewhere which model

        if weights is None:
            if cnnModel == 'InceptionV3':
                # Get model with pretrained weights.
                base_model = InceptionV3(
                    weights='imagenet',
                    include_top=True
                )

                # We'll extract features at the final pool layer.
                self.model = Model(
                    inputs=base_model.input,
                    outputs=base_model.get_layer('avg_pool').output
                )
                self.target_size = (299,299)
                self.preprocess_input = inception_v3_preprocessor
            elif cnnModel == 'VGG19':
                # Get model with pretrained weights.
                base_model = VGG19(
                    weights='imagenet',
                    include_top=True
                )

                # We'll extract features at the final pool layer.
                self.model = Model(
                    inputs=base_model.input,
                    outputs=base_model.get_layer('fc2').output
                )
                self.target_size = (224,224)
                self.preprocess_input = vgg19_preprocessor
            elif cnnModel == 'InceptionResNetV2':
                # Get model with pretrained weights.
                base_model = InceptionResNetV2(
                    weights='imagenet',
                    include_top=True
                )

                # We'll extract features at the final pool layer.
                self.model = Model(
                    inputs=base_model.input,
                    outputs=base_model.get_layer('avg_pool').output
                )
                self.target_size = (299,299)
                self.preprocess_input = inception_resnet_v2_preprocessor

            elif cnnModel == 'NASNetMobile':
                # Get model with pretrained weights.
                base_model = NASNetMobile(
                    weights='imagenet',
                    include_top=True
                )

                # We'll extract features at the final pool layer.
                self.model = Model(
                    inputs=base_model.input,
                    outputs=base_model.get_layer('global_average_pooling2d_1').output
                )
                self.target_size = (224,224)
                self.preprocess_input = nasnet_preprocessor
            elif cnnModel == 'NASNetMobile2':
                # Get model with pretrained weights.
                base_model = NASNetMobile(
                    weights='imagenet',
                    include_top=False,
                    pooling = 'avg'
                )

                base_model.add(Flatten())
                self.model = base_model

                self.target_size = (224,224)
                self.preprocess_input = nasnet_preprocessor



            elif cnnModel == 'NASNetLarge':
                # Get model with pretrained weights.
                base_model2 = NASNetLarge(
                    weights='imagenet',
                    include_top=True
                )

                # We'll extract features at the final pool layer.
                self.model = Model(
                    inputs=base_model.input,
                    outputs=base_model.get_layer('global_average_pooling2d_1').output
                )
                self.target_size = (331,331)
                self.preprocess_input = nasnet_preprocessor

            elif cnnModel == 'cifar10vgg':
                base_model = cifar10vgg(False)
                interModel = Model(
                    inputs=base_model.model.input,
                    #outputs=base_model.model.get_layer('flatten_1').output
                    outputs=base_model.model.get_layer('max_pooling2d_3').output

                    #outputs=base_model.model.get_layer('dropout_10').output
                    #outputs=base_model.model.get_layer('max_pooling2d_3').output
                )

                self.model = Sequential()
                self.model.add(interModel)
                self.model.add(Flatten())
                #self.model.compile()
                self.target_size = (32,32)
        else:

            # Load the model first.
            self.model = load_model(weights)

            # Then remove the top so we get features not predictions.
            # From: https://github.com/fchollet/keras/issues/2371
            self.model.layers.pop()
            self.model.layers.pop()  # two pops to get to pool layer
            self.model.outputs = [self.model.layers[-1].output]
            self.model.output_layers = [self.model.layers[-1]]
            self.model.layers[-1].outbound_nodes = []

    def normalize_production_here(self,x):
         #this function is used to normalize instances in production according to saved training set statistics
         # Input: X - a training set
         # Output X - a normalized training set according to normalization constants.

         #these values produced during first training and are general for the standard cifar10 training set normalization
         mean = 120.707
         std = 64.15
         return (x-mean)/(std+1e-7)

    def extract(self, image_path):

        if self.cnnModel == 'cifar10vgg':
            img = image.load_img(image_path)
            x = image.img_to_array(img)
            x = self.centeredCrop(x, 32, 32)
            x = np.expand_dims(x, axis=0)
            #x =  self.normalize_production_here(x)
        else:
            img = image.load_img(image_path, target_size=self.target_size)
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            x = self.preprocess_input(x)

        # Get the prediction.
        features = self.model.predict(x)

        if self.weights is None:
            # For imagenet/default network:
            features = features[0]
        else:
            # For loaded network:
            features = features[0]

        return features

    def centeredCrop(self, img, new_height, new_width):

        width =  np.size(img,1)
        height =  np.size(img,0)

        left = np.ceil((width - new_width)/2)
        top = np.ceil((height - new_height)/2)
        right = np.floor((width + new_width)/2)
        bottom = np.floor((height + new_height)/2)
        cImg = img[int(top):int(bottom), int(left):int(right),:]
        return cImg
