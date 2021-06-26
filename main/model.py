"""
    Fine Tuning VGG-16 Pre-Trained Model
"""

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import VGG16
from tensorflow.keras.activations import relu, softmax
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.losses import categorical_crossentropy


# Instantiating the model with passing default parameters values
vgg_model = VGG16(weights="imagenet", include_top=False, input_shape=(224, 244, 3))

# Freezing the model layers except last block
for layer in vgg_model.layers[:15]:
    layer.trainable = False

# Adding flatten and dense layers to vgg-16 in order final model
output = vgg_model.output
output = Flatten()(output)
output = Dense(512, activation=relu)(output)
output = Dropout(0.5)(output)
output = Dense(8, activation=softmax)(output)
final_model = Model(inputs=vgg_model.inputs, outputs=output)
