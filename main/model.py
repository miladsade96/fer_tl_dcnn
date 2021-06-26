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
