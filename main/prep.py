"""
    Preparing Data For Training Fine-Tuned VGG-16 Model
"""

from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Defining batch size value
batch_size = 32

# Setting image data generator parameters
# Some of these are used for image augmentation
data_gen_args = dict(
    rotation_range=0.2,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    rescale=1./255,
    zoom_range=0.3,
    validation_split=0.3
)

# Instantiating image data generator
data_gen = ImageDataGenerator(**data_gen_args)

# Preparing data for training process
train_generator = data_gen.flow_from_directory(
    directory="Dataset",
    target_size=(224, 224),
    batch_size=batch_size,
    interpolation="lanczos",
    shuffle=True,
    subset="training"
)

# Preparing data for validation process
validation_generator = data_gen.flow_from_directory(
    directory="Dataset",
    target_size=(224, 224),
    batch_size=batch_size,
    interpolation="lanczos",
    shuffle=True,
    subset="validation"
)
