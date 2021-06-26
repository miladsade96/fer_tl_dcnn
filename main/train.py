"""
    Training The Fine-Tuned VGG-16 Model
"""

from math import ceil
from model import final_model
from prep import train_generator, validation_generator, batch_size

# Number of epochs
n_epochs = 30

# Train Section
history = final_model.fit_generator(
    train_generator,
    steps_per_epoch=ceil(train_generator.samples // batch_size),
    epochs=n_epochs,
    validation_data=validation_generator,
    validation_steps=ceil(validation_generator.samples // batch_size),
    verbose=1,
    workers=4
)
