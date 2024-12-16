import tensorflow as tf
from tensorflow.keras import layers, mode

def create_classification_model(input_channels, num_classes):
    model = models.Sequential([
        layers.Input(shape=(None, input_channels)),  # Input shape: (sequence_length, channels)
        layers.Conv1D(32, kernel_size=3, activation='relu'),
        layers.MaxPooling1D(pool_size=2),
        layers.Conv1D(64, kernel_size=3, activation='relu'),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(num_classes, activation='softmax')  # Output for classification
    ])
    return model
