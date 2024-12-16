import torch
import torch.nn as nn
import torch.nn.functional as F

class ClassificationModel(nn.Module):
    def __init__(self, input_channels=1, num_classes=2):
        """
        A simple classification model for ECG data using 1D convolutional layers.

        Args:
            input_channels (int): Number of input channels (e.g., 1 for single-channel ECG data).
            num_classes (int): Number of output classes for classification.
        """
        super(ClassificationModel, self).__init__()

        # 1D Convolutional layers
        self.conv1 = nn.Conv1d(input_channels, 16, kernel_size=5, stride=1, padding=2)  # Output: (16, L)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=5, stride=1, padding=2)  # Output: (32, L/2)
        self.conv3 = nn.Conv1d(32, 64, kernel_size=5, stride=1, padding=2)  # Output: (64, L/4)

        # Pooling layers
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)  # Halves the sequence length

        # Fully connected layers
        self.fc1 = nn.Linear(64 * 1250, 256)  # Assuming input length is 10,000
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        """
        Forward pass for the classification model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_channels, seq_length).

        Returns:
            torch.Tensor: Output logits of shape (batch_size, num_classes).
        """
        x = F.relu(self.conv1(x))  # Apply first convolution + activation
        x = self.pool(x)  # Apply pooling

        x = F.relu(self.conv2(x))  # Apply second convolution + activation
        x = self.pool(x)  # Apply pooling

        x = F.relu(self.conv3(x))  # Apply third convolution + activation
        x = self.pool(x)  # Apply pooling

        x = torch.flatten(x, start_dim=1)  # Flatten for fully connected layers

        x = F.relu(self.fc1(x))  # First fully connected layer + activation
        x = self.fc2(x)  # Output layer

        return x
