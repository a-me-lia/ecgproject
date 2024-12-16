import os
import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np

class ECGDataset(Dataset):
    def __init__(self, data_dir, metadata_csv, target_length=10000):
        """
        Args:
            data_dir (str): Path to the directory containing ECG data files.
            metadata_csv (str): Path to the CSV file containing metadata with labels.
            target_length (int): Desired length for all ECG data sections.
        """
        self.data_dir = data_dir
        self.metadata = pd.read_csv(metadata_csv)
        self.target_length = target_length

        # Ensure metadata contains the required columns
        if 'Filename' not in self.metadata.columns or 'Condition' not in self.metadata.columns:
            raise ValueError("Metadata CSV must contain 'Filename' and 'Condition' columns.")

        # Encode labels as integers
        self.label_encoder = {label: idx for idx, label in enumerate(self.metadata['Condition'].unique())}
        self.metadata['EncodedLabel'] = self.metadata['Condition'].map(self.label_encoder)

    def __len__(self):
        """Returns the total number of samples."""
        return len(self.metadata)

    def __getitem__(self, idx):
        """
        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            tuple: (ecg_section, label) where ecg_section is a torch.Tensor and label is an integer.
        """
        # Retrieve the file name and label from the metadata
        row = self.metadata.iloc[idx]
        file_path = row['Filename']
        label = row['EncodedLabel']

        # Load the ECG data file (assuming .dat format)
         
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File {file_path} does not exist.")

        # Load the data and process it to match the target length
        ecg_data = pd.read_csv(file_path, header=None).values.flatten()  # Flatten to a 1D array
        ecg_data = self._process_ecg_data(ecg_data)

        # Convert the processed data to a tensor
        ecg_section = torch.tensor(ecg_data, dtype=torch.float32)

        return ecg_section, label

    def _process_ecg_data(self, ecg_data):
        """
        Truncates or pads ECG data to match the target length.

        Args:
            ecg_data (np.ndarray): The raw ECG data.

        Returns:
            np.ndarray: Processed ECG data with the target length.
        """
        if len(ecg_data) > self.target_length:
            # Truncate if data is too long
            ecg_data = ecg_data[:self.target_length]
        elif len(ecg_data) < self.target_length:
            # Pad with zeros if data is too short
            padding = self.target_length - len(ecg_data)
            ecg_data = np.pad(ecg_data, (0, padding), mode='constant')
        return ecg_data
