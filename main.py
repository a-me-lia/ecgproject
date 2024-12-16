import os
import torch
from torch.utils.data import DataLoader
from models.classification_model import ClassificationModel
from preprocess import preprocess_all_data  # Assuming this function preprocesses data
from utils import train_model, run_inference
import pandas as pd
from ECGdataset import ECGDataset

def main():
    print("\nWelcome to the ECG Analysis CLI!")
    while True:
        print("\nOptions:")
        print("1. Preprocess Data")
        print("2. Train Classification Model")
        print("3. Run Inference")
        print("4. Exit")

        choice = input("Enter your choice: ")

        if choice == "1":
            data_dir = input("Enter path to dataset: ")
            output_dir = input("Enter path to save processed data: ")
            try:
                preprocess_all_data(data_dir, output_dir)
            except Exception as e:
                print(f"Error preprocessing data: {e}")

        elif choice == "2":
            train_dir = input("Enter path to preprocessed data directory: ")

            metadata_csv = os.path.join(train_dir, 'metadata.csv')  # Path to your metadata.csv file
            dataset = ECGDataset(train_dir, metadata_csv)
            train_loader = DataLoader(dataset, batch_size=32, shuffle=True)  # Shuffle data during training

            # Automatically determine the number of classes from metadata
            metadata = pd.read_csv(metadata_csv)
            num_classes = len(metadata['Condition'].unique())  # Get the number of unique classes

            # Assuming the number of input channels is fixed
            input_channels = 1  # Adjust based on your data format

            model = ClassificationModel(input_channels, num_classes).cuda() if torch.cuda.is_available() else ClassificationModel(input_channels, num_classes)
            optimizer = torch.optim.Adam(model.parameters())
            criterion = torch.nn.CrossEntropyLoss()
            epochs = int(input("Enter number of epochs: "))

            try:
                train_model(model, train_loader, optimizer, criterion, epochs)
            except Exception as e:
                print(f"Error training classification model: {e}")

        elif choice == "3":
            model_path = input("Enter path to trained model file: ")
            model = torch.load(model_path)
            model.eval()

            inputs_path = input("Enter path to input .dat file: ")
            inputs = torch.tensor(pd.read_csv(inputs_path, header=None).values, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # Adjust for batch and channel dimensions

            try:
                result = run_inference(model, inputs)
                print(f"Inference result: {result}")
            except Exception as e:
                print(f"Error during inference: {e}")

        elif choice == "4":
            print("Exiting.")
            break

        else:
            print("Invalid option.")

if __name__ == "__main__":
    main()
