import torch
from tqdm import tqdm  # To display training progress in terminal
import torch.optim as optim
import pandas as pd
from sklearn.preprocessing import LabelEncoder  # For converting labels to integers

def get_num_classes_from_metadata(metadata_csv):
    """Extract the number of unique conditions (labels) from the metadata CSV."""
    try:
        # Load the CSV file into a DataFrame
        df = pd.read_csv(metadata_csv)

        # Extract the condition/label column (assuming the condition is in a column named 'Condition')
        conditions = df['Condition'].unique()

        # Return the number of unique conditions (classes)
        return len(conditions)
    
    except Exception as e:
        print(f"Error reading metadata CSV: {e}")
        return None

# Function for training the model
def train_model(model, train_loader, optimizer, criterion, epochs, device='cuda'):
    model.to(device)
    model.train()  # Set model to training mode

    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        
        with tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", ncols=100) as tbar:
            for ecg_section, label in tbar:
                # Ensure that ecg_section is a tensor
                if not isinstance(ecg_section, torch.Tensor):
                    ecg_section = torch.tensor(ecg_section, dtype=torch.float32)  # Convert to tensor if necessary
                ecg_section = ecg_section.unsqueeze(1).to(device)  # Add channel dimension and move to device

                # Map labels to integers if they are strings (ensure label is a tensor)
                if isinstance(label, str):
                    label = label_encoder.transform([label])  # Convert string to integer label
                    label = torch.tensor(label, dtype=torch.long)  # Ensure the label is a LongTensor for classification
                
                label = label.to(device)

                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward pass
                output = model(ecg_section)

                # Calculate loss
                loss = criterion(output, label)
                
                # Backward pass and optimization
                loss.backward()
                optimizer.step()

                # Update loss and accuracy
                running_loss += loss.item()
                _, predicted = torch.max(output, 1)
                correct += (predicted == label).sum().item()
                total += label.size(0)

                # Update progress bar
                tbar.set_postfix(loss=running_loss/len(train_loader), accuracy=100 * correct/total)

        print(f"Epoch {epoch+1} - Loss: {running_loss/len(train_loader):.4f}, Accuracy: {100*correct/total:.2f}%")

# Function to run inference on the model
def run_inference(model, inputs, device='cuda'):
    model.to(device)
    model.eval()  # Set model to evaluation mode
    
    # Check if inputs is a tensor
    if not isinstance(inputs, torch.Tensor):
        inputs = torch.tensor(inputs, dtype=torch.float32)  # Convert to tensor if necessary

    inputs = inputs.unsqueeze(1).to(device)  # Add channel dimension and move to device

    with torch.no_grad():  # Disable gradient calculation during inference
        output = model(inputs)
        _, predicted = torch.max(output, 1)

    return predicted.item()

# Assuming you have loaded your metadata CSV
metadata_csv = "datap/metadata.csv"  # Update with your actual path

# Create a label encoder to map string labels to integers
label_encoder = LabelEncoder()

# Load the metadata CSV
df = pd.read_csv(metadata_csv)
# Fit the label encoder on the condition column
label_encoder.fit(df['Condition'])

# Now, label_encoder can be used to transform string labels to integer labels in the dataset
