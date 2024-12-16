import numpy as np
import matplotlib.pyplot as plt
from tkinter import Tk, filedialog

def choose_file():
    """Open a file dialog to choose a .dat file."""
    root = Tk()
    root.withdraw()  # Hide the root window
    file_path = filedialog.askopenfilename(
        title="Select a .dat file", 
        filetypes=[("DAT Files", "*.dat")]
    )
    return file_path

def load_waveform(file_path):
    """Load waveform data from the selected .dat file."""
    try:
        # Read the .dat file as a binary file (common for ECG signals)
        data = np.fromfile(file_path, dtype=np.int16)  # Adjust dtype based on data format
        return data
    except Exception as e:
        print(f"Error loading .dat file: {e}")
        return None

def visualize_waveform(waveform_data):
    """Visualize the waveform using Matplotlib."""
    plt.figure(figsize=(12, 6))
    
    # Plot the first 1000 samples (or adjust as needed)
    plt.plot(waveform_data[:1000])
    plt.title("ECG Waveform (First 1000 samples)")
    plt.xlabel("Samples")
    plt.ylabel("Amplitude")
    plt.show()

def main():
    # Prompt the user to select a .dat file
    dat_file = choose_file()

    if not dat_file:
        print("No file selected.")
        return
    
    # Load the waveform data from the selected file
    waveform_data = load_waveform(dat_file)

    if waveform_data is None:
        print("Failed to load the data.")
        return
    
    # Visualize the waveform
    visualize_waveform(waveform_data)

if __name__ == "__visualize__":
    main()
