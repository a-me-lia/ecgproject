import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

def parse_header(file_path):
    """Parse the .hea file to extract relevant metadata and leads."""
    metadata = {}
    leads = {}

    with open(file_path, 'r') as file:
        lines = file.readlines()

        # Extract metadata
        for line in lines:
            if line.startswith('#'):
                if ':' in line:
                    key, value = line[1:].split(':', 1)
                    metadata[key.strip()] = value.strip()
            elif line.strip():
                parts = line.split()
                if len(parts) > 8 and parts[8] in ["i", "ii", "iii"]:
                    leads[parts[8]] = int(parts[2])  # Sampling frequency

    return metadata, leads

def load_waveform(data_file, lead_indices):
    """Load waveform data from the .dat file for the specified leads."""
    data = np.fromfile(data_file, dtype=np.int16)
    num_leads = len(lead_indices)
    samples = data.reshape(-1, num_leads).T

    # Extract only the leads of interest
    lead_data = {lead: samples[idx] for lead, idx in lead_indices.items()}
    return lead_data


def process_patient_data(patient_folder):
    """Process one patient's data: parse metadata and load waveform."""
    try:
        # Find .hea and .dat files
        hea_files = [f for f in os.listdir(patient_folder) if f.endswith('.hea')]
        dat_files = [f for f in os.listdir(patient_folder) if f.endswith('.dat')]

        if len(hea_files) == 0 or len(dat_files) == 0:
            raise FileNotFoundError(f"Missing .hea or .dat files in {patient_folder}")

        hea_file = hea_files[0]
        dat_file = dat_files[0]

        hea_path = os.path.join(patient_folder, hea_file)
        dat_path = os.path.join(patient_folder, dat_file)

        # Parse the header for metadata and leads information
        metadata, leads = parse_header(hea_path)

        # Load the waveform data based on the leads
        waveform_data = load_waveform(dat_path, {k: i for i, k in enumerate(leads.keys())})

        # Extract condition or set to "Unknown" if not available
        condition = metadata.get("Reason for admission", "Unknown")

        return waveform_data, condition

    except Exception as e:
        print(f"Error processing patient folder {patient_folder}: {e}")
        return None, None

def visualize_patient_data(waveform_data, condition):
    """Visualize a patient's waveform and print the condition."""
    plt.figure(figsize=(12, 6))

    for i, (lead, data) in enumerate(waveform_data.items(), 1):
        plt.subplot(len(waveform_data), 1, i)
        plt.plot(data[:1000])  # Plot first 1000 samples
        plt.title(f"Lead {lead}")
        plt.xlabel("Time (ms)")
        plt.ylabel("Amplitude")

    plt.tight_layout()
    plt.suptitle(f"Condition: {condition}", fontsize=16)
    plt.subplots_adjust(top=0.88)
    plt.show()

import csv

def preprocess_all_data(patient_dir, output_dir):
    """Preprocess all patient data in the given directory and save to the output directory."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    metadata_file = os.path.join(output_dir, 'metadata.csv')
    
    # Open the CSV file for writing metadata
    with open(metadata_file, 'w', newline='') as csvfile:
        fieldnames = ['Filename', 'Condition']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        # Iterate through all patient folders
        for patient_folder in os.listdir(patient_dir):
            patient_folder_path = os.path.join(patient_dir, patient_folder)
            if os.path.isdir(patient_folder_path):
                try:
                    # Use process_patient_data to load waveform data and condition
                    waveform_data, condition = process_patient_data(patient_folder_path)

                    if waveform_data is None:
                        continue  # Skip folders that don't have valid data

                    # Split the waveform into 16-cycle segments and save the data
                    part_number = 1
                    patient_filename_base = f"{patient_folder}_"
                    for lead, data in waveform_data.items():
                        # Split data into sections of 16 heart cycles
                        sections = split_into_cycles(data)

                        for i, section in enumerate(sections):
                            # Create a filename for the part
                            output_filename = os.path.join(output_dir, f"{patient_filename_base}{part_number}.dat")
                            # Save the section data
                            save_waveform_section(output_filename, section)

                            # Write metadata to CSV
                            writer.writerow({'Filename': output_filename, 'Condition': condition})

                            part_number += 1
                            if part_number > 4:
                                break

                except Exception as e:
                    print(f"Error processing {patient_folder}: {e}")


import numpy as np
from scipy.signal import find_peaks, butter, filtfilt

def butter_bandpass(lowcut, highcut, fs, order=4):
    """Design a Butterworth bandpass filter."""
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return b, a

def bandpass_filter(data, lowcut, highcut, fs):
    """Apply a bandpass filter to the ECG data."""
    b, a = butter_bandpass(lowcut, highcut, fs)
    return filtfilt(b, a, data)

def detect_r_peaks(ecg_data, fs):
    """Detect R-peaks in an ECG signal."""
    # Bandpass filter to remove noise and isolate heart rate
    filtered_ecg = bandpass_filter(ecg_data, 0.5, 50, fs)

    # Differentiate the signal
    diff_ecg = np.diff(filtered_ecg)

    # Square the signal
    squared_ecg = diff_ecg ** 2

    # Integrate the signal (moving window)
    window_size = int(0.150 * fs)  # 150 ms window for QRS detection
    integrated_ecg = np.convolve(squared_ecg, np.ones(window_size)/window_size, mode='same')

    # Find peaks in the integrated signal
    r_peaks, _ = find_peaks(integrated_ecg, distance=int(0.6 * fs))  # Minimum distance between R-peaks (600 ms)

    return r_peaks

def split_into_cycles(waveform_data, fs=1000):
    """Split waveform data into sections of 16 heart cycles using R-peaks."""
    # Detect R-peaks in the waveform data
    r_peaks = detect_r_peaks(waveform_data, fs)

    # Ensure that we have enough R-peaks to create full 16-cycle sections
    num_cycles = len(r_peaks) - 1  # One less than the number of R-peaks
    sections = []

    # Split the waveform into 16-cycle sections based on R-peaks
    cycle_length = 16  # The number of cycles to include in each section
    for i in range(0, num_cycles - cycle_length, cycle_length):
        # Use R-peaks to get data for each cycle
        start_idx = r_peaks[i]
        end_idx = r_peaks[i + cycle_length]

        # Extract the section between these R-peaks
        section = waveform_data[start_idx:end_idx]
        sections.append(section)

    return sections


def save_waveform_section(filename, section):
    """Save a single section of waveform data to a file."""
    # Assuming section is a list/array of waveform samples
    np.savetxt(filename, section)

# Example Usage
if __name__ == "__main__":
    data_folder = "path_to_data_folder"

    # Preprocess all data
    all_data, all_labels, label_map = preprocess_all_data(data_folder)

    # Visualize one patient
    patient_folder = os.path.join(data_folder, "patient001")
    waveform_data, condition = process_patient_data(patient_folder)
    visualize_patient_data(waveform_data, condition)

    print(f"Data shape: {all_data.shape}, Labels shape: {all_labels.shape}")
    print(f"Label Map: {label_map}")
