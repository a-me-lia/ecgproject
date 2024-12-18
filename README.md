# ECG Data Processing Pipeline

A Python-based ECG (Electrocardiogram) data processing pipeline designed for CUDA-enabled systems. This tool processes ECG waveform data from .hea and .dat files, with functionality for R-peak detection, cycle segmentation, and data visualization.

## System Requirements

- CUDA-capable GPU
- NVIDIA CUDA Toolkit
- PyTorch with CUDA support
- Python 3.6+

## Dependencies


torch
torchvision
torchaudio
numpy
scikit-learn
matplotlib
scipy
wfdb


## Key Features

- Parses ECG header (.hea) files for metadata extraction
- Loads and processes waveform (.dat) files
- Implements R-peak detection using bandpass filtering
- Segments ECG data into 16-cycle sections
- Provides visualization tools for ECG waveforms
- Supports batch processing of multiple patient records

## Limitations

- **CUDA Dependency**: This implementation is specifically designed for CUDA-enabled systems and may not run on CPU-only environments
- **GPU Memory**: Large datasets may require significant GPU memory
- **Lead Support**: Currently optimized for leads I, II, and III only
- **File Format**: Only supports specific .hea and .dat file formats
