# TTS Fine-tuning Project Guide

## 📖 Project Overview

This project is a complete text-to-speech (TTS) model fine-tuning system based on the Orpheus 3B model, supporting custom voice data fine-tuning.

## 🗂️ Directory Structure

### 📂 `/data` - Data Management
Stores all training-related data files

- **`audio_with_transcript/`** - Raw audio and transcript files
  - `JRE_2223_episode/` - Joe Rogan Experience Episode 2223 audio data
  - `JRE_2281_episode/` - Joe Rogan Experience Episode 2281 audio data
  
- **`dataset_merged/`** - Merged training dataset
  - `train/` - Training set (80%)
  - `validation/` - Validation set (10%)  
  - `test/` - Test set (10%)
  - `dataset_dict.json` - Dataset metadata

### 📂 `/scripts` - Execution Scripts
Contains all data processing and inference scripts

- **`dataset_processor.py`** - Data processing script
  - Audio segmentation and format conversion
  - Transcript file processing
  - Dataset splitting
  
- **`transcript_merger.py`** - Transcript merging script
  - Multiple transcript file merging
  - Format standardization
  
- **`inference_pipeline.py`** - Inference pipeline
  - Text-to-speech generation
  - Model loading and inference
  
- **`utils.py`** - Utility functions
  - Audio processing tools
  - File operation helpers

### 📂 `/configs` - Configuration Management
Stores model and training configuration files

- **`training_config.yaml`** - Main configuration file
  - Model parameter settings
  - Training hyperparameters
  - Data path configuration

### 📂 `/models` - Model Storage
Manages all model files

- **`base_models/`** - Base pretrained models
  - `orpheus-3b-0.1-ft/` - Orpheus 3B base model
  
- **Fine-tuned Model Version Control:**
  - `finetuned_v1.0_baseline/` - v1.0 baseline version
  - `finetuned_v1.1_merged/` - v1.1 merged version
  - `finetuned_v2.0_latest/` - v2.0 latest version (recommended)

### 📂 `/logs` - Experiment Logging
Tracks training process and results

- **`training.log`** - Detailed training logs
- **`wandb/`** - Weights & Biases experiment tracking
- **`sample_output_v2.0.wav`** - Generated audio sample

## 🚀 Usage Workflow

### 1. Environment Setup