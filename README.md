# TTS Voice Fine-tuning Project

This repository provides a recommended project structure and guidelines for fine-tuning Text-to-Speech (TTS) models using your own voice data.

## 📁 Project Structure

```
TTS_finetune/
├── 📂 data/                           # Data-related folders
│   ├── raw_data/                      # Original audio and transcript files
│   │   ├── example1/                  # Example: audio and transcipt data
│   │   │   ├── audio/                 # Audio files for this episode
│   │   │   └── transcript/            # Transcript files for this episode
│   │   └── example2/                  # Example: audio and transcipt data
│   │       ├── audio/                 # Audio files for this episode
│   │       └── transcript/            # Transcript files for this episode
│   └── processed_data/                # Merged training dataset
│       ├── train/                     # Training set
│       ├── validation/                # Validation set
│       ├── test/                      # Test set
│       └── dataset_dict.json          # Dataset configuration file
├── 📂 configs/                        # Configuration files
│   └── training_config.yaml           # Training configuration file
├── 📂 logs/                           # Logs and experiment records
│   ├── training.log                   # Training logs
│   └── wandb/                         # Weights & Biases experiment records
├── requirements.txt                   # Python dependencies
└── README.md                          # Project documentation
```

## 🚀 Quick Start

**Note**: This repository provides the recommended structure. You need to implement your own scripts and organize your data according to this structure.

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Prepare Data
```bash
# Example commands (implement your own scripts):
python scripts/dataset_processor.py --input_dir data/raw_data --output_dir data/processed_data
```

### 3. Fine-tune Model
```bash
# Example training command (adapt to your setup):
python tts_pipeline.py --stage train --dataset_path ./data/processed_dataset
```

### 4. Generate Speech
```bash
# Example inference command (implement your own pipeline):
python tts_pipeline.py --stage inference --model_path models/finetuned_model--prompt "This is not just voice synthesis — it is signal extraction. We began with raw, noisy audio from real-world conversations, meetings, and phone calls, and fine-tuned it into a voice that is clear, responsive, and context-aware." --audio_output generated_audio.wav
```

## 📋 Features

- ✅ Multi-language TTS fine-tuning support
- ✅ Weights & Biases experiment tracking integration
- ✅ Accelerate distributed training support
- ✅ Modular data processing pipeline
- ✅ Simplified inference interface

## 🛠️ Technical Specifications

- **Base Model**: Orpheus 3B
- **Training Framework**: PyTorch + Transformers + Accelerate
- **Audio Processing**: LibROSA + SoundFile
- **Experiment Tracking**: Weights & Biases



## 🔧 System Requirements

- Python 3.8+
- PyTorch 2.0+
- CUDA 11.8+ (GPU recommended)
- At least 16GB RAM
- At least 10GB available disk space

## 📝 Notes

- This repository provides a template structure - you need to implement your own scripts
- Ensure audio quality is good with minimal background noise
- Recommend using at least 10-20 audio samples for fine-tuning
- Training time depends on hardware configuration
- Make sure to set appropriate WANDB API key
- Organize your data according to the suggested folder structure

## 🤝 Contributing

Issue reports and feature requests are welcome!

## 📚 Documentation

For complete usage guide, please refer to: [Project Usage Guide](docs/PROJECT_GUIDE.md)

## 📄 License

This project is licensed under the MIT License. 