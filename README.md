# TTS Voice Fine-tuning Project

This repository contains scripts for fine-tuning Text-to-Speech (TTS) models using your own voice data.

## 📁 Project Structure

```
TTS_finetune/
├── 📂 data/                           # Data-related folders
│   ├── audio_with_transcript/         # Original audio and transcript files
│   │   ├── JRE_2223_episode/          # Joe Rogan Experience Episode 2223 audio data
│   │   └── JRE_2281_episode/          # Joe Rogan Experience Episode 2281 audio data
│   └── dataset_merged/                # Merged training dataset
│       ├── train/                     # Training set
│       ├── validation/                # Validation set
│       ├── test/                      # Test set
│       └── dataset_dict.json          # Dataset configuration file
├── 📂 models/                         # Model-related folders
│   ├── base_models/                   # Base pre-trained models
│   │   └── orpheus-3b-0.1-ft/         # Orpheus 3B base model
│   ├── finetuned_v1.0_baseline/       # Fine-tuned model v1.0 baseline version
│   ├── finetuned_v1.1_merged/         # Fine-tuned model v1.1 merged version
│   └── finetuned_v2.0_latest/         # Fine-tuned model v2.0 latest version
├── 📂 scripts/                        # Script files
│   ├── dataset_processor.py           # Data processing script
│   ├── transcript_merger.py           # Transcript merging script
│   ├── inference_pipeline.py          # Inference pipeline
│   └── utils.py                       # Utility functions
├── 📂 configs/                        # Configuration files
│   └── training_config.yaml           # Training configuration file
├── 📂 logs/                           # Logs and experiment records
│   ├── training.log                   # Training logs
│   ├── wandb/                         # Weights & Biases experiment records
│   └── sample_output_v2.0.wav         # Generated audio sample
├── 📂 docs/                           # Documentation folder
│   └── PROJECT_GUIDE.md               # Project usage guide
├── requirements.txt                   # Python dependencies
└── README.md                          # Project documentation
```

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Prepare Data
```bash
python scripts/dataset_processor.py --input_dir data/audio_with_transcript --output_dir data/dataset_merged
python scripts/transcript_merger.py --input_dir data/audio_with_transcript --output_file data/merged_transcripts.json
```

### 3. Fine-tune Model
```bash
# Run training in background using TMUX
tmux new -d -s train WANDB_API_KEY=<your_key> accelerate launch --config_file accelerate_config.yaml train.py
```

### 4. Generate Speech
```bash
python scripts/inference_pipeline.py \
    --config configs/training_config.yaml \
    --model_path models/finetuned_v2.0_latest/checkpoint-3500 \
    --text "Hello, this is a test message" \
    --output_path logs/test_output.wav
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

## 📊 Model Versions

| Version | Folder Name | Description | Recommended Use |
|---------|-------------|-------------|-----------------|
| v1.0 | finetuned_v1.0_baseline | Baseline version | Experimental testing |
| v1.1 | finetuned_v1.1_merged | Merged version | Medium quality |
| v2.0 | finetuned_v2.0_latest | Latest version | Production use |

## 🔧 System Requirements

- Python 3.8+
- PyTorch 2.0+
- CUDA 11.8+ (GPU recommended)
- At least 16GB RAM
- At least 10GB available disk space

## 📝 Notes

- Ensure audio quality is good with minimal background noise
- Recommend using at least 10-20 audio samples for fine-tuning
- Training time depends on hardware configuration
- Make sure to set appropriate WANDB API key

## 🤝 Contributing

Issue reports and feature requests are welcome!

## 📚 Documentation

For complete usage guide, please refer to: [Project Usage Guide](docs/PROJECT_GUIDE.md)

## 📄 License

This project is licensed under the MIT License. 