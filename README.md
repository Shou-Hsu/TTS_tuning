# TTS 語音微調專案 (TTS Voice Fine-tuning Project)

這個存儲庫包含用於使用您自己的語音微調文字轉語音(TTS)模型的腳本。

## 📁 專案結構 (Project Structure)

```
TTS_finetune/
├── 📂 data/                           # 數據相關資料夾
│   ├── audio_with_transcript/         # 原始音頻和轉錄檔案
│   │   ├── JRE_2223_episode/          # Joe Rogan Experience 2223集音頻數據
│   │   └── JRE_2281_episode/          # Joe Rogan Experience 2281集音頻數據
│   └── dataset_merged/                # 合併後的訓練數據集
│       ├── train/                     # 訓練集
│       ├── validation/                # 驗證集
│       ├── test/                      # 測試集
│       └── dataset_dict.json          # 數據集配置檔案
├── 📂 models/                         # 模型相關資料夾
│   ├── base_models/                   # 基礎預訓練模型
│   │   └── orpheus-3b-0.1-ft/         # Orpheus 3B基礎模型
│   ├── finetuned_v1.0_baseline/       # 微調模型 v1.0 基準版本
│   ├── finetuned_v1.1_merged/         # 微調模型 v1.1 合併版本
│   └── finetuned_v2.0_latest/         # 微調模型 v2.0 最新版本
├── 📂 scripts/                        # 腳本檔案
│   ├── dataset_processor.py           # 數據處理腳本
│   ├── transcript_merger.py           # 轉錄合併腳本
│   ├── inference_pipeline.py          # 推理管線
│   └── utils.py                       # 工具函數
├── 📂 configs/                        # 配置檔案
│   └── training_config.yaml           # 訓練配置檔案
├── 📂 logs/                           # 日誌和實驗記錄
│   ├── training.log                   # 訓練日誌
│   ├── wandb/                         # Weights & Biases實驗記錄
│   └── sample_output_v2.0.wav         # 生成的音頻範例
├── 📂 docs/                           # 文檔資料夾
│   └── PROJECT_GUIDE.md               # 專案使用指南
├── requirements.txt                   # Python依賴關係
└── README.md                          # 專案說明檔案
```

## 🚀 快速開始 (Quick Start)

### 1. 安裝依賴 (Install Dependencies)
```bash
pip install -r requirements.txt
```

### 2. 準備數據 (Prepare Data)
```bash
python scripts/dataset_processor.py --input_dir data/audio_with_transcript --output_dir data/dataset_merged
python scripts/transcript_merger.py --input_dir data/audio_with_transcript --output_file data/merged_transcripts.json
```

### 3. 微調模型 (Fine-tune Model)
```bash
# 使用TMUX在背景執行訓練
tmux new -d -s train WANDB_API_KEY=<your_key> accelerate launch --config_file accelerate_config.yaml train.py
```

### 4. 生成語音 (Generate Speech)
```bash
python scripts/inference_pipeline.py \
    --config configs/training_config.yaml \
    --model_path models/finetuned_v2.0_latest/checkpoint-3500 \
    --text "您好，這是一個測試訊息" \
    --output_path logs/test_output.wav
```

## 📋 功能特色 (Features)

- ✅ 支援多語言TTS微調
- ✅ 整合Weights & Biases實驗追蹤
- ✅ 支援Accelerate分散式訓練
- ✅ 模組化的數據處理管線
- ✅ 簡化的推理介面

## 🛠️ 技術規格 (Technical Specifications)

- **基礎模型**: Orpheus 3B
- **訓練框架**: PyTorch + Transformers + Accelerate
- **音頻處理**: LibROSA + SoundFile
- **實驗追蹤**: Weights & Biases

## 📊 模型版本 (Model Versions)

| 版本 | 資料夾名稱 | 描述 | 推薦用途 |
|------|------------|------|----------|
| v1.0 | finetuned_v1.0_baseline | 基準版本 | 實驗測試 |
| v1.1 | finetuned_v1.1_merged | 合併版本 | 中等品質 |
| v2.0 | finetuned_v2.0_latest | 最新版本 | 生產使用 |

## 🔧 系統需求 (System Requirements)

- Python 3.8+
- PyTorch 2.0+
- CUDA 11.8+ (建議使用GPU)
- 至少16GB RAM
- 至少10GB可用磁碟空間

## 📝 注意事項 (Notes)

- 請確保音頻品質良好且背景噪音最小
- 建議至少使用10-20個音頻樣本進行微調
- 訓練時間視硬體配置而定
- 請確保已設定適當的WANDB API密鑰

## 🤝 貢獻 (Contributing)

歡迎提交問題報告和功能請求！

## 📚 詳細文檔 (Documentation)

完整的使用指南請參考：[專案使用指南](docs/PROJECT_GUIDE.md)

## 📄 授權 (License)

本專案採用MIT授權條款。 