import argparse
import os
import json
import time
import wave
import logging
import yaml
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

import torch
import soundfile as sf
from datasets import Dataset, Features, Audio, Value, DatasetDict
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model
from orpheus_tts import OrpheusModel
from snac import SNAC

from utils import (
    tokenise_audio, remove_duplicate_frames, create_training_example,
    custom_data_collator, compute_metrics, evaluate_model
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TTSPipeline:
    def __init__(self, config_path="configs/training_config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
        self.base_model_name = self.config['model']['base_model']
        self.target_sr = self.config['model']['target_sample_rate']
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")
    
    def preprocess_data(self, audio_dir, output_dir, transcript_file=None):
        """Step 1: Preprocess audio and text data (supports merging multiple folders)"""
        logger.info("Starting data preprocessing...")
        audio_dir = Path(audio_dir)
        processed_data = {split: [] for split in ['train', 'validation', 'test']}

        # If there are multiple folders with audio_chunks and transcripts, use merge mode
        processed_dirs = [d for d in audio_dir.iterdir() if d.is_dir() and 
                         (d / "audio").exists() and (d / "transcripts").exists()]
        if processed_dirs:
            logger.info(f"Detected {len(processed_dirs)} subdirectories, entering merge mode")
            all_audio = []
            all_transcripts = {}
            for dataset_dir in processed_dirs:
                logger.info(f"Processing: {dataset_dir}")
                transcript_path = dataset_dir / "transcripts" / "all_transcripts.json"
                if not transcript_path.exists():
                    logger.warning(f"{transcript_path} does not exist, skipping")
                    continue
                with open(transcript_path, 'r', encoding='utf-8') as f:
                    transcript_data = json.load(f)
                if 'transcriptions' in transcript_data:
                    for item in transcript_data['transcriptions']:
                        if isinstance(item, dict) and 'filename' in item and 'text' in item:
                            all_transcripts[item['filename']] = item['text']
                else:
                    logger.warning(f"{transcript_path} missing 'transcriptions' field, skipping")
                    continue
                audio_dir_path = dataset_dir / "audio"
                if not audio_dir_path.exists():
                    logger.warning(f"{audio_dir_path} does not exist, skipping")
                    continue
                for audio_file in audio_dir_path.rglob("*.wav"):
                    all_audio.append(str(audio_file))
            if not all_audio:
                raise ValueError("No audio files found!")
            logger.info(f"Found a total of {len(all_audio)} audio files and {len(all_transcripts)} transcripts")
            # Randomly split the data
            import random
            random.shuffle(all_audio)
            n = len(all_audio)
            n_train = int(n * 0.8)
            n_val = int(n * 0.1)
            train_files = all_audio[:n_train]
            val_files = all_audio[n_train:n_train+n_val]
            test_files = all_audio[n_train+n_val:]
            splits = {'train': train_files, 'validation': val_files, 'test': test_files}
            for split, files in splits.items():
                for audio_path in files:
                    fname = os.path.basename(audio_path)
                    if fname not in all_transcripts:
                        logger.warning(f"{fname} has no corresponding transcript, skipping")
                        continue
                    # Load audio file
                    try:
                        audio, sr = sf.read(audio_path)
                        if len(audio.shape) > 1:
                            audio = audio.mean(axis=1)
                        if sr != self.target_sr:
                            from scipy import signal
                            audio = signal.resample(audio, int(len(audio) * self.target_sr / sr))
                        processed_data[split].append({
                            'text': all_transcripts[fname],
                            'audio': {'array': audio, 'sampling_rate': self.target_sr}
                        })
                    except Exception as e:
                        logger.error(f"Failed to process {audio_path}: {e}")
            logger.info(f"Split results: train={len(processed_data['train'])}, validation={len(processed_data['validation'])}, test={len(processed_data['test'])}")
        else:
            # Single folder mode
            if transcript_file is None:
                raise ValueError("transcript_file is required in single folder mode")
            # Load transcripts
            with open(transcript_file, 'r', encoding='utf-8') as f:
                transcripts = json.load(f)
            # Convert to dict if needed
            if isinstance(transcripts, list):
                transcript_dict = {item['filename']: item['text'] for item in transcripts}
            elif 'transcriptions' in transcripts:
                transcript_dict = {item['filename']: item['text'] for item in transcripts['transcriptions']}
            else:
                transcript_dict = transcripts
            logger.info(f"Loaded {len(transcript_dict)} transcripts")
            # Process audio files
            for split in processed_data.keys():
                split_dir = audio_dir / split
                if not split_dir.exists():
                    logger.warning(f"Split directory {split} not found")
                    continue
                for audio_file in split_dir.rglob("*.wav"):
                    filename = audio_file.name
                    if filename not in transcript_dict:
                        logger.warning(f"No transcript for {filename}")
                        continue
                    try:
                        audio, sr = sf.read(audio_file)
                        if len(audio.shape) > 1:
                            audio = audio.mean(axis=1)
                        if sr != self.target_sr:
                            from scipy import signal
                            audio = signal.resample(audio, int(len(audio) * self.target_sr / sr))
                        processed_data[split].append({
                            'text': transcript_dict[filename],
                            'audio': {'array': audio, 'sampling_rate': self.target_sr}
                        })
                    except Exception as e:
                        logger.error(f"Error processing {filename}: {e}")
        if not any(processed_data.values()):
            raise ValueError("No valid audio-text pairs found!")
        # Create dataset
        features = Features({
            'text': Value('string'),
            'audio': Audio(sampling_rate=self.target_sr)
        })
        split_dataset = DatasetDict({
            split: Dataset.from_list(data, features=features)
            for split, data in processed_data.items()
            if data
        })
        # Save dataset
        os.makedirs(output_dir, exist_ok=True)
        split_dataset.save_to_disk(output_dir)
        logger.info(f"Dataset saved to {output_dir}")
        for split, dataset in split_dataset.items():
            logger.info(f"{split.capitalize()}: {len(dataset)} samples")
        return split_dataset
    
    def prepare_training_data(self, dataset):
        """Prepare data for training"""
        logger.info("Preparing training data...")
        
        # Load SNAC model
        snac_model = SNAC.from_pretrained(self.config['snac']['model_name']).eval()
        if torch.cuda.is_available():
            snac_model = snac_model.cuda()
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(self.base_model_name)
        
        processed_datasets = {split: [] for split in ['train', 'validation', 'test']}
        
        # Process each split        
        total_examples = sum(len(dataset.get(split, [])) for split in processed_datasets.keys())
        processed_count = 0
        start_time = datetime.now()
        
        for split in processed_datasets.keys():
            if split not in dataset:
                logger.warning(f"Split {split} not found")
                continue
                
            logger.info(f"Processing {split} split...")
            split_examples = dataset[split]
            
            with tqdm(total=len(split_examples), 
                     desc=f"Processing {split} split",
                     unit="samples") as pbar:
                
                for i, example in enumerate(split_examples):
                    try:
                        text = example['text']
                        audio_array = example['audio']['array']
                        sample_rate = example['audio']['sampling_rate']
                        
                        audio_codes = tokenise_audio(audio_array, snac_model, sample_rate)
                        audio_codes = remove_duplicate_frames(audio_codes)
                        
                        processed_example = create_training_example(text, audio_codes, tokenizer)
                        
                        if processed_example is not None:
                            processed_datasets[split].append(processed_example)
                        
                        processed_count += 1
                        elapsed_time = (datetime.now() - start_time).total_seconds()
                        avg_time = elapsed_time / processed_count
                        remaining_examples = total_examples - processed_count
                        eta = remaining_examples * avg_time
                        
                        pbar.set_postfix({
                            'Success Rate': f"{len(processed_datasets[split])}/{i+1}",
                            'Avg Time': f"{avg_time:.2f}s/sample",
                            'ETA': f"{eta/60:.1f}min"
                        })
                        pbar.update(1)
                        
                    except Exception as e:
                        logger.error(f"Error processing sample {i} in {split} split: {str(e)}")
                        pbar.update(1)
            
            success_rate = len(processed_datasets[split]) / len(split_examples) * 100
            logger.info(f"Completed {split} split - Processed {len(processed_datasets[split])}/{len(split_examples)} samples ({success_rate:.1f}%)")
        
        total_time = (datetime.now() - start_time).total_seconds()
        logger.info(f"Data processing complete! Processed {processed_count} samples in {total_time/60:.1f} minutes")
        return processed_datasets, tokenizer



    def train_model(self, dataset_path, output_dir=None):
        """Step 2: Fine-tune the model with LoRA"""
        logger.info("Starting model training...")
        
        if output_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = f"models/{self.config['output']['model_dir']}_{timestamp}"
        else:
            # Ensure output_dir is within models directory
            if not output_dir.startswith("models/"):
                output_dir = f"models/{output_dir}"
        
        # Ensure models directory exists
        os.makedirs("models", exist_ok=True)
        
        # Load dataset
        dataset = DatasetDict.load_from_disk(dataset_path)
        
        # Prepare all datasets
        processed_datasets, tokenizer = self.prepare_training_data(dataset)
        
        if not processed_datasets['train']:
            raise ValueError("No training examples were created!")
        
        # Load base model
        logger.info("Loading base model...")
        model = AutoModelForCausalLM.from_pretrained(
            self.base_model_name,
            torch_dtype=getattr(torch, self.config['model']['dtype']),
            device_map=self.config['model']['device_map'],
            trust_remote_code=self.config['model']['trust_remote_code']
        )
        
        # Setup LoRA
        lora_config = LoraConfig(
            r=self.config['training']['lora_r'],
            lora_alpha=self.config['training']['lora_alpha'],
            target_modules=self.config['training']['target_modules'],
            lora_dropout=self.config['training']['lora_dropout'],
            bias=self.config['training']['bias'],
            task_type=self.config['training']['task_type']
        )
        
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=self.config['training']['epochs'],
            per_device_train_batch_size=self.config['training']['batch_size'],
            per_device_eval_batch_size=self.config['training']['batch_size'],
            gradient_accumulation_steps=self.config['training']['gradient_accumulation_steps'],
            warmup_steps=self.config['training']['warmup_steps'],
            learning_rate=self.config['training']['learning_rate'],
            fp16=self.config['training']['fp16'],
            logging_steps=self.config['training']['logging_steps'],
            save_steps=self.config['training']['save_steps'],
            eval_steps=self.config['training']['eval_steps'],
            eval_strategy="steps" if processed_datasets['validation'] else "no",
            save_total_limit=self.config['training']['save_total_limit'],
            remove_unused_columns=self.config['training']['remove_unused_columns'],
            dataloader_pin_memory=self.config['training']['dataloader_pin_memory'],
            load_best_model_at_end=bool(processed_datasets['validation']),
            metric_for_best_model="eval_loss" if processed_datasets['validation'] else None,
            greater_is_better=False if processed_datasets['validation'] else None,
            max_grad_norm=self.config['training']['max_grad_norm'],
        )
        
        # Create trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=Dataset.from_list(processed_datasets['train']),
            eval_dataset=Dataset.from_list(processed_datasets['validation']) if processed_datasets['validation'] else None,
            tokenizer=tokenizer,
            data_collator=lambda features: custom_data_collator(features, tokenizer),
            compute_metrics=compute_metrics if processed_datasets['validation'] else None,
        )
        
        # Train
        logger.info("Starting training...")
        train_result = trainer.train()
        
        # Save training metrics
        metrics = train_result.metrics
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        
        # Clear GPU memory before evaluation
        torch.cuda.empty_cache()
        
        # Evaluate on validation and test sets
        evaluation_results = {'train': metrics}
        
        if processed_datasets['validation']:
            # Use smaller batch size for evaluation to prevent OOM
            eval_batch_size = max(1, self.config['training']['batch_size'] // 4)
            validation_metrics = evaluate_model(model, processed_datasets['validation'], tokenizer, "validation", eval_batch_size)
            trainer.log_metrics("validation", validation_metrics)
            trainer.save_metrics("validation", validation_metrics)
            evaluation_results['validation'] = validation_metrics
        
        if processed_datasets['test']:
            # Use smaller batch size for evaluation to prevent OOM
            eval_batch_size = max(1, self.config['training']['batch_size'] // 4)
            test_metrics = evaluate_model(model, processed_datasets['test'], tokenizer, "test", eval_batch_size)
            trainer.log_metrics("test", test_metrics)
            trainer.save_metrics("test", test_metrics)
            evaluation_results['test'] = test_metrics
        
        # Save model
        trainer.save_model()
        tokenizer.save_pretrained(output_dir)
        
        # Merge and save the final model
        logger.info("Merging LoRA adapters...")
        merged_model = model.merge_and_unload()
        merged_output_dir = f"{output_dir}_merged"
        merged_model.save_pretrained(merged_output_dir)
        tokenizer.save_pretrained(merged_output_dir)
        
        # Save evaluation results
        with open(os.path.join(output_dir, 'evaluation_results.json'), 'w') as f:
            json.dump(evaluation_results, f, indent=2)
        
        logger.info(f"Training completed! Model saved to {output_dir}")
        logger.info(f"Merged model saved to {merged_output_dir}")
        logger.info("Final evaluation results:")
        for split, metrics in evaluation_results.items():
            if metrics:
                logger.info(f"{split.capitalize()}: {metrics}")
        
        return merged_output_dir
    
    def test_inference(self, model_path, prompt, output_file=None):
        """Step 3: Test model inference"""
        logger.info("Testing model inference...")
        
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"generated_audio_{timestamp}.wav"
        
        # Load model
        logger.info(f"Loading model from {model_path}")
        model = OrpheusModel(
            model_name=model_path,
            dtype=getattr(torch, self.config['model']['dtype'])
        )
        
        # Generate speech
        logger.info(f"Generating speech for: {prompt[:100]}...")
        start_time = time.time()
        
        syn_tokens = model.generate_speech(
            prompt=prompt,
            temperature=self.config['generation']['temperature'],
            top_p=self.config['generation']['top_p'],
            repetition_penalty=self.config['generation']['repetition_penalty'],
            max_tokens=self.config['generation']['max_tokens']
        )
        
        # Save audio
        with wave.open(output_file, "wb") as wf:
            wf.setnchannels(self.config['audio']['channels'])
            wf.setsampwidth(self.config['audio']['sample_width'])
            wf.setframerate(self.config['audio']['sample_rate'])
            
            total_frames = 0
            for audio_chunk in syn_tokens:
                frame_count = len(audio_chunk) // (wf.getsampwidth() * wf.getnchannels())
                total_frames += frame_count
                wf.writeframes(audio_chunk)
            
            duration = total_frames / wf.getframerate()
        
        generation_time = time.time() - start_time
        
        logger.info(f"Generated {duration:.2f}s audio in {generation_time:.2f}s")
        logger.info(f"Saved to: {output_file}")
        logger.info(f"Speed: {duration/generation_time:.2f}x realtime")
        
        return output_file, duration, generation_time

def main():
    parser = argparse.ArgumentParser(description="Complete TTS Pipeline: Preprocessing, Training, and Inference")
    parser.add_argument("--stage", choices=["preprocess", "train", "inference"], 
                       required=True, help="Pipeline stage to run")
    
    # Preprocessing arguments
    parser.add_argument("--audio_dir", help="Directory containing audio files")
    parser.add_argument("--transcript_file", help="JSON file with transcripts")
    parser.add_argument("--dataset_output", default="processed_dataset", 
                       help="Output directory for processed dataset")
    
    # Training arguments
    parser.add_argument("--dataset_path", help="Path to processed dataset")
    parser.add_argument("--model_output", help="Output directory for trained model")
    
    # Inference arguments
    parser.add_argument("--model_path", help="Path to trained model")
    parser.add_argument("--prompt", help="Text prompt for speech generation")
    parser.add_argument("--audio_output", help="Output audio file")
    
    args = parser.parse_args()
    
    pipeline = TTSPipeline()
    
    if args.stage == "preprocess":
        if not args.audio_dir:
            print("Error: --audio_dir required for preprocessing")
            return
        # Merge mode: transcript_file can be None
        pipeline.preprocess_data(args.audio_dir, args.dataset_output, args.transcript_file)
    
    elif args.stage == "train":
        if not args.dataset_path:
            print("Error: --dataset_path required for training")
            return
        pipeline.train_model(args.dataset_path, args.model_output)
    
    elif args.stage == "inference":
        if not args.model_path or not args.prompt:
            print("Error: --model_path and --prompt required for inference")
            return
        pipeline.test_inference(args.model_path, args.prompt, args.audio_output)

if __name__ == "__main__":
    main() 