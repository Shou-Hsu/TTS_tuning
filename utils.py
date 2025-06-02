"""
Shared audio processing utilities for TTS training.
Contains common functions used by both train.py and train_elon_specific.py.
"""

import torch
import torchaudio.transforms as T
import logging
from datasets import Dataset
from transformers import Trainer, TrainingArguments
import os
import shutil

logger = logging.getLogger(__name__)

# Define special tokens
tokeniser_length = 128256
start_of_text = 128000
end_of_text = 128009

start_of_speech = tokeniser_length + 1
end_of_speech = tokeniser_length + 2

start_of_human = tokeniser_length + 3
end_of_human = tokeniser_length + 4

start_of_ai = tokeniser_length + 5
end_of_ai = tokeniser_length + 6
pad_token_id = tokeniser_length + 7

audio_tokens_start = tokeniser_length + 10


def tokenise_audio(waveform, model, ds_sample_rate=22050):
    """Tokenize audio using SNAC model"""
    waveform = torch.from_numpy(waveform).unsqueeze(0)
    waveform = waveform.to(dtype=torch.float32)
    resample_transform = T.Resample(orig_freq=ds_sample_rate, new_freq=24000)
    waveform = resample_transform(waveform)

    waveform = waveform.unsqueeze(0).to("cuda")

    # Generate the codes from snac
    with torch.inference_mode():
        codes = model.encode(waveform)

    all_codes = []
    for i in range(codes[0].shape[1]):
        all_codes.append(codes[0][0][i].item() + 128266)
        all_codes.append(codes[1][0][2*i].item() + 128266 + 4096)
        all_codes.append(codes[2][0][4*i].item() + 128266 + (2*4096))
        all_codes.append(codes[2][0][(4*i)+1].item() + 128266 + (3*4096))
        all_codes.append(codes[1][0][(2*i)+1].item() + 128266 + (4*4096))
        all_codes.append(codes[2][0][(4*i)+2].item() + 128266 + (5*4096))
        all_codes.append(codes[2][0][(4*i)+3].item() + 128266 + (6*4096))

    return all_codes


def remove_duplicate_frames(codes_list):
    """Remove duplicate frames from audio codes"""
    if len(codes_list) % 7 != 0:
        raise ValueError("Input list length must be divisible by 7")

    result = codes_list[:7]

    for i in range(7, len(codes_list), 7):
        current_first = codes_list[i]
        previous_first = result[-7]

        if current_first != previous_first:
            result.extend(codes_list[i:i+7])

    return result


def create_training_example(text, audio_codes, tokenizer, max_audio_tokens=2000):
    """Create training example with proper token structure and length limiting"""
    text_ids = tokenizer.encode(text, add_special_tokens=True)
    text_ids.append(end_of_text)
    
    # Limit audio codes to prevent memory issues
    if len(audio_codes) > max_audio_tokens:
        audio_codes = audio_codes[:max_audio_tokens]
    
    input_ids = (
        [start_of_human]
        + text_ids
        + [end_of_human]
        + [start_of_ai]
        + [start_of_speech]
        + audio_codes
        + [end_of_speech]
        + [end_of_ai]
    )
    
    # Limit total sequence length to 4096 tokens
    max_length = 4096
    if len(input_ids) > max_length:
        # Truncate audio codes if sequence is too long
        text_part_length = len([start_of_human] + text_ids + [end_of_human] + [start_of_ai] + [start_of_speech])
        ending_length = len([end_of_speech] + [end_of_ai])
        available_audio_length = max_length - text_part_length - ending_length
        
        if available_audio_length > 0:
            truncated_audio = audio_codes[:available_audio_length]
            input_ids = (
                [start_of_human]
                + text_ids
                + [end_of_human]
                + [start_of_ai]
                + [start_of_speech]
                + truncated_audio
                + [end_of_speech]
                + [end_of_ai]
            )
        else:
            # If even text is too long, skip this example
            return None
    
    return {
        "input_ids": input_ids,
        "labels": input_ids.copy(),  # Same as input_ids for causal LM
        "attention_mask": [1] * len(input_ids)
    }


def custom_data_collator(features, tokenizer):
    """Custom data collator for training that handles padding and attention masks"""
    batch = {
        "input_ids": [],
        "labels": [],
        "attention_mask": []
    }
    
    # Find max length in this batch
    max_length = max(len(f["input_ids"]) for f in features)
    
    for feature in features:
        # Pad sequences to max length in batch
        padding_length = max_length - len(feature["input_ids"])
        
        # Pad input_ids and labels with pad_token_id
        padded_input_ids = feature["input_ids"] + [pad_token_id] * padding_length
        padded_labels = feature["labels"] + [-100] * padding_length  # -100 is ignored in loss computation
        padded_attention_mask = feature["attention_mask"] + [0] * padding_length
        
        batch["input_ids"].append(padded_input_ids)
        batch["labels"].append(padded_labels)
        batch["attention_mask"].append(padded_attention_mask)
    
    # Convert to tensors
    batch = {k: torch.tensor(v) for k, v in batch.items()}
    
    return batch


def compute_metrics(eval_preds):
    """Compute evaluation metrics"""
    predictions, labels = eval_preds
    loss = torch.nn.functional.cross_entropy(
        torch.tensor(predictions).view(-1, predictions.shape[-1]),
        torch.tensor(labels).view(-1),
        ignore_index=-100
    )
    return {"eval_loss": loss.item()}


def evaluate_model(model, dataset, tokenizer, split_name, batch_size=1):
    """Evaluate model on a specific dataset split with memory optimization"""
    logger.info(f"Evaluating model on {split_name} set...")
    
    # Use smaller batch size for evaluation to prevent OOM    
    eval_dataset = Dataset.from_list(dataset)
    temp_eval_dir = "temp_eval"
    eval_trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=temp_eval_dir,
            per_device_eval_batch_size=batch_size,
            remove_unused_columns=False,
            prediction_loss_only=True,  # Only compute loss, don't store predictions
            dataloader_pin_memory=False,  # Disable pin memory to save GPU memory
        ),
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=lambda features: custom_data_collator(features, tokenizer),
        compute_metrics=compute_metrics,
    )
    
    # Clear GPU cache before evaluation
    torch.cuda.empty_cache()
    
    # Evaluate with gradient disabled and in no_grad context
    with torch.no_grad():
        eval_metrics = eval_trainer.evaluate()
    
    # Clear cache after evaluation
    torch.cuda.empty_cache()
    
    # Remove temporary evaluation directory
    if os.path.exists(temp_eval_dir):
        shutil.rmtree(temp_eval_dir)
        logger.info(f"Removed temporary evaluation directory: {temp_eval_dir}")
    
    logger.info(f"{split_name} metrics: {eval_metrics}")
    return eval_metrics 