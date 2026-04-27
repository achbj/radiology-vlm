# even setting up following the documentation took more than 40 minutes
# the problem with internal caches for transformers
# solved it with unittest.mock.patch as context manager

import modal
import os

image = modal.Image.debian_slim().pip_install(
    "torch",
    "torchvision",
    "transformers==4.42.4",
    "datasets<=2.19.0",
    "peft==0.11.1", # problem with direct version --> 15 min wasted
    "einops",
    "timm",
    "accelerate>=0.21.0",
    "pillow"
)

volume = modal.Volume.from_name("vlm-model-vol", create_if_missing=True)
app = modal.App("radiology-vlm")

@app.function(
    image=image,
    gpu="a100-40GB",  # Using A100-40GB 
    volumes={"/vol/model": volume},
    timeout=7200 # I think it will take less than an hour  ## need to verify
)
def train():
    import logging
    import json
    import torch
    from datasets import load_dataset
    from transformers import (
        AutoModelForCausalLM, 
        AutoProcessor, 
        TrainingArguments, 
        Trainer,
        TrainerCallback
    )
    from peft import LoraConfig, get_peft_model
    
    # 1. Setup Logging -> need this for performance review
    log_file = "/vol/model/training.log"
    metrics_file = "/vol/model/training_metrics.json"
    
    os.makedirs("/vol/model", exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)
    logger.info("Starting Florence-2 VLM LoRA Fine-Tuning Pipeline on A100-40GB")
    
    # 2. Load Dataset
    logger.info("Loading VQA-RAD dataset...")
    dataset = load_dataset("flaviagiammarino/vqa-rad")
    train_ds = dataset["train"]
    logger.info(f"Loaded {len(train_ds)} examples for training.")
    
    # 3. Load model & Processor
    model_id = "microsoft/Florence-2-base"
    logger.info(f"Loading Base Model: {model_id} (trust_remote_code=True required for Florence-2)")
    
    # Bypass the transformers import checker using unittest.mock.patch to guarantee it intercepts the call
    from unittest.mock import patch
    with patch("transformers.dynamic_module_utils.check_imports", return_value=[]):
        processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True)
    
    # 4. LoRA Configuration --> following documentation
    logger.info("Injecting LoRA adapters...")
    config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj", "o_proj"],  # Common attention projection layers
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, config)
    logger.info(f"Trainable Parameters: {model.get_nb_trainable_parameters()}")
    
    # 5. Data Collator
    def collate_fn(batch):
        questions = [f"<QA> {item['question']}" for item in batch]
        answers = [item['answer'] for item in batch]
        images = [item['image'].convert("RGB") for item in batch]
        
        inputs = processor(text=questions, images=images, return_tensors="pt", padding=True)
        labels = processor.tokenizer(text=answers, return_tensors="pt", padding=True, return_attention_mask=False).input_ids
        
        # AI Suggestion: ->  Replace pad tokens with -100 so they are ignored in loss computation
        labels[labels == processor.tokenizer.pad_token_id] = -100
        inputs["labels"] = labels
        return inputs
        
    # 6. Metrics Callback
    class MetricsCallback(TrainerCallback):
        def __init__(self, metrics_file):
            self.metrics_file = metrics_file
            self.history = []
            
        def on_log(self, args, state, control, logs=None, **kwargs):
            if logs:
                self.history.append(logs)
                with open(self.metrics_file, 'w') as f:
                    json.dump(self.history, f, indent=4)
                    
    # 7. Training Arguments
    training_args = TrainingArguments(
        output_dir="/tmp/florence-vlm",
        per_device_train_batch_size=8,  # 40GB expected to fit 8 batch sizes for florance-vlm
        num_train_epochs=3,
        fp16=True,
        save_strategy="epoch",
        logging_steps=10,
        learning_rate=5e-4, # gemini suggest 5e-4 for florence-vlm LoRA ##### need to verify
        remove_unused_columns=False,
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        data_collator=collate_fn,
        callbacks=[MetricsCallback(metrics_file)]
    )
    
    logger.info("Starting PEFT Training Loop...")
    trainer.train()
    
    # 8. Save Final Model
    logger.info("Training complete. Saving LoRA weights to Modal Volume...")
    model.save_pretrained("/vol/model/florence-vqa-lora")
    processor.save_pretrained("/vol/model/florence-vqa-lora")
    volume.commit()
    logger.info("Saved successfully to /vol/model/florence-vqa-lora!")

@app.local_entrypoint()
def main():
    print("Triggering Florence-2 VLM LoRA training job on Modal A100...")
    train.remote()
