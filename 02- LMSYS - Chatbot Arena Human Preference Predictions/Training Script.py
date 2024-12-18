import argparse
import logging
from dataclasses import dataclass, asdict
import os
import copy
import numpy as np
import torch
from datasets import Dataset
from sklearn.preprocessing import LabelEncoder
from transformers import (
    BitsAndBytesConfig,
    Gemma2ForSequenceClassification,
    GemmaTokenizerFast,
    Gemma2Config,
    PreTrainedTokenizerBase,
    EvalPrediction,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
    TrainerCallback,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
from sklearn.metrics import log_loss, accuracy_score
from huggingface_hub import HfApi
from datasets import concatenate_datasets
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.trainer_utils import EvalPrediction
from dataclasses import field
from typing import List
from datasets import concatenate_datasets
import pandas as pd
import warnings

import argparse
import logging
import pandas as pd
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from transformers import Trainer, TrainingArguments, DataCollatorWithPadding
from sklearn.preprocessing import LabelEncoder
from datasets import Dataset

warnings.simplefilter('ignore')


@dataclass
class Config:
    VER: int = 19
    max_length: int = 3072
    n_splits: int = 5
    fold_idx: int = 0
    per_device_train_batch_size: int = 32
    gradient_accumulation_steps: int = 1
    per_device_eval_batch_size: int = 32
    n_epochs: int = 1
    freeze_layers: int = 16
    lr: float = 2e-4
    warmup_steps: int = 20
    lora_r: int = 16
    lora_alpha: float = 32
    lora_dropout: float = 0.05
    lora_bias: str = "none"
    eval_strategy: str = "epoch"
    save_strategy: str = "steps"
    save_steps: int = 200
    logging_steps: int = 5
    output_dir: str = 'lmsys-output/'
    checkpoint: str = "google/gemma-2-9b-it"  #"/root/autodl-tmp/lmsys/gemma/gemma-2-9b-it-bnb-4bit_mirror"
    mirror_url: str = "https://hf-mirror.com"
    optim_type: str = "paged_adamw_32bit"
    train_csv: str = "lmsys-chatbot-arena/train.csv"
    extra_train: str = "lmsys-additional-33k-labelled-conversations/lmsys-33k-deduplicated.csv"
    extra_train2: str = "lmsys-additional-33k-labelled-conversations/hf52k_deduplicated.csv"

class CustomTokenizer:
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        max_length: int
    ) -> None: 
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, batch: dict) -> dict:
        prompt = ["<prompt>: " + self.process_text(t) for t in batch["prompt"]]
        response_a = ["\n\n<response_a>: " + self.process_text(t) for t in batch["response_a"]]
        response_b = ["\n\n<response_b>: " + self.process_text(t) for t in batch["response_b"]]
        texts = [p + r_a + r_b for p, r_a, r_b in zip(prompt, response_a, response_b)]
        
        # Calculate lengths
        lengths = [len(text) for text in texts]
        
        # Tokenize the texts
        tokenized = self.tokenizer(texts, max_length=self.max_length, truncation=True)
        
        # Generate labels
        labels = []
        for a_win, b_win in zip(batch["winner_model_a"], batch["winner_model_b"]):
            if a_win:
                label = 0
            elif b_win:
                label = 1
            else:
                label = 2
            labels.append(label)
        
        # Add lengths to the returned dictionary
        return {**tokenized, "labels": labels, "lengths": lengths}

    @staticmethod
    def process_text(text: str) -> str:
        return " ".join(eval(text, {"null": ""}))

class EvaluationCallback(TrainerCallback):
    def __init__(self, trainer, eval_dataset, logger):
        self.trainer = trainer
        self.eval_dataset = eval_dataset
        self.logger = logger

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % 300 == 0:
            self.logger.info(f"Evaluating at step {state.global_step}...")
            metrics = self.trainer.evaluate(eval_dataset=self.eval_dataset)
            self.logger.info(f"Step {state.global_step} evaluation results:")
            for key, value in metrics.items():
                self.logger.info(f"  {key}: {value}")

def compute_metrics(eval_preds: EvalPrediction) -> dict:
    preds = eval_preds.predictions
    labels = eval_preds.label_ids
    probs = torch.from_numpy(preds).float().softmax(-1).numpy()
    loss = log_loss(y_true=labels, y_pred=probs)
    acc = accuracy_score(y_true=labels, y_pred=preds.argmax(-1))
    return {"acc": acc, "log_loss": loss}

def setup_logging(config):
    log_dir = os.path.join(config.output_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"training_log_v{config.VER}.txt")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def log_parameters(logger, config):
    logger.info("=== Parameter Settings ===")
    for key, value in asdict(config).items():
        logger.info(f"  {key}: {value}")
    logger.info("="*100)


def main():
    parser = argparse.ArgumentParser(description="Train Gemma2 model with QLoRA")
    parser.add_argument("--ver", type=int, default=19, help="Version number")
    parser.add_argument("--max_len", type=int, default=3072, help="Maximum sequence length")
    parser.add_argument("--batch_size", type=int, default=8, help="Per device train batch size")
    parser.add_argument("--grad_acc_steps", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--warmup_steps", type=int, default=20, help="Warmup steps")
    parser.add_argument("--save_steps", type=int, default=200, help="Save steps")
    parser.add_argument("--eval_batch_size", type=int, default=32, help="Eval batch size")
    parser.add_argument("--extra_train", type=str, default="lmsys-additional-33k-labelled-conversations/lmsys-33k-deduplicated.csv", help="External Train Data Path")
    parser.add_argument("--extra_train2", type=str, default="lmsys-additional-33k-labelled-conversations/hf52k_deduplicated.csv", help="External Train Data Path 2")
    parser.add_argument("--freeze_layers", type=int, default=16, help="Freeze layers")
    parser.add_argument("--lora_r", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--n_splits", type=int, default=10, help="Number of data splits for cross-validation")
    parser.add_argument("--fold_idx", type=int, default=0, help="Index of the current fold")
    parser.add_argument("--epochs", type=int, default=1, help="Number of training epochs")
    parser.add_argument("--local_rank", type=int, default=-1, help="Local rank for distributed training")
    
    args = parser.parse_args()

    if args.local_rank == -1:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        dist.init_process_group(backend='nccl')
        torch.cuda.set_device(args.local_rank)
        device = torch.device(f"cuda:{args.local_rank}")

    config = Config(
        VER=args.ver,
        max_length=args.max_len,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_acc_steps,
        warmup_steps=args.warmup_steps,
        save_steps=args.save_steps,
        per_device_eval_batch_size=args.eval_batch_size,
        extra_train=args.extra_train,
        freeze_layers=args.freeze_layers,
        lora_r=args.lora_r,
        lora_alpha=args.lora_r*2,
        lora_dropout=args.lora_dropout,
        lr=args.lr,
        n_splits=args.n_splits,
        fold_idx=args.fold_idx,
        n_epochs=args.epochs,
        output_dir=f"Gemma2_QLoRA_ft"
    )

    logger = setup_logging(config)
    log_parameters(logger, config)

    train = pd.read_csv(config.train_csv)
    train['label'] = train[['winner_model_a', 'winner_model_b', 'winner_tie']].idxmax(axis=1)
    label_encoder = LabelEncoder()
    train['label'] = label_encoder.fit_transform(train['label'])
    flipped_train = train.copy()
    flipped_train['response_a'], flipped_train['response_b'] = train['response_b'], train['response_a']
    flipped_train['winner_model_a'], flipped_train['winner_model_b'] = train['winner_model_b'], train['winner_model_a']
    # augmented_train = pd.concat([train, flipped_train], ignore_index=True)
    augmented_train = train
    train = train[["prompt", "response_a", "response_b", 'winner_model_a', 'winner_model_b', 'winner_tie', 'label']]
    train_extended = augmented_train[["prompt", "response_a", "response_b", 'winner_model_a', 'winner_model_b', 'winner_tie', 'label']]

    if config.extra_train != "none":
        # extra_data = pd.concat([pd.read_csv(config.extra_train),pd.read_csv(config.extra_train2)],axis=0).reset_index(drop=True)
        extra_data = pd.read_csv(config.extra_train)
        extra_data['label'] = extra_data[['winner_model_a', 'winner_model_b', 'winner_tie']].idxmax(axis=1)
        extra_data['label'] = label_encoder.transform(extra_data['label'])
        flipped_extra_train = extra_data.copy()
        flipped_extra_train['response_a'], flipped_extra_train['response_b'] = extra_data['response_b'], extra_data['response_a']
        flipped_extra_train['winner_model_a'], flipped_extra_train['winner_model_b'] = extra_data['winner_model_b'], extra_data['winner_model_a']
        # augmented_extra_train = pd.concat([extra_data, flipped_extra_train], ignore_index=True)
        augmented_extra_train = extra_data
        extra_data = augmented_extra_train[["prompt", "response_a", "response_b", 'winner_model_a', 'winner_model_b', 'winner_tie', 'label']]
        train_extended = pd.concat([train_extended, extra_data], ignore_index=True)
        train_extended = train_extended.drop_duplicates()
        logger.info(f"Loaded extra data with {len(extra_data)} samples")
        

    # Convert DataFrame to Dataset
    # Create train/eval split
    folds = [
        (
            [i for i in range(len(train)) if i % config.n_splits != config.fold_idx],
            [i for i in range(len(train)) if i % config.n_splits == config.fold_idx]
        )
    ]
    train_idx, eval_idx = folds[config.fold_idx]

    tokenizer = GemmaTokenizerFast.from_pretrained(config.checkpoint)
    tokenizer.add_eos_token = True
    tokenizer.padding_side = "right"

    if config.extra_train != "none":
        train_eval = train.iloc[eval_idx]

        mask = train_extended.apply(tuple, 1).isin(train_eval.apply(tuple, 1))
        train_extended = train_extended[~mask]

        print("Length of extended data: ", len(train_extended))

        custom_tokenizer = CustomTokenizer(tokenizer, config.max_length)
        original_ds = Dataset.from_pandas(train).map(
            custom_tokenizer,
            batched=True,
            remove_columns=train.columns.tolist()
        )

        train_extended = Dataset.from_pandas(train_extended).map(
            custom_tokenizer,
            batched=True,
            remove_columns=train_extended.columns.tolist()
        )

    if config.extra_train == "none":
        print("Not using augmented data!")

    # Combine original training data with extra data
    if config.extra_train != "none":
        train_ds = train_extended   #concatenate_datasets([original_ds.select(train_idx), extra_ds])

    else:
        train_ds = original_ds.select(train_idx)

    eval_ds = original_ds.select(eval_idx)

    logger.info(f"Training dataset size: {len(train_ds)}")
    logger.info(f"Evaluation dataset size: {len(eval_ds)}")

    training_args = TrainingArguments(
        output_dir=config.output_dir,
        save_total_limit=1,
        overwrite_output_dir=True,
        report_to="none",
        num_train_epochs=config.n_epochs,
        per_device_train_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        per_device_eval_batch_size=config.per_device_eval_batch_size,
        logging_steps=config.logging_steps,
        evaluation_strategy="no",
        save_strategy=config.save_strategy,
        save_steps=config.save_steps,
        optim=config.optim_type,
        # fp16=True,
        # bf16=True,
        learning_rate=config.lr,
        warmup_steps=config.warmup_steps,
        # local_rank=args.local_rank,
        # lr_scheduler_type='cosine',
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={'use_reentrant':False},
        max_grad_norm=10.0,
        # ddp_find_unused_parameters=False,
        # dataloader_pin_memory=False,
    )

    # Uncomment to train Lora, otherwise train full model.
    # lora_config = LoraConfig(
    #     r=config.lora_r,
    #     lora_alpha=config.lora_alpha,
    #     target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj"],
    #     layers_to_transform=[i for i in range(42) if i >= config.freeze_layers],
    #     lora_dropout=config.lora_dropout,
    #     bias=config.lora_bias,
    #     task_type=TaskType.SEQ_CLS,
    # )


    # Remove columns other than input_ids, attention_mask, and labels
    # train_ds.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
    # eval_ds.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

    model = Gemma2ForSequenceClassification.from_pretrained(
        config.checkpoint,
        num_labels=3,
        torch_dtype=torch.float16,
        device_map='balanced'
    )
    model.config.use_cache = False

    # Freeze the first k layers
    k = 26  # Change this to the number of layers you want to freeze
    for name, param in model.named_parameters():
        if "layers" in name:
            layer_num = int(name.split('.')[2])
            if layer_num < k:
                param.requires_grad = False
    # model = prepare_model_for_kbit_training(model)
    # model = get_peft_model(model, lora_config)

    # model.to(device)
    # if args.local_rank != -1:
    #     model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank,find_unused_parameters=False)

    def print_trainable_parameters(model):
        trainable_params = 0
        all_param = 0
        for _, param in model.named_parameters():
            all_param += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
        return f"trainable params: {trainable_params:,d} || all params: {all_param:,d} || trainable%: {100 * trainable_params / all_param:.2f}"
    
    logger.info("Model loaded and prepared for training.")
    logger.info(print_trainable_parameters(model))

    # train_sampler = DistributedSampler(train_ds, num_replicas=dist.get_world_size(), rank=dist.get_rank()) if args.local_rank != -1 else None
    # train_loader = DataLoader(train_ds, batch_size=config.per_device_train_batch_size, sampler=train_sampler)
    
    # eval_sampler = DistributedSampler(eval_ds, num_replicas=dist.get_world_size(), rank=dist.get_rank()) if args.local_rank != -1 else None
    # eval_loader = DataLoader(eval_ds, batch_size=config.per_device_eval_batch_size, sampler=eval_sampler)



    logger.info(f"Training dataset size: {len(train_ds)}")
    logger.info(f"Evaluation dataset size: {len(eval_ds)}")

    trainer = Trainer(
        args=training_args,
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        compute_metrics=compute_metrics,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer)
    )

    evaluation_callback = EvaluationCallback(trainer, eval_ds, logger)
    trainer.add_callback(evaluation_callback)

    logger.info("Starting training...")
    train_result = trainer.train()

    logger.info("Training completed. Results:")
    for key, value in train_result.metrics.items():
        logger.info(f"  {key}: {value}")

    logger.info("Performing final evaluation...")
    eval_result = trainer.evaluate()
    logger.info("Final evaluation results:")
    for key, value in eval_result.items():
        logger.info(f"  {key}: {value}")

if __name__ == "__main__":
    main()
