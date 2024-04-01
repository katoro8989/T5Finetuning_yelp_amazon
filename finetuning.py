import os
import time
import torch
import numpy as np
from transformers import Trainer, TrainingArguments, T5ForConditionalGeneration, T5Tokenizer
import dataset
import model
import argparse
import datetime
import wandb
import evaluate

def finetune(args):
    train_dataset = args.dataset
    
    run_id = f'{args.model}'
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f'{run_id}_{timestamp}'
    output_dir = os.path.join(args.output_dir, run_id)

    assert train_dataset is not None, "Please provide a training dataset."
    dataset_class = dataset.get_dataset(args.dataset, args.model)
    
    if args.checkpoint_path == None:
        model_class = model.get_model(args.model)
    else:
        model_class = T5ForConditionalGeneration.from_pretrained(args.checkpoint_path)
    
    tokenizer = T5Tokenizer.from_pretrained(args.model)
    
    if args.wandb:
        report = "wandb"
        print("=====wandb logging starts=====")
        wandb.init(project="t5-finrtunes",
            name=run_id,
            group="katoro13")
    else:
        report = None
    
    training_args = TrainingArguments(
        output_dir=output_dir,          
        num_train_epochs=args.epochs,
        gradient_accumulation_steps=args.gradient_accumulation_steps, 
        learning_rate=args.lr,        
        per_device_train_batch_size=args.train_batch_size,  
        per_device_eval_batch_size=args.eval_batch_size,   
        warmup_steps=args.warmup_steps,                
        weight_decay=args.weight_decay,               
        logging_strategy=args.logging_strategy, 
        run_name=run_id,  
        report_to=report, 
        fp16=args.fp16, 
        logging_dir='./logs', 
        evaluation_strategy=args.evaluation_strategy, 
        fp16_full_eval=args.fp16_full_eval, 
        eval_steps=args.eval_steps, 
        eval_accumulation_steps=args.eval_accumulation_steps, 
        auto_find_batch_size=args.auto_find_batch_size, 
    )
    
    print(training_args.device)
    
    sample_batch = next(iter(dataset_class["train"]))
    print(np.array(sample_batch["input_ids"]).shape)
    print(np.array(sample_batch["attention_mask"]).shape)
    print(np.array(sample_batch["labels"]).shape)
    
    metric = evaluate.load("accuracy")
    
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        
        predictions = [pred if pred in ['0', '1'] else '2' for pred in predictions] 
        
        # predictions = np.array([int(pred) for pred in predictions])
        # labels = np.array([int(label) for label in labels])

        # decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)
        
        # correct_predictions = [pred == label for pred, label in zip(predictions, labels)]
        # accuracy = sum(correct_predictions) / len(correct_predictions)
        # return accuracy
        
        return metric.compute(predictions=predictions, references=labels)

    
    def preprocess_logits_for_metrics(logits, labels):
        if isinstance(logits, tuple):
            logits = logits[0]
        return logits.argmax(dim=-1)

    
    trainer = Trainer(
        model=model_class,
        args=training_args,
        train_dataset=dataset_class["train"],
        eval_dataset=dataset_class["test"],
        compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
    )
    
    trainer.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Finetuning of T5')
    parser.add_argument('--dataset', type=str, default="yelp_polarity")
    parser.add_argument('--model', type=str, default="t5-small")
    parser.add_argument('--output_dir', type=str)
    parser.add_argument('--epochs', type=float, default=1.)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=8)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--train_batch_size', type=int, default=2)
    parser.add_argument('--eval_batch_size', type=int, default=2)
    parser.add_argument('--warmup_steps', type=int, default=0)
    parser.add_argument('--weight_decay', type=int, default=0)
    parser.add_argument('--fp16', action='store_true', help='whether fp16')
    parser.add_argument('--logging_dir', type=int, default=None)
    parser.add_argument('--logging_strategy', type=str, default="steps")
    parser.add_argument('--evaluation_strategy', type=str, default="steps")
    parser.add_argument('--wandb', action='store_true', help='whether log on wandb')
    parser.add_argument('--exp_id', type=str, default=None, help='exp id for reporting')
    parser.add_argument('--fp16_full_eval', action='store_true')
    parser.add_argument('--eval_steps', type=float, default=500.)
    parser.add_argument('--eval_accumulation_steps', type=int)
    parser.add_argument('--checkpoint_path', type=str, default=None)
    parser.add_argument('--auto_find_batch_size', action='store_true')
    args = parser.parse_args()
                    
    
    finetune(args)
