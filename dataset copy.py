import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt

from datasets import load_dataset
from transformers import T5Tokenizer


DATASETS = [
    "yelp_polarity", 
    "amazon_polarity", 
]


def get_dataset(dataset_name, model_name):
    '''Return a tokenized dataset as a Dataset of pytorch'''
    if dataset_name not in DATASETS:
        raise NotImplementedError("Dataset not found: {}".format(dataset_name))
    
    dataset = load_dataset(dataset_name)
    
    if dataset_name == "amazon_polarity":
        dataset = dataset.map(amazon_preprocess_function, batched=True, fn_kwargs={'model_name': model_name})
    elif dataset_name == "yelp_polarity":
        dataset = dataset.map(yelp_preprocess_function, batched=True, fn_kwargs={'model_name': model_name})
    
    return dataset


def amazon_preprocess_function(batch, model_name):
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model_inputs = tokenizer(batch["content"], truncation=True, padding='max_length', return_tensors='pt', max_length=128)
    label_temp = batch["label"]
    if not isinstance(label_temp, list):
        label_temp = [label_temp]
    labels_as_text = [str(label) for label in label_temp]
    labels = tokenizer(labels_as_text, truncation=True, padding='max_length', return_tensors='pt', max_length=128).input_ids
    model_inputs["labels"] = labels
    return model_inputs

def yelp_preprocess_function(batch, model_name):
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model_inputs = tokenizer(batch["text"], truncation=True, padding='max_length', return_tensors='pt', max_length=128)
    label_temp = batch["label"]
    if not isinstance(label_temp, list):
        label_temp = [label_temp]
    labels_as_text = [str(label) for label in label_temp]
    labels = tokenizer(labels_as_text, truncation=True, padding='max_length', return_tensors='pt', max_length=128).input_ids
    model_inputs["labels"] = labels
    return model_inputs

