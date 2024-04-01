import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models

from accelerate import Accelerator

import numpy as np

from transformers import T5ForConditionalGeneration


MODELS = [
    "t5-small", 
    "t5-base", 
    "t5-large",  
]

def get_model(modelname):
    if modelname not in MODELS:
        raise NotImplementedError("Model not found: {}".format(modelname))

    # device_index = Accelerator().process_index
    # device_map = {"": device_index}
    return T5ForConditionalGeneration.from_pretrained(modelname, device_map="auto")
