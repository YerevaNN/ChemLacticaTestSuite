import torch
import yaml
import os
from typing import Callable, Dict, Iterable, List, Optional, Tuple, Union, Any
from transformers.generation import LogitsProcessor,LogitsProcessorList
from dataclasses import dataclass
import importlib


@dataclass
class LogitsProcessorConfig:
    class_name: str
    module: str
    kwargs: Dict[str, Any]

def instantiate_processors(config: List[LogitsProcessorConfig]) -> List[LogitsProcessor]:
    processors = []
    for processor_config in config:
        module = importlib.import_module(processor_config.module)
        processor_class = getattr(module, processor_config.class_name)
        processor = processor_class(**processor_config.kwargs)
        processors.append(processor)
    return processors

def load_processor_config(file_path: str) -> List[LogitsProcessorConfig]:
    with open(file_path, 'r') as file:
        config_data = yaml.safe_load(file)
        return [LogitsProcessorConfig(class_name=processor['class_name'],module = processor['module'], kwargs=processor['kwargs']) for processor in config_data['logits_processors']]

def get_logits_processors(logits_processors_config_path = None):
    # current_dir = os.path.dirname(os.path.abspath(__file__))
    # config_file_path = os.path.join(current_dir, "logit_configs","best_config.yaml")
    if logits_processors_config_path:
        logit_processors_config = load_processor_config(logits_processors_config_path)
        logit_processors = instantiate_processors(logit_processors_config)
        logit_processor_list = LogitsProcessorList(logit_processors)
        return logit_processor_list
    else:
        return None

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    config_file_path = os.path.join(current_dir, "logit_configs","best_config.yaml")
    logit_processors_config = load_processor_config(config_file_path)
    logit_processors = instantiate_processors(logit_processors_config)
    logit_processor_list = LogitsProcessorList(logit_processors)
    print(logit_processor_list)

