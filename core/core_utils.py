from pathlib import Path
import json

import torch
from ATV.logger import logger
import torch
import yaml 
from yaml.loader import SafeLoader
from ensure import ensure_annotations

@ensure_annotations
def load_model(path: str):
    """Loads a model from the specified path.
    
    Args:
        path (str): The file path to the model.
    Returns:
        The loaded model object."""
    
    try :
        model = torch.load(path)
        logger.info(f"Model loaded successfully from {path}")
        model.setPreferableBackend(torch.backends.cudnn)
        model.setPreferableTarget(torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'))
        return model
    except FileNotFoundError:
        raise FileNotFoundError(f"Model file {path} does not exist.")
    except Exception as e:
        raise e
    

@ensure_annotations
def load_labels(labels_file_path: str):
    """Loads labels from a text file.
    
    Args:
        labels_file_path (str): The file path to the labels text file.
    Returns:
        A list of labels."""
    try : 
        with open(labels_file_path, 'r') as f:
            data_labels = yaml.load(f, Loader=SafeLoader)
        logger.info(f"Labels loaded successfully from {labels_file_path}")
        return data_labels["names"]
    except FileNotFoundError:
        raise FileNotFoundError(f"Labels file {labels_file_path} does not exist.")
    except yaml.YAMLError as e:
        raise ValueError(f"Invalid YAML file: {e}") from e
    except Exception as e:
        raise e
