import os 
import yaml
import json
import joblib
from ensure import ensure_annotations
from pathlib import Path 
from typing import Any, List, Dict
import base64
from ATV.logger import logger


@ensure_annotations
def read_yaml(path_to_yaml: Path) -> Dict[str, Any]:
    """reads yaml file and returns a dict
    Args:
        path_to_yaml (Path) : path like input

    raises: 
        FileNotFoundError: if the file does not exist
        ValueError: if yaml parsing fails
    returns:
        dict : parsed YAML as dict
    """
    try: 
        with open(path_to_yaml) as yaml_file:
            content = yaml.safe_load(yaml_file)
            if content is None:
                content = {}
            logger.info(f"YAML file {path_to_yaml} loaded successfully.")
            return content
    except FileNotFoundError:
        raise FileNotFoundError(f"File {path_to_yaml} does not exist.")
    except yaml.YAMLError as e:
        raise ValueError(f"Invalid YAML file: {e}") from e
    except Exception as e:
        raise e
    
@ensure_annotations
def create_directories(path_to_directories: list, verbose=True):

    """Creates directories if they do not exist.
    
    Args:
        path_to_directories (list): List of directory paths to create.
        verbose (bool): If True, logs the creation of directories.
    """
    for path in path_to_directories:
        path = Path(path)
        os.makedirs(path, exist_ok = True)
        if verbose:
            logger.info(f"Directory created at {path}")
    


@ensure_annotations
def save_json(path :Path, data:Dict[str, Any]):
    """save json data
    
    args:
        path(Path) : Path at which json will be saved 
        data(dict) : data to be saved in that json file    
    """
    with open(path,"w") as f:
        json.dump(data,f,indent=5)
    logger.info(f"Json file saved at {path}")

@ensure_annotations
def load_json(path: Path) -> Dict[str, Any]:
    """load json file
    Args:
        path (Path): Path from which json file to be load 
    returns:
        dict : parsed JSON    
    """
    with open(path,"r") as f:
        content = json.load(f)
    logger.info(f"Json file loaded from {path}")
    return content


@ensure_annotations
def get_size(path: Path) -> str:
    """
    Get file size kB
    Args :
         path (Path) : Path of the file to get size
    Returns:
        int : Size of the file in kB
    """
    size_in_kb = round(os.path.getsize(path) / 1024)
    return f"{size_in_kb} kB"