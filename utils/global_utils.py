import os 
import cv2
import yaml
import json
import joblib
from ensure import ensure_annotations
from pathlib import Path 
from typing import Any, List, Dict
import base64
from ATV.config import config
from ATV.logger import logger
from ATV.constants import Evidence_path

@ensure_annotations
def read_yaml(path_to_yaml: Path) -> dict:
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
def load_json(path: Path) -> dict:
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

@ensure_annotations
def read_video(video_path:str):
    """Reads a video from the specified path and returns a list of frames.  
    Args:
        video_path (str): The file path to the video.
    Returns:
        list: A list of frames extracted from the video.    
    """
    
    # option for live Video feed can be added by using cv2.VideoCapture(0)  <-- to be added 
    # instead of video path


    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret ,frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    logger.info(f"Video has been read succesfully")
    return frames

@ensure_annotations
def save_evidence(frame, filename: str , output_dir= Evidence_path):
    """Saves a frame as an image file in the specified output directory with the given filename.
    
    Args:
        frame: The video frame to be saved as an image.
        output_dir (str): The directory where the image will be saved.
        filename (str): The name of the image file output_dir= Evidence_path (without extension).
    """
    create_directories([output_dir])
    output_path = os.path.join(output_dir, f"{filename}.jpg")
    cv2.imwrite(output_path, frame)
    logger.info(f"Evidence saved at {output_path}")  # <-- to be added The track details 
                                                     # such as track id, timestamp, and bounding box
                                                     #  coordinates can be included in the log message for better traceability.