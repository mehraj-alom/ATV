from config.config import Config
import torch 
import onnxruntime as ort
from logging import getLogger
import cv2


config = Config()
logger = getLogger(__name__)


class CoreEngine:
    """
    Docstring for CoreEngine
    """
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.conf_thres = Config.CONF_THRESHOLD
        self.iou_thres = Config.IOU_THRESHOLD

        try : 
            self.session = ort.InferenceSession(
            self.model_path,
            providers=Config.PROVIDERS
            )
            self.input_names = [i.name for i in self.session.get_inputs()]
            self.output_names = [o.name for o in self.session.get_outputs()]
            self.input_shape =  Config.INPUT_SIZE
            self.img_size = Config.INPUT_SIZE
            self.classes = Config.CLASSES
            logger.info(f"ONNX model loaded successfully from {self.model_path}")
        except Exception as e:
            logger.error(f"Failed to load ONNX model from {self.model_path}: {e}")
            raise e
       

    def Preprocess(self,img):
        """
            Preprocess the input image for ONNX model inference. 
            This includes resizing, normalization, and any other transformations required by the model.
            Args: 
                    img (np.ndarray): The input image in BGR format (as read by OpenCV). 
            Returns:
                    np.ndarray: The preprocessed image ready for model inference. 
        """ 
        
        original_height, original_width = img.shape[:2]
        target_height, target_width = self.img_size, self.img_size

        scale_ratio = min(target_height / original_height, target_width / original_width)

        # new image size after scaling
        resized_width = int(round(original_width * scale_ratio))
        resized_height = int(round(original_height * scale_ratio))

        # padding to reach target size
        pad_width = target_width - resized_width
        pad_height = target_height - resized_height

        # divide padding evenly
        pad_left = pad_width / 2
        pad_right = pad_width / 2
        pad_top = pad_height / 2
        pad_bottom = pad_height / 2

        # resize if needed
        if (original_width, original_height) != (resized_width, resized_height):
            img = cv2.resize(img, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)
        