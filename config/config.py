class DetectionConfig:
    def __init__(self, confidence_threshold=0.5, 
                 probability_threshold=0.5,
                 nms_threshold=0.4,
                 input_width = 640,
                input_height = 640):
        self.confidence_threshold = confidence_threshold
        self.probability_threshold = probability_threshold
        self.nms_threshold = nms_threshold
        self.input_width = input_width
        self.input_height = input_height