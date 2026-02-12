import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from config.config import Config
# print(Config.MODEL_PATH)
# print(Config.CLASSES)

from core.input_feed import Mock_input_feed 
mock_feed = Mock_input_feed(source = "videos/vid1.mp4")
mock_feed.get_frame()
mock_feed.release()