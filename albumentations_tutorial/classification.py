import cv2
import albumentations as A
import numpy as np
from utils import plots_examples
from PIL import Image
from tqdm import tqdm

image = Image.open('images/elon.jpeg')
