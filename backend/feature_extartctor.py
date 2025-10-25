import tensorflow.keras.preprocessing.image as image
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from skimage.feature import local_binary_pattern, hog
from tensorflow.keras.models import Model 
import numpy as np
from config import data_path
from skimage  import io
from io import BytesIO
import base64
from PIL import Image


class FeatureExtractor:
    def __init__(self):
        base_model = VGG16(weights='imagenet')
        self.model = Model(inputs=base_model.input, outputs=base_model.get_layer('fc1').output)

    def _extract(self,img):
        if isinstance(img, str):
            img = Image.fromarray(img)
        img = img.resize((224, 224))
        img = img.convert("RGB")
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        features = self.model.predict(x)[0]
        return features / np.linalg.norm(features)
    
    def get_from_link(self,img_link:str):
        image = io.imread(img_link)
        return self._extract(image)

    def get_from_image(self,img): 
        image_data = base64.b64decode(img)
        image = Image.open(BytesIO(image_data))
        return self._extract(image)
class LBPExtractor:
    def __init__(self, P=8, R=1):
        self.P = P
        self.R = R

    def _extract(self,img):
        if isinstance(img, str):
            img = Image.fromarray(img)
        img = img.resize((128, 128))
        img = img.convert("L")
        x = image.img_to_array(img)
        x = np.squeeze(x)
        lbp = local_binary_pattern(x, self.P, self.R, method="uniform")
        (hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, self.P + 3), range=(0, self.P + 2))
        hist = hist.astype("float")
        hist /= (hist.sum() + 1e-7)
        return hist

    def get_from_link(self,img_link:str):
        image = io.imread(img_link)
        return self._extract(image)

    def get_from_image(self,img): 
        image_data = base64.b64decode(img)
        image = Image.open(BytesIO(image_data))
        return self._extract(image)

class HOGExtractor:
    def __init__(self, pixels_per_cell=(16, 16), cells_per_block=(2, 2), orientations=9):
        self.pixels_per_cell = pixels_per_cell
        self.cells_per_block = cells_per_block
        self.orientations = orientations

    def _extract(self,img):
        if isinstance(img, str):
            img = Image.fromarray(img)
        img = img.resize((128, 128))
        img = img.convert("L")
        x = image.img_to_array(img)
        x = np.squeeze(x)
        hog_features = hog(x, orientations=self.orientations, pixels_per_cell=self.pixels_per_cell,
                           cells_per_block=self.cells_per_block, block_norm='L2-Hys')
        return hog_features / np.linalg.norm(hog_features)

    def get_from_link(self,img_link:str):
        image = io.imread(img_link)
        return self._extract(image)

    def get_from_image(self,img): 
        image_data = base64.b64decode(img)
        image = Image.open(BytesIO(image_data))
        return self._extract(image)