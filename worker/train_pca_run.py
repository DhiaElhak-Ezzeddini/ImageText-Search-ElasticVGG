# train_pca_run.py (example)
from shared.feature_extractor import HOGExtractor
from shared.pca_tools import train_ipca, save_model
import glob
import os 
data = os.getenv("DATA_ROOT", "/data")  # Use /data as default if DATA_ROOT is not set

hog_ext = HOGExtractor()

# generators for HOG and LBP features
def hog_gen(paths):
    for p in paths:
        yield hog_ext.extract(p)


print("Collecting image paths...")
paths = []
for i in range(5):  # 0,1,2,3,4 ==> 50k images for training PCA
    paths.extend(glob.glob(f"{data}/{i}/*.jpg"))
print(f"Found {len(paths)} images")

ipca_hog = train_ipca(hog_gen(paths), n_components=256)
save_model(ipca_hog, "/models/ipca_hog.joblib")

