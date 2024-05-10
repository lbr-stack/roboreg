import glob
import os

files = glob.glob("image_*.png")
for file in files:
    os.rename(file, file.replace("image_", "mask_"))
