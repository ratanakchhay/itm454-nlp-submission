import os
import json
import argparse
import shutil
# 

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--images", type = str, default = None)
parser.add_argument("-g", "--golden", type = str, default = None)
parser.add_argument("-o", "--out", type = str, default = None)
args = parser.parse_args()

# Directory of the images and labels 
data_dir = os.path.join("..", "..", "data")
image_dir = os.path.join(data_dir, args.images)
golden_dir = os.path.join(data_dir, args.golden)
record = os.path.join(data_dir, args.out)

# Predefined extensions for the images
ext = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}

# Default to procedural as it is run only once 
if __name__ == "__main__":
    list_image_dir, list_golden_dir = os.listdir(image_dir), os.listdir(golden_dir)
    records = [] # For keeping track of the records of the files
    os.makedirs(os.path.join(data_dir, "unlabeled"), exist_ok = True)
    for image in list_image_dir:
        # Checking whether the golden record exists
        # If not will be used for testing data
        image_filename, image_ext = os.path.splitext(image)
        # 
        if image_ext.lower() not in ext:
            continue
        # 
        if image_filename + ".json" not in list_golden_dir:
            # Moving into the unlabeled dir for testing
            shutil.move(
                src = os.path.join(image_dir, image),
                dst= os.path.join(data_dir, "unlabeled", image)
            )
            continue 
        # 
        # Opening the golden record and appending
        with open(os.path.join(golden_dir, image_filename + ".json"), encoding = "utf-8") as f:
            answer = json.load(f)
        records.append({"image": os.path.join("data", args.images, image), "answer": answer})
    # 
    with open(record, "w", encoding = "utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii = False) + "\n")