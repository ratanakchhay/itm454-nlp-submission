import os
import random
import shutil
import argparse
#

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--images", type = str, default = os.path.join("..", "..", "data", "images"), help = "path to the images")
parser.add_argument("-s", "--seed", type = int, default = 0, help = "seed for the shuffle, keeps the split reproducible")
parser.add_argument("--train", type = float, default = 0.70)
parser.add_argument("--test", type = float, default = 0.15)
args = parser.parse_args()

# Predefined extensions for the images
ext = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}

# Running this procedurally as it is run only once
if __name__ == "__main__":
    if not os.path.exists(args.images):
        print("Path to the images directory does not exist")
        exit()

    # Claude generated code; manually verified as it is simple list comprehension
    # 
    # Sorted before shuffling so the same seed gives the same split
    images = sorted(f for f in os.listdir(args.images) if os.path.splitext(f)[1].lower() in ext)
    random.Random(args.seed).shuffle(images)
    # The rest goes to val, so no image is dropped by the rounding
    n_train = round(len(images) * args.train)
    n_test = round(len(images) * args.test)
    splits = {
        "train": images[:n_train],
        "test": images[n_train:n_train + n_test],
        "val": images[n_train + n_test:],
    }
    # 

    # Moving the images
    for split, files in splits.items():
        os.makedirs(os.path.join(args.images, split), exist_ok = True)
        for f in files:
            shutil.move(
                src = os.path.join(args.images, f),
                dst = os.path.join(args.images, split, f)
            )
        print(f"{split}: {len(files)}")
