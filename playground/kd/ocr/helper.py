from pathlib import Path
# 
from constant import IMG_EXTENSIONS

def check_valid_extension(img_path : str) -> bool:
    """
    Function: check_valid_extension
    @param: img_path (str)
    @return: bool

    Checks whether the image path is of valid extension.
    """
    if img_path == "":
        return False
    
    if Path(img_path).suffix.lower() not in IMG_EXTENSIONS:
        return False

    return True