from paddleocr import PaddleOCR
# 
from helper import check_valid_extension

def run_ocr_inference(ocr_model : PaddleOCR, img_path : str) -> dict:
    """
        Function: run_ocr_inference
        @param: ocr_model (PaddleOCR), img_path (str)
        @return: dict
    
        Returns the result of the OCR inference.
    """
    if ocr_model == None:
        return {}
    
    if not check_valid_extension(img_path):
        return {}

    return ocr_model.predict(input= img_path)

def extract_bbox_list(inf_res : dict) -> list:
    """
        Function: extract_bbox_list
        @param: inf_res (dict)
        @return: list
    
        Returns the bbox coordinates as a list.
    """
    if inf_res == {}:
        return []

    try:
        bbox_coord_list : list = inf_res[0]["rec_polys"]
    except Exception as e:
        return []

    return bbox_coord_list