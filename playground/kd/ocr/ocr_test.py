import os
# 
from paddleocr import PaddleOCR
#
from ocr import (
    run_ocr_inference,
    extract_bbox_list
)

if __name__ == "__main__":
    # Getting the image path
    img_path = os.path.join(".", "public", "testing_pp_ocr.jpeg")
    if not os.path.exists(img_path):
        print("Error while accessing the image")
        quit()

    # Instantiating OCR model 
    try:
        ocr_model = PaddleOCR(
            text_detection_model_name="PP-OCRv6_medium_det", 
            text_recognition_model_name="PP-OCRv6_medium_rec", 
            engine="transformers", 
            use_doc_orientation_classify=False, 
            use_doc_unwarping=False, 
            use_textline_orientation=False,
        )
    except Exception as e:
        print("Error while trying to load the OCR model: {e}")
        quit()

    # Testing the inference on the image
    inf_res : dict = run_ocr_inference(img_path=img_path, ocr_model= ocr_model)
    if inf_res == {}:
        print("Error while running inference on the image")
        quit()

    # Testing the extraction of bbox coordinates
    bbox_list : list = extract_bbox_list(inf_res=inf_res)
    if bbox_list == []:
        print("Error while extracting bbox list on the image")
        quit()

    # Return result
    for idx, bbox in enumerate(bbox_list):
        print(f"{idx}: {bbox}\n")