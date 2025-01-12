from utils.ocr_tools.surya_ocr_tool import get_image_text_suryaocr
from utils.ocr_tools.easy_ocr_tool import get_image_text_easyocr
from utils.ocr_tools.reframe_ocr_text import reframe_the_ocr_text_into_a_proper_format
from utils.Document_validation.electricity_bill import electricity_bill_extract_event_information
from utils.Document_validation.blurness_detection.blurness_detection import blur_detection

def document_type_verification(doc_path):
    blurness_value = blur_detection(doc_path)
    if blurness_value == False:
        print("Image clear")
        text_extracted_surya_ocr = get_image_text_suryaocr(doc_path)
        #text_extracted_easy_ocr = get_image_text_easyocr(doc_path)
        print("Surya OCR-> ", text_extracted_surya_ocr)
        #print("Easy OCR-> ", text_extracted_easy_ocr)
        formatted_text = reframe_the_ocr_text_into_a_proper_format(text_extracted_surya_ocr,  " ")
        document_validation = electricity_bill_extract_event_information(formatted_text)
    else:
        return ("Image is not clear to the OCR")
    return  document_validation

image_path = "/root/rohit/document_ocr/images/151663301_Dec-24.pdf"
document_validation_result = document_type_verification(image_path)
print("Result -> ", document_validation_result)