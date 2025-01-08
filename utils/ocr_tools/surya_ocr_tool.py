from PIL import Image
from surya.ocr import run_ocr
from surya.model.detection.model import load_model as load_det_model, load_processor as load_det_processor
from surya.model.recognition.model import load_model as load_rec_model
from surya.model.recognition.processor import load_processor as load_rec_processor
import os
from pdf2image import convert_from_path

def get_image_text_suryaocr(image_path: str):
    """Extracts text from the given image using OCR models.
    
    :param image_path: Path to the image file.
    :return: Extracted text as a string or error message.
    """
    try:
        # Open the image file or convert PDF to images
        if image_path.lower().endswith(".pdf"):
            print(f"Converting PDF {image_path} to images...")
            images = convert_from_path(image_path)
            print("Converted PDF to image")
            image = images[0]  # Use the first page of the PDF as an image
        else:
            image = Image.open(image_path)
    
        # Load detection and recognition models and processors
        det_processor, det_model = load_det_processor(), load_det_model()
        rec_model, rec_processor = load_rec_model(), load_rec_processor()
        
        # Run OCR on the image
        predictions = run_ocr([image], [['en']], det_model, det_processor, rec_model, rec_processor)
        
        # Extract text from predictions
        text_lines = [text_line.text_lines for text_line in predictions]
        extracted_text = " ".join([line.text for line in text_lines[0]])
        
        return extracted_text
    except FileNotFoundError:
        return f"Error: File not found at {image_path}"
    except Exception as e:
        return f"Error: Could not process the image. {str(e)}"

# Example usage
if __name__ == "__main__":
    image_path = "path_to_image_or_pdf"  # Replace with the actual path to your image/PDF
    result = get_image_text_suryaocr(image_path)
    print(result)
