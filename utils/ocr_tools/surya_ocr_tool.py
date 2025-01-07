from PIL import Image
from surya.ocr import run_ocr
from surya.model.detection.model import load_model as load_det_model, load_processor as load_det_processor
from surya.model.recognition.model import load_model as load_rec_model
from surya.model.recognition.processor import load_processor as load_rec_processor
import os

from pdf2image import convert_from_pathsettings


async def get_image_text(image_path: str):
    """ Extracts text from the given image using OCR models.
    :param image_path: Path to the image file. return: Extracted text as a string.
    """
    # Open the image file
    try:
        if image_path.lower().endswith(".pdf"):
            print("Converting PDF {image_path} to  images...")
            images = convert_from_path(image_path)
            print("Converted PDF to image")
            image = images[0]
        else:
            image=Image.open(image_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found at {image_path}")
    except Exception as e:
        raise Exception(f"Could not open Image/PDF.{e}")

    langs = ["en"] # Replace with your languages - optional but recommended
    # Load detection and recognition models and processors
    det_processor, det_model = load_det_processor(), load_det_model()
    rec_model, rec_processor = load_rec_model(), load_rec_processor()
    # Run OCR on the image
    predictions = run_ocr([image], [langs], det_model, det_processor, rec_model, rec_processor)
    # Extract text lines from predictions
    text_lines = [text_line.text_lines for text_line in predictions]
    ocr = " ".join([line.text for line in text_lines[0]])
    return ocr

def save_text_to_file(text: str, output_path: str):
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(text)
        print("Extracted text saved to :{output_path}")


if __name__ == "__main__":
    import asyncio
    import sys
    # Check if the image path is provided
    if len(sys.argv) < 2:
        print("Usage: proper execution code")
        sys.exit(1)
    # Get the image path from the command-line arguments
    image_path = sys.argv[1]
    print("Image path -> ", image_path)
    # Run the OCR function
    try:
        ocr_result = asyncio.run(get_image_text(image_path))
        print("Extracted Text:")
        print(ocr_result)
        base_name, _ = os.path.splitext(image_path)
        output_file = f"{base_name}.txt"
        save_text_to_file(ocr_result, output_file)
    except Exception as e:
        print(f"Error: {e}")
