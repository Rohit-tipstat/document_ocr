import easyocr
from pdf2image import convert_from_path
import numpy as np
from PIL import Image
import os

def load_file(input_path):
    """Load a file and convert it to a list of image arrays."""
    if input_path.lower().endswith('.pdf'):
        print("Converting PDF to images...")
        images = convert_from_path(input_path)
        return [np.array(image) for image in images]  # Convert each image to numpy array
    elif input_path.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp')):
        print("Loading image...")
        image = Image.open(input_path)
        return [np.array(image)]  # Return a list with a single numpy array
    else:
        raise ValueError("Unsupported file type. Please provide a PDF or an image file.")

def extract_text_from_images(image_arrays):
    """Extract text using EasyOCR from a list of image arrays."""
    reader = easyocr.Reader(['en'])  # Initialize EasyOCR with English language
    extracted_text = []

    for i, image_array in enumerate(image_arrays):
        print(f"Processing page {i + 1}...")
        result = reader.readtext(image_array)  # Extract text
        page_text = '\n'.join([text[1] for text in result])
        extracted_text.append(f"--- Page {i + 1} ---\n{page_text}")

    return "\n\n".join(extracted_text)

# Main function to handle OCR
def main(input_path):
    try:
        # Load input file as images
        image_arrays = load_file(input_path)  
        
        # Extract text from images
        extracted_text = extract_text_from_images(image_arrays)  
        
        # Return the extracted text
        return extracted_text
    except ValueError as e:
        print(e)
        return None

# Input path for the document (could be a PDF or image file)
input_path = "/root/images/12th (extract.me)/Document_2_App_1(add_comment) (5).pdf"

# Execute the main function and print the extracted text
extracted_text = main(input_path)

# Output the extracted text or handle as needed
if extracted_text:
    print(extracted_text)
