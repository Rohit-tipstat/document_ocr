import easyocr
from pdf2image import convert_from_path
import numpy as np
from PIL import Image
import os

def load_file(input_path):
    """Load a file and convert to a list of image arrays."""
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

def save_text_to_file(text, output_file):
    """Save the extracted text to a file."""
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(text)

# Main function to handle OCR
def main(input_path, output_dir):
    try:
        # Extract the base name of the input file and change the extension to .txt
        base_name = os.path.splitext(os.path.basename(input_path))[0]
        output_file = os.path.join(output_dir, f"{base_name}_extracted.txt")
        
        image_arrays = load_file(input_path)  # Load input file as images
        extracted_text = extract_text_from_images(image_arrays)  # Extract text
        save_text_to_file(extracted_text, output_file)  # Save the text
        print(f"Text extraction complete. Check the output file: {output_file}")
    except ValueError as e:
        print(e)

# Input and output paths
input_path = "/root/images/12th (extract.me)/Document_2_App_1(add_comment) (5).pdf"
output_dir = '/home/rohit/document_OCR/data/ocr_datastorage'  # Output directory for the extracted text

# Execute the main function
main(input_path, output_dir)
