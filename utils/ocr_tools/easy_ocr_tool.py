from PIL import Image
from pdf2image import convert_from_path
import easyocr
import os
import numpy as np

def load_file(input_path):
    """Load a file and convert it to a list of PIL Images."""
    try:
        if input_path.lower().endswith('.pdf'):
            print(f"Converting PDF {input_path} to images...")
            images = convert_from_path(input_path)
            print("Converted PDF to images")
            return images  # Return a list of PIL Image objects
        elif input_path.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp')):
            print(f"Loading image {input_path}...")
            image = Image.open(input_path)
            return [image]  # Return a list with a single PIL Image object
        else:
            raise ValueError("Unsupported file type. Please provide a PDF or an image file.")
    except FileNotFoundError:
        raise FileNotFoundError(f"Error: File not found at {input_path}")
    except Exception as e:
        raise Exception(f"Error: Could not load the file. {str(e)}")

def extract_text_from_images(images):
    """Extract text using EasyOCR from a list of PIL Images."""
    try:
        reader = easyocr.Reader(['en'])  # Initialize EasyOCR with English language
        extracted_text = []

        for i, image in enumerate(images):
            print(f"Processing page {i + 1}...")
            image_array = np.array(image)  # Convert PIL Image to NumPy array
            results = reader.readtext(image_array)  # Extract text
            page_text = '\n'.join([text[1] for text in results])
            extracted_text.append(f"--- Page {i + 1} ---\n{page_text}")

        return "\n\n".join(extracted_text)
    except Exception as e:
        raise Exception(f"Error: Could not extract text from images. {str(e)}")

def get_image_text_easyocr(input_path):
    """Extract text from the given image or PDF using EasyOCR.
    
    :param input_path: Path to the image or PDF file.
    :return: Extracted text as a string or error message.
    """
    try:
        # Load file as images
        images = load_file(input_path)
        
        # Extract text from images
        extracted_text = extract_text_from_images(images)
        
        return extracted_text
    except FileNotFoundError as e:
        return str(e)
    except ValueError as e:
        return str(e)
    except Exception as e:
        return f"Error: {str(e)}"

# Example usage
if __name__ == "__main__":
    input_path = "/root/images/12th (extract.me)/Document_2_App_1(add_comment) (5).pdf"  # Replace with your file path
    result = get_image_text_easyocr(input_path)
    print(result)
