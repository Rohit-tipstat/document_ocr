import cv2
import numpy as np
from pdf2image import convert_from_path

def convert_pdf_to_image(pdf_path: str, dpi: int = 300) -> np.ndarray:
    """
    Convert the first page of a PDF to an image.

    Args:
        pdf_path (str): Path to the PDF file.
        dpi (int): DPI resolution for the conversion. Default is 300.

    Returns:
        np.ndarray: Image converted from the first page of the PDF.
    """
    pages = convert_from_path(pdf_path, dpi=dpi)
    if pages:
        return np.array(pages[0])  # Convert the first page to a NumPy array
    raise ValueError("Could not extract any pages from the PDF")


def brighten_image(image: np.ndarray, brightness_factor: float = 1.2) -> np.ndarray:
    """
    Brighten an image by scaling its pixel values.

    Args:
        image (np.ndarray): Input image in BGR format (as read by OpenCV or converted from PDF).
        brightness_factor (float): Factor by which to increase brightness. Default is 1.2.

    Returns:
        np.ndarray: Brightened image.
    """
    if not isinstance(image, np.ndarray):
        raise ValueError("Input must be a NumPy array representing the image.")
    
    # Convert image to float32 to avoid overflow during multiplication
    image_float = image.astype(np.float32)
    
    # Scale pixel values by the brightness factor
    brightened_image = image_float * brightness_factor
    
    # Clip pixel values to valid range [0, 255] and convert back to uint8
    brightened_image = np.clip(brightened_image, 0, 255).astype(np.uint8)
    
    return brightened_image


def process_file(file_path: str, brightness_factor: float = 1.2) -> np.ndarray:
    """
    Process a file (image or PDF) to brighten its content.

    Args:
        file_path (str): Path to the input file (image or PDF).
        brightness_factor (float): Factor by which to increase brightness. Default is 1.2.

    Returns:
        np.ndarray: Brightened image.
    """
    if file_path.lower().endswith(".pdf"):
        image = convert_pdf_to_image(file_path)
    else:
        image = cv2.imread(file_path)
        if image is None:
            raise ValueError("Failed to load the image. Please check the file path.")
    
    return brighten_image(image, brightness_factor)


if __name__ == "__main__":
    # Provide the path to an image or PDF file
    file_path = "/home/rohit/document_OCR/images/Data_auxilo/7095364-Piyush_Lalit_deshmukh/10th_Marksheet.pdf"  # Replace with the path to your PDF or image
    brightness_factor = 1.5  # Adjust brightness level if needed

    try:
        brightened_image = process_file(file_path, brightness_factor=brightness_factor)

        # Display the brightened image
        cv2.imshow("Brightened Image", brightened_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # Optionally, save the brightened image
        output_path = "brightened_output.jpg"
        cv2.imwrite(output_path, brightened_image)
        print(f"Brightened image saved to {output_path}")

    except Exception as e:
        print(f"Error: {e}")
