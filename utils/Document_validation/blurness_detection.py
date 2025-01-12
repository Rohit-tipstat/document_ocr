import cv2
import numpy as np
from skimage.feature import canny
from skimage.filters import gaussian
from pdf2image import convert_from_path
import os
import shutil


def compute_laplacian_variance(image: np.ndarray) -> float:
    """Compute the Laplacian variance of the image."""
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    laplacian_var = cv2.Laplacian(gray_image, cv2.CV_64F).var()
    return laplacian_var


def compute_edge_density(image: np.ndarray) -> float:
    """Compute the edge density of the image."""
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = canny(gray_image)
    edge_density = np.sum(edges) / edges.size * 100
    return edge_density


def compute_noise_level(image: np.ndarray) -> float:
    """Estimate the noise level of the image."""
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred_image = gaussian(gray_image, sigma=1)
    noise = gray_image - blurred_image
    noise_level = np.std(noise)
    return noise_level


def classify_image_quality(laplacian_var: float, edge_density: float, noise_level: float, thresholds: dict, weights: dict) -> str:
    """
    Classify the image quality based on the Laplacian variance, edge density, and noise level.
    
    Returns:
        "Not Blurry" or "Blurry" based on the calculated score.
    """
    score = (
        weights["laplacian"] * (laplacian_var / thresholds["laplacian"]) +
        weights["edge_density"] * (edge_density / thresholds["edge_density"]) +
        weights["noise_level"] * (noise_level / thresholds["noise_level"])
    )
    return "Not Blurry" if score > 1 else "Blurry"


def convert_pdf_to_image(pdf_path: str, temp_dir: str) -> np.ndarray:
    """
    Convert the first page of a PDF to an image.
    
    Args:
        pdf_path: Path to the PDF file.
        temp_dir: Temporary directory for storing intermediate files.

    Returns:
        A NumPy array representing the first page of the PDF as an image.
    """
    pages = convert_from_path(pdf_path, dpi=300, output_folder=temp_dir)
    if pages:
        return np.array(pages[0])
    raise ValueError("Could not extract any pages from the PDF.")


def process_image(file_path: str) -> dict:
    """
    Process an image or PDF file and compute quality metrics.
    
    Args:
        file_path: Path to the input image or PDF file.

    Returns:
        A dictionary with computed metrics and classification results.
    """
    temp_dir = None
    try:
        if file_path.lower().endswith(".pdf"):
            temp_dir = "./temp_pdf_conversion"
            os.makedirs(temp_dir, exist_ok=True)
            image = convert_pdf_to_image(file_path, temp_dir)
            thresholds = {
                "laplacian": 190,
                "edge_density": 2.0,
                "noise_level": 50
            }
            weights = {
                "laplacian": 0.8,
                "edge_density": 0.1,
                "noise_level": 0.1
            }
        else:
            image = cv2.imread(file_path)
            if image is None:
                raise ValueError("Invalid image file provided.")
            thresholds = {
                "laplacian": 1000,
                "edge_density": 12,
                "noise_level": 5000,
            }
            weights = {
                "laplacian": 0.5,
                "edge_density": 0.3,
                "noise_level": 0.2,
            }

        laplacian_var = compute_laplacian_variance(image)
        edge_density = compute_edge_density(image)
        noise_level = compute_noise_level(image)

        classification = classify_image_quality(laplacian_var, edge_density, noise_level, thresholds, weights)
        return {
            "laplacian_var": laplacian_var,
            "edge_density": edge_density,
            "noise_level": noise_level,
            "classification": classification,
        }
    finally:
        if temp_dir and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)


def blur_detection(file_path: str) -> bool:
    """
    Main function to detect if a file is blurry.
    
    Args:
        file_path: Path to the input file (image or PDF).

    Returns:
        True if the file is classified as "Blurry", otherwise False.
    """
    result = process_image(file_path)
    #print(f"Image Quality Analysis: {result}")
    if result['classification'] == 'Not Blurry':
        return False
    else:
        return True


if __name__ == "__main__":
    # Provide the path to an image or PDF
    file_path = "/home/rohit/document_OCR/images/35140083-41a2-4284-b8a0-e11e42af2600.jpg"
    print(blur_detection(file_path))