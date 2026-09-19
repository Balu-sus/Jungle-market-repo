"""
Jungle Market - Image Enhancement Module
Cleans up artisan-submitted product photos for e-commerce presentation:
auto white balance, contrast enhancement (CLAHE), and denoising.
"""
import cv2
import numpy as np


def auto_white_balance(img):
    """Corrects color cast using the LAB color space gray-world assumption."""
    result = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    avg_a = np.average(result[:, :, 1])
    avg_b = np.average(result[:, :, 2])
    result[:, :, 1] -= ((avg_a - 128) * (result[:, :, 0] / 255.0) * 1.1)
    result[:, :, 2] -= ((avg_b - 128) * (result[:, :, 0] / 255.0) * 1.1)
    result = np.clip(result, 0, 255).astype(np.uint8)
    return cv2.cvtColor(result, cv2.COLOR_LAB2BGR)


def enhance_contrast(img):
    """Applies CLAHE (adaptive contrast enhancement) on the lightness channel only."""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)


def denoise(img):
    """Removes compression/sensor noise typical of mobile camera photos."""
    return cv2.fastNlMeansDenoisingColored(img, None, 5, 5, 7, 21)


def enhance_product_photo(image_path, output_path=None):
    """Full enhancement pipeline: white balance -> contrast -> denoise."""
    img = cv2.imread(image_path)
    if img is None:
        return {"error": f"Could not read image at {image_path}"}

    img = auto_white_balance(img)
    img = enhance_contrast(img)
    img = denoise(img)

    if output_path:
        cv2.imwrite(output_path, img)

    return {"status": "success", "output_path": output_path}


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python image_enhancer.py <image_path> [output_path]")
        sys.exit(1)
    path = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) > 2 else "enhanced_output.jpg"
    print(enhance_product_photo(path, out))
