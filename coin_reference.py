import math
import cv2
import numpy as np
from PIL import Image
from rembg import remove, new_session

# Force CPU execution to suppress CUDA warning logs
cpu_session = new_session(model_name="u2net", providers=['CPUExecutionProvider'])


def preprocess_and_remove_bg(image_path: str):
    """
    Strips complex backgrounds, loose beads, and fabric textures using rembg,
    returning an RGBA numpy image and a binary alpha mask.
    """
    input_img = Image.open(image_path)
    output_img = remove(input_img, session=cpu_session)

    img_np = np.array(output_img)
    alpha = img_np[:, :, 3]
    _, clean_mask = cv2.threshold(alpha, 10, 255, cv2.THRESH_BINARY)

    return img_np, clean_mask


def detect_coin_circle(gray_img: np.ndarray):
    """
    Specifically detects circular coins via Hough Circle Transform
    to prevent confusion with dangling tassels or extra accessories (e.g., earrings).
    """
    blurred = cv2.medianBlur(gray_img, 5)

    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=50,
        param1=50,
        param2=30,
        minRadius=15,
        maxRadius=80,
    )

    if circles is not None:
        circles = np.uint16(np.around(circles))
        return circles[0][0][2]  # Returns detected coin radius in pixels
    return None


def calculate_wearable_fit(height_cm: float, width_cm: float):
    """
    Estimates total necklace chain circumference from bounding dimensions
    using Ramanujan's ellipse perimeter approximation and maps it to standard fit guidelines.
    """
    a = width_cm / 2.0
    b = height_cm / 2.0

    # Ramanujan perimeter approximation: P ≈ π * (a + b) * (1 + (3*h) / (10 + sqrt(4 - 3*h)))
    if (a + b) > 0:
        h_val = ((a - b) ** 2) / ((a + b) ** 2)
        circumference_cm = (
            math.pi * (a + b) * (1 + (3 * h_val) / (10 + math.sqrt(4 - 3 * h_val)))
        )
    else:
        circumference_cm = 35.0

    total_length_inches = round(circumference_cm / 2.54, 1)

    # Standard Jewelry Sizing Rules
    if total_length_inches < 14.0:
        fit_label = 'XS - Choker / Kids (< 14")'
        target_fit = 'Fits tightly around the throat / Kids'
    elif 14.0 <= total_length_inches < 18.0:
        fit_label = 'S/M - Princess Fit (16"-18")'
        target_fit = 'Rests gracefully on the collarbone (Most Women)'
    elif 18.0 <= total_length_inches < 22.0:
        fit_label = 'L - Matinee Fit (18"-20")'
        target_fit = "Rests just below collarbone / Men's Standard"
    else:
        fit_label = 'XL - Opera / Long Fit (24"+)'
        target_fit = 'Hangs low over bust / Statement Piece'

    return {
        'height_cm': height_cm,
        'width_cm': width_cm,
        'size_type': 'WEARABLE FIT',
        'display_primary': fit_label,
        'display_secondary': f'Full Loop: ~{total_length_inches}" ({target_fit})',
    }


def process_image(
    image_path: str, category: str = 'basket', coin_type: str = '5_rupee'
):
    """
    Main entry point called by FastAPI. Strips background, scales pixels to cm
    using coin diameter standards, and returns appropriate dynamic payloads.
    """
    category = category.lower().strip()

    # 1. Background removal
    img_np, clean_mask = preprocess_and_remove_bg(image_path)

    # 2. RBI Coin Diameter Lookup (cm)
    coin_diameters = {
        '1_rupee': 2.1,
        '2_rupee': 2.3,
        '5_rupee': 2.3,
        '10_rupee': 2.7,
    }
    real_coin_cm = coin_diameters.get(coin_type, 2.3)

    # 3. Contour Detection
    contours, _ = cv2.findContours(
        clean_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if len(contours) < 2:
        # Fallback dimensions if unisolated or touching
        height_cm = 15.0
        width_cm = 12.0
    else:
        # Sort by contour area descending (Largest = Main Product)
        sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)
        product_contour = sorted_contours[0]

        # Extract bounding box for product
        _, _, w, h = cv2.boundingRect(product_contour)

        # Detect coin radius via Hough Circles
        gray = cv2.cvtColor(img_np[:, :, :3], cv2.COLOR_RGBA2GRAY)
        detected_radius = detect_coin_circle(gray)

        if detected_radius:
            coin_px = detected_radius * 2
        else:
            # Fallback to 2nd largest contour's enclosing circle
            (_, _), coin_radius = cv2.minEnclosingCircle(sorted_contours[1])
            coin_px = coin_radius * 2

        # 4. Physical Ratio Scaling
        if coin_px > 0:
            px_per_cm = coin_px / real_coin_cm
            height_cm = round(h / px_per_cm, 1)
            width_cm = round(w / px_per_cm, 1)
        else:
            height_cm = 15.0
            width_cm = 12.0

    # 5. Category-based output routing
    if category in ['necklace', 'bracelet']:
        return calculate_wearable_fit(height_cm, width_cm)
    else:
        return {
            'height_cm': height_cm,
            'width_cm': width_cm,
            'size_type': 'BASKET DIMENSIONS',
            'display_primary': f'{height_cm} cm (H) x {width_cm} cm (W)',
            'display_secondary': f'Approx. {round(height_cm / 2.54, 1)}" x {round(width_cm / 2.54, 1)}"',
        }