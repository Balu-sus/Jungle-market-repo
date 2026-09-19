"""
Jungle Market - Dimension Estimation Module
Estimates real-world product dimensions using a coin as a scale reference.

Method: detect the coin (a circle) via Hough Circle Transform, detect the
product's outline via contour detection, then convert pixel measurements
to real-world centimeters using the coin's known diameter.
"""
import cv2
import numpy as np

# Common Indian coin diameters in cm (add more as needed)
COIN_DIAMETERS_CM = {
    "5_rupee": 2.3,
    "10_rupee": 2.7,
    "2_rupee": 2.5,
    "1_rupee": 2.1,
}


def estimate_dimensions(image_path, coin_type="5_rupee", debug=False):
    """
    Estimate a product's real-world width and height from a photo that
    includes a reference coin next to the product.

    Returns a dict with width_cm, height_cm, confidence, and a message.
    """
    coin_diameter_cm = COIN_DIAMETERS_CM.get(coin_type, 2.3)

    img = cv2.imread(image_path)
    if img is None:
        return {"error": f"Could not read image at {image_path}"}

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)

    # --- Step 1: detect the coin (circle) ---
    circles = cv2.HoughCircles(
        blurred, cv2.HOUGH_GRADIENT, dp=1.2, minDist=100,
        param1=50, param2=30, minRadius=15, maxRadius=200
    )

    if circles is None:
        return {
            "width_cm": None, "height_cm": None,
            "confidence": "low",
            "message": "Coin not detected. Retake photo with the coin clearly visible and well-lit.",
        }

    circles = np.round(circles[0, :]).astype("int")
    # If multiple circles detected, this is ambiguous -> lower confidence
    multi_circle_flag = circles.shape[0] > 1
    coin_radius_px = circles[0][2]
    pixel_to_cm_ratio = coin_diameter_cm / (2 * coin_radius_px)

    # --- Step 2: detect the product's outline ---
    edges = cv2.Canny(gray, 50, 150)
    edges = cv2.dilate(edges, None, iterations=1)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return {
            "width_cm": None, "height_cm": None,
            "confidence": "low",
            "message": "Product outline not detected. Try a plainer background.",
        }

    # Exclude the coin's own contour by filtering out anything near the coin's location/size
    cx, cy, cr = circles[0]
    candidate_contours = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        area = cv2.contourArea(c)
        # skip tiny noise contours and the coin itself
        if area < 500:
            continue
        center_dist = np.hypot((x + w / 2) - cx, (y + h / 2) - cy)
        if center_dist < cr * 1.5 and abs(w - 2 * cr) < cr and abs(h - 2 * cr) < cr:
            continue  # likely the coin itself
        candidate_contours.append(c)

    if not candidate_contours:
        return {
            "width_cm": None, "height_cm": None,
            "confidence": "low",
            "message": "Could not distinguish product from coin. Ensure product is clearly larger and separated from the coin.",
        }

    largest_contour = max(candidate_contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)

    width_cm = round(w * pixel_to_cm_ratio, 1)
    height_cm = round(h * pixel_to_cm_ratio, 1)

    confidence = "medium" if multi_circle_flag else "high"

    result = {
        "width_cm": width_cm,
        "height_cm": height_cm,
        "confidence": confidence,
        "message": "Estimated using coin reference. Please confirm or adjust.",
    }

    if debug:
        debug_img = img.copy()
        cv2.circle(debug_img, (cx, cy), cr, (0, 255, 0), 2)
        cv2.rectangle(debug_img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.imwrite("debug_dimension_output.jpg", debug_img)
        result["debug_image"] = "debug_dimension_output.jpg"

    return result


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python coin_reference_estimator.py <image_path> [coin_type]")
        sys.exit(1)
    path = sys.argv[1]
    coin = sys.argv[2] if len(sys.argv) > 2 else "5_rupee"
    print(estimate_dimensions(path, coin_type=coin, debug=True))
