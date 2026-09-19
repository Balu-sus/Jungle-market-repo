import os
# Import your modules
# (Adjust module names based on your actual file names)
from image_enhancer import enhance_image
from coin_reference import estimate_dimensions
from knn_pricing import predict_price

def run_test():
    sample_image = "test_craft.jpg"
    
    if not os.path.exists(sample_image):
        print(f"Please place a sample image named '{sample_image}' in the folder.")
        return

    print("--- 1. Testing Image Enhancer ---")
    enhanced_img = enhance_image(sample_image)
    print("Image enhanced successfully.")

    print("\n--- 2. Testing Dimension Estimator ---")
    height, width = estimate_dimensions(enhanced_img)
    print(f"Calculated Dimensions: {height} cm x {width} cm")

    print("\n--- 3. Testing KNN Pricing Model ---")
    suggested_price = predict_price(category="Woodcraft", height=height, width=width)
    print(f"Suggested Price: ₹{suggested_price}")

if __name__ == "__main__":
    run_test()
  
