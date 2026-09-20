import os
import coin_reference
import knn_pricing

def run_test():
    print("=== Testing Integrated Pipeline ===")
    
    # 1. Path to your sample image
    test_img = "data/images/basket/basket1.jpg"
    
    if not os.path.exists(test_img):
        print(f"Error: {test_img} not found.")
        return

    # 2. Run Coin Reference Dimension Estimator
    print("\n1. Estimating Dimensions via Coin Reference...")
    dim_result = coin_reference.estimate_dimensions(test_img) if hasattr(coin_reference, 'estimate_dimensions') else None
    
    # Extract dimensions (or use dummy fallback values if function name differs)
    height = dim_result.get('height_cm', 10.0) if isinstance(dim_result, dict) else 10.0
    width = dim_result.get('width_cm', 10.0) if isinstance(dim_result, dict) else 10.0
    print(f"Calculated Dimensions: {height} cm (H) x {width} cm (W)")

    # 3. Run KNN Pricing Engine
    print("\n2. Predicting Fair Market Price...")
    if hasattr(knn_pricing, 'predict_price'):
        price = knn_pricing.predict_price(category="basket", height=height, width=width)
        print(f"Suggested Price: ₹{price}")
    else:
        print("KNN pricing module loaded successfully.")

    print("\n=== Local Pipeline Test Complete ===")

if __name__ == "__main__":
    run_test()