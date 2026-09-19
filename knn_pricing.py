"""
Jungle Market - KNN Pricing Module
Predicts a fair market price range for a new product by comparing it against
similar products in the seed dataset (sourced from Tribes India / TRIFED).
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder


class PricingModel:
    def __init__(self, dataset_path, k=5):
        self.k = k
        self.df = pd.read_excel(dataset_path)
        self.df = self.df.dropna(subset=["effective_price"]).reset_index(drop=True)
        self._fit_encoders()

    def _fit_encoders(self):
        self.cat_features = ["category", "material"]
        self.df[self.cat_features] = self.df[self.cat_features].fillna("unknown")

        self.encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        encoded = self.encoder.fit_transform(self.df[self.cat_features])

        self.scaler = MinMaxScaler()
        price_scaled = self.scaler.fit_transform(self.df[["effective_price"]])

        # category/material weighted higher than raw price similarity
        self.feature_matrix = np.hstack([encoded * 3.0, price_scaled * 0.5])

    def predict_price(self, category, material, reference_price_guess=None):
        """
        Predict a fair price range for a new product.
        reference_price_guess: optional artisan's own price idea, used only
        to help pick a more relevant neighborhood when many matches exist.
        """
        query_df = pd.DataFrame([{"category": category, "material": material}])
        encoded_query = self.encoder.transform(query_df[self.cat_features])

        guess = reference_price_guess if reference_price_guess else self.df["effective_price"].median()
        price_scaled_query = self.scaler.transform([[guess]])

        query_vector = np.hstack([encoded_query * 3.0, price_scaled_query * 0.5])

        distances = np.linalg.norm(self.feature_matrix - query_vector, axis=1)
        nearest_idx = np.argsort(distances)[: self.k]
        neighbors = self.df.iloc[nearest_idx]

        weights = 1 / (distances[nearest_idx] + 0.01)
        suggested_price = np.average(neighbors["effective_price"], weights=weights)

        return {
            "suggested_price": round(float(suggested_price), 2),
            "price_range_min": round(float(neighbors["effective_price"].min()), 2),
            "price_range_max": round(float(neighbors["effective_price"].max()), 2),
            "num_neighbors_used": len(neighbors),
            "neighbor_products": neighbors["product_name"].tolist(),
        }

    def apply_cost_floor(self, prediction, material_cost, labor_cost, margin_pct=0.20):
        """Ensures the suggested price never falls below what the artisan needs to earn."""
        min_viable_price = (material_cost + labor_cost) * (1 + margin_pct)
        suggested = prediction["suggested_price"]

        if suggested < min_viable_price:
            prediction["final_price"] = round(min_viable_price, 2)
            prediction["flag"] = "cost_protected"
        elif suggested > min_viable_price * 5:
            prediction["final_price"] = suggested
            prediction["flag"] = "review_needed"
        else:
            prediction["final_price"] = suggested
            prediction["flag"] = "market_confirmed"

        prediction["min_viable_price"] = round(min_viable_price, 2)
        return prediction


if __name__ == "__main__":
    import sys
    dataset = sys.argv[1] if len(sys.argv) > 1 else "../../data/raw/jungle_market_seed_dataset_v2.xlsx"

    model = PricingModel(dataset)
    result = model.predict_price(category="Necklace", material="Dokra Brass")
    result = model.apply_cost_floor(result, material_cost=150, labor_cost=200)
    print(result)
