from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
import shutil
import os

# Import your Phase 1 scripts
# (Adjust module and function names based on your exact file code)
import image_enhancer
import coin_reference
import knn_pricing

app = FastAPI(title="Jungle Market AI Engine")

# Allow request calls from your PWA frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def home():
    return {"status": "Jungle Market AI API is active"}

@app.post("/process-handicraft/")
async def process_handicraft(
    category: str = Form(...),
    file: UploadFile = File(...)
):
    # Save uploaded artisan image temporarily
    temp_path = f"temp_{file.filename}"
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        # 1. Enhance Image
        enhanced_path = image_enhancer.enhance(temp_path)

        # 2. Estimate Dimensions (Height x Width)
        dimensions = coin_reference.get_dimensions(enhanced_path)

        # 3. Predict Fair Price
        recommended_price = knn_pricing.predict(
            category=category, 
            dimensions=dimensions
        )

        return {
            "success": True,
            "category": category,
            "dimensions": dimensions,
            "recommended_price": recommended_price
        }

    except Exception as e:
        return {"success": False, "error": str(e)}

    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
          
