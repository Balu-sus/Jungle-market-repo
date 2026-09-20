import os
import shutil
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

import coin_reference
import product_classifier
from db_interface import DummyRepository

app = FastAPI(title="Jungle Market - Artisan API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize database repository abstraction
db_repo = DummyRepository()

if os.path.exists("public"):
    app.mount("/public", StaticFiles(directory="public"), name="public")

@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    index_path = os.path.join("public", "index.html")
    if os.path.exists(index_path):
        with open(index_path, "r", encoding="utf-8") as f:
            return f.read()
    return "<h2>index.html was not found inside the public/ directory!</h2>"

@app.post("/process-handicraft/")
async def process_handicraft(
    category: str = Form("auto"),
    coin_type: str = Form("5_rupee"),
    file: UploadFile = File(...)
):
    temp_path = f"temp_{file.filename}"
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        clean_category = category.lower().strip()

        # 1. Zero-shot CLIP auto-detection if category is "auto" or unlisted
        if clean_category in ["auto", "", "undefined"]:
            detected_cat, confidence = product_classifier.identify_unknown_product(temp_path)
            clean_category = detected_cat

        # 2. Extract dimensions and wearable fit sizing
        dims = coin_reference.process_image(
            temp_path, 
            category=clean_category, 
            coin_type=coin_type
        )

        response_payload = {
            "success": True,
            "category": clean_category,
            "dimensions": dims,
            "suggested_price": 500
        }

        # 3. Save to database via repository pattern
        await db_repo.save_product_analysis(response_payload)

        return response_payload

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
