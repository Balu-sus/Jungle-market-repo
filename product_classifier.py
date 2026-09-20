import torch
from PIL import Image
import open_clip

# Load model and preprocessing
model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
tokenizer = open_clip.get_tokenizer('ViT-B-32')

CANDIDATE_LABELS = [
    "necklace", "bracelet", "earrings", 
    "woven basket", "metal tray", "terracotta pot", 
    "wooden sculpture", "fabric wall hanging"
]

def identify_unknown_product(image_path: str, db_categories: list = None):
    """
    Performs zero-shot image classification using CLIP.
    """
    labels = db_categories if db_categories else CANDIDATE_LABELS
    
    image = preprocess(Image.open(image_path)).unsqueeze(0)
    text = tokenizer([f"a photo of a handicraft {label}" for label in labels])

    with torch.no_grad():
        image_features = model.encode_image(image)
        text_features = model.encode_text(text)
        
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)

    top_idx = text_probs[0].argmax().item()
    detected_category = labels[top_idx]
    confidence = float(text_probs[0][top_idx])

    return detected_category, round(confidence, 2)
