"""
Image Classification API using Pre-trained MobileNet Model
Model: google/mobilenet_v2_1.0_224 (optimized for low memory)
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from huggingface_hub import InferenceClient
from PIL import Image
import io
import uvicorn
from pathlib import Path
import os

# Determine whether to use Hugging Face Inference API before importing heavy libs
USE_HF_API = os.getenv("USE_HF_API", "").lower() == "true"
HF_API_TOKEN = os.getenv("HF_API_TOKEN", "")
HF_MODEL_ID = os.getenv("HF_MODEL_ID", "openai/clip-vit-base-patch32")

if not USE_HF_API:
    # Import heavy deps only when doing local inference
    from transformers import pipeline
    import torch

# Initialize FastAPI app
app = FastAPI(
    title="Image Classification API",
    description="AI-powered image classification using Google's Vision Transformer",
    version="1.0.0"
)

# Add CORS middleware to allow requests from any origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variable for the model
classifier = None
inference_client = None

# Fixed list of Philippine herbal plants (50 labels)
HERBAL_LABELS = [
    "Lagundi (Vitex negundo)",
    "Sambong (Blumea balsamifera)",
    "Tsaang gubat (Ehretia microphylla)",
    "Niyog-niyogan (Quisqualis indica)",
    "Akapulko (Senna alata)",
    "Bayabas / Guava (Psidium guajava)",
    "Yerba buena (Clinopodium douglasii)",
    "Ulasimang-bato / Pansit-pansitan (Peperomia pellucida)",
    "Bawang / Garlic (Allium sativum)",
    "Ampalaya / Bitter gourd (Momordica charantia)",
    "Banaba (Lagerstroemia speciosa)",
    "Tawa-tawa / Asthma weed (Euphorbia hirta)",
    "Oregano / Cuban oregano (Plectranthus amboinicus)",
    "Tanglad / Lemongrass (Cymbopogon citratus)",
    "Luya / Ginger (Zingiber officinale)",
    "Luyang dilaw / Turmeric (Curcuma longa)",
    "Pandan (Pandanus amaryllifolius)",
    "Malunggay / Moringa (Moringa oleifera)",
    "Makabuhay (Tinospora crispa)",
    "Alagaw (Premna odorata)",
    "Gugo (Entada phaseoloides)",
    "Balbas pusa (Orthosiphon aristatus)",
    "Guyabano / Soursop (Annona muricata)",
    "Duhat / Java plum (Syzygium cumini)",
    "Ikmo / Betel (Piper betle)",
    "Kamias / Bilimbi (Averrhoa bilimbi)",
    "Makahiya / Sensitive plant (Mimosa pudica)",
    "Kataka-taka (Bryophyllum pinnatum)",
    "Kakawate / Madre de cacao (Gliricidia sepium)",
    "Mayana (Coleus scutellarioides)",
    "Sabila / Aloe vera (Aloe vera)",
    "Calamansi (Citrofortunella microcarpa)",
    "Papaya leaves (Carica papaya)",
    "Mango leaves (Mangifera indica)",
    "Guyabano leaves (Annona muricata leaves)",
    "Eucalyptus (Eucalyptus globulus)",
    "Citronella (Cymbopogon nardus)",
    "Gotu kola / Takip-kuhol (Centella asiatica)",
    "Pandakaki (Tabernaemontana pandacaqui)",
    "Anislag (Premna serratifolia)",
    "Insulin plant (Costus igneus)",
    "Tsaang kalabaw (Ehretia buxifolia)",
    "Atsuete / Annatto (Bixa orellana)",
    "Noni / Indian mulberry (Morinda citrifolia)",
    "Katuray (Sesbania grandiflora)",
    "Betel leaves (Piper betle leaves)",
    "Sampaguita leaves (Jasminum sambac leaves)",
    "Ilang-ilang leaves (Cananga odorata leaves)",
    "Bay leaf / Laurel (Laurus nobilis)",
    "Lagikway (Abelmoschus manihot)"
]

@app.on_event("startup")
async def load_model():
    """Load the pre-trained model on startup"""
    global classifier
    # For local inference, reduce threads to fit low-memory environments (e.g., Render 512Mi)
    if not USE_HF_API:
        os.environ.setdefault("OMP_NUM_THREADS", "1")
        os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
        os.environ.setdefault("MKL_NUM_THREADS", "1")
        try:
            import torch as _torch
            _torch.set_num_threads(1)
        except Exception:
            pass

    if USE_HF_API:
        # Use Hugging Face Inference API (no local model load)
        global inference_client
        print("🔄 Using Hugging Face Inference API for zero-shot image classification...")
        if not HF_API_TOKEN:
            print("⚠️  HF_API_TOKEN not set; requests will fail until token is provided.")
        inference_client = InferenceClient(token=HF_API_TOKEN)
        print("✅ Inference client initialized.")
    else:
        print("🔄 Loading zero-shot image classification model...")
        print("⏳ This may take a couple of minutes on first run (downloading model)...")
        classifier = pipeline("zero-shot-image-classification", model=HF_MODEL_ID)
        print("✅ Model loaded successfully!")
        print("🚀 API is ready to classify images!")


@app.get("/")
@app.head("/")
async def root():
    """Serve the custom UI"""
    html_path = Path(__file__).parent / "index.html"
    if html_path.exists():
        return FileResponse(html_path)
    else:
        # Fallback to API info if HTML not found
        return {
            "message": "Image Classification API",
            "model": f"{HF_MODEL_ID} (zero-shot)" if not USE_HF_API else f"{HF_MODEL_ID} via HF Inference API",
            "status": "running",
            "endpoints": {
                "/classify": "POST - Upload an image to classify",
                "/health": "GET - Check API health status",
                "/docs": "GET - Interactive API documentation"
            }
        }


@app.get("/api")
async def api_info():
    """API information endpoint"""
    return {
        "message": "Image Classification API",
        "model": f"{HF_MODEL_ID} (zero-shot)" if not USE_HF_API else f"{HF_MODEL_ID} via HF Inference API",
        "status": "running",
        "endpoints": {
            "/classify": "POST - Upload an image to classify",
            "/health": "GET - Check API health status",
            "/docs": "GET - Interactive API documentation",
            "/labels": "GET - List of herbal plant labels"
        }
    }


@app.get("/labels")
async def get_labels():
    """Return the full list of herbal plant labels and count"""
    return {
        "count": len(HERBAL_LABELS),
        "labels": HERBAL_LABELS
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": classifier is not None
    }


@app.post("/classify")
async def classify_image(file: UploadFile = File(...)):
    """
    Classify an uploaded image
    
    Args:
        file: Image file (JPG, PNG, etc.)
    
    Returns:
        JSON with top predictions and confidence scores
    """
    
    # Check if model/API is available
    if not USE_HF_API and classifier is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet. Please wait...")
    if USE_HF_API and inference_client is None:
        raise HTTPException(status_code=503, detail="Inference API not initialized. Set USE_HF_API=true and HF_API_TOKEN.")
    
    # Validate file type
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        # Read image file
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes))
        
        # Convert to RGB if necessary (handles PNG with alpha channel, etc.)
        if image.mode != "RGB":
            image = image.convert("RGB")
        
        # Classify using either local model or HF Inference API
        if USE_HF_API:
            # Use HF Inference API (returns list of {label, score})
            results = inference_client.zero_shot_image_classification(
                image=image,
                labels=HERBAL_LABELS,
                model=HF_MODEL_ID,
            )
            predictions = sorted(results, key=lambda x: x.get("score", 0), reverse=True)[:5]
        else:
            predictions = classifier(image, candidate_labels=HERBAL_LABELS, multi_label=False)[:5]
        
        # Format response
        return {
            "success": True,
            "filename": file.filename,
            "predictions": [
                {
                    "label": pred["label"],
                    "confidence": round(pred["score"] * 100, 2),  # Convert to percentage
                    "score": round(pred["score"], 4)
                }
                for pred in predictions
            ],
            "top_prediction": {
                "label": predictions[0]["label"],
                "confidence": round(predictions[0]["score"] * 100, 2)
            }
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")


@app.post("/classify-top")
async def classify_image_top_only(file: UploadFile = File(...)):
    """
    Classify an uploaded image and return only the top prediction
    
    Args:
        file: Image file (JPG, PNG, etc.)
    
    Returns:
        JSON with only the top prediction
    """
    
    if classifier is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet. Please wait...")
    
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes))
        
        if image.mode != "RGB":
            image = image.convert("RGB")
        
        predictions = classifier(image, candidate_labels=HERBAL_LABELS, multi_label=False)
        
        return {
            "success": True,
            "filename": file.filename,
            "label": predictions[0]["label"],
            "confidence": round(predictions[0]["score"] * 100, 2),
            "score": round(predictions[0]["score"], 4)
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")


if __name__ == "__main__":
    print("=" * 60)
    print("🤖 Image Classification API")
    print("=" * 60)
    print(f"📦 Model: {HF_MODEL_ID} {'via HF Inference API' if USE_HF_API else '(zero-shot)'}")
    print("🌐 Custom UI: http://localhost:8000")
    print("📚 API Docs: http://localhost:8000/docs")
    print("🔧 API Info: http://localhost:8000/api")
    print("=" * 60)
    
    uvicorn.run(app, host="0.0.0.0", port=8000)
