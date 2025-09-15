from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
import os
import shutil
import pickle
from utils import process_image, process_video

# Create FastAPI app
app = FastAPI(title="Head Pose Estimation API")

# Load trained model
with open('svr_model.pkl', 'rb') as f:
        model = pickle.load(f)

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    """
    Single endpoint for both image and video.
    - Detects file type by extension
    - Runs corresponding processing
    - Returns output file path
    """
    # Save uploaded file temporarily
    temp_file = f"temp_{file.filename}"
    with open(temp_file, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    filename = file.filename.lower()

    # Detect file type by extension
    if filename.endswith((".jpg", ".jpeg", ".png")):
        output_path = "output_image.jpg"
        processed_path, _ = process_image(temp_file, model, output_path)

    elif filename.endswith((".mp4", ".avi", ".mov", ".mkv")):
        output_path = "output_video.mp4"
        processed_path= process_video(temp_file, model, output_path)

    else:
        os.remove(temp_file)
        return {"error": "Unsupported file type. Use .jpg/.png for images or .mp4/.avi/.mov for videos."}

    os.remove(temp_file)

    if processed_path:
        return FileResponse(processed_path, filename=os.path.basename(processed_path))
    else:
        return {"error": "No face detected"}
