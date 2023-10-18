from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, File, UploadFile
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from utils import BrainTumorClassifier, augment_image_cv2
from fastapi import FastAPI
from pathlib import Path
import numpy as np
import cv2

classifier = BrainTumorClassifier()

labels = ['Glioma', 'Meningioma', 'No tumor', 'Pituitary']

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    file_bytes = np.asarray(bytearray(await file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    image = augment_image_cv2(image)

    array = np.expand_dims(image, axis=0) / 255.0
    prediction = classifier.model.predict(array)
    label = np.argmax(prediction)

    return {"label": labels[label]}


@app.get('/')
async def index():
    return FileResponse(Path("static/index.html"))
