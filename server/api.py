from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from utils import augment_image_cv2, load_brain_tumor_classifier

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
    classifier_model = load_brain_tumor_classifier("DenseNet")

    array = np.expand_dims(image, axis=0) / 255.0
    prediction = classifier_model.predict(array)
    label = np.argmax(prediction)

    return {"label": labels[label]}


@app.get('/')
async def index():
    return FileResponse(Path("static/index.html"))
