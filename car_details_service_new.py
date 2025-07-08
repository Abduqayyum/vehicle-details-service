from fastapi import FastAPI, File, Form, UploadFile, BackgroundTasks
import uvicorn
from fastapi.responses import JSONResponse
from transformers import ViTForImageClassification, AutoImageProcessor
from PIL import Image
import torch
import io
from ultralytics import YOLO
import asyncio
import uuid
import numpy as np
from config import settings
from minio import Minio
import filetype
import cv2

app = FastAPI()

car_model = ViTForImageClassification.from_pretrained("vit-car-model")
car_processor = AutoImageProcessor.from_pretrained("vit-car-model")

color_model = ViTForImageClassification.from_pretrained("vit-color-model")
color_processor = AutoImageProcessor.from_pretrained("vit-color-model")

car_model.eval()
color_model.eval()

allowed_mime_types = ['image/jpeg', 'image/png', 'image/jpg', 'image/webp']

detection_model = YOLO("yolo11m.pt")
license_plate_model = YOLO("latest-license-plate-model.pt")

CLASSES_ID = [2, 3, 5, 7]


def detect_license_plate(image, model):
    result = model(image, verbose=False)
    if len(result) == 0 or len(result[0]) == 0 or len(result[0].boxes) == 0:
        return None
    
    for r in result:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            roi = image[y1:y2, x1:x2]
            blurred_roi = cv2.GaussianBlur(roi, (29, 29), 20)
            image[y1:y2, x1:x2] = blurred_roi
    return image


def predict_single(image, processor, model):
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        pred_class = logits.argmax(-1).item()
    return model.config.id2label[pred_class]


async def predict_async(cropped_car, detect_vehicle=False, recognize_color=False):
    pil_image = Image.fromarray(cv2.cvtColor(cropped_car, cv2.COLOR_BGR2RGB))

    tasks = []
    if detect_vehicle:
        tasks.append(asyncio.to_thread(predict_single, pil_image, car_processor, car_model))
    else:
        tasks.append(asyncio.sleep(0)) 
    if recognize_color:
        tasks.append(asyncio.to_thread(predict_single, pil_image, color_processor, color_model))
    else:
        tasks.append(asyncio.sleep(0)) 

    car_name, color_name = await asyncio.gather(*tasks)
    return car_name if detect_vehicle else None, color_name if recognize_color else None

@app.post("/car-details")
async def main(
    image: UploadFile = File(...),
    detect_vehicle: bool = Form(False),
    recognize_color: bool = Form(False),
):
    contents = await image.read()
    kind = filetype.guess(contents)
    if kind is None or kind.mime not in allowed_mime_types:
        return JSONResponse(status_code=400, content={"success": {}, "error": {"description": "Invalid file type. Only image files are accepted!"}})
    
    try:
        encoded_img = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(encoded_img, cv2.IMREAD_COLOR)
        results = detection_model(img, classes=CLASSES_ID, conf=0.5, verbose=False)

        if results[0].boxes is not None and len(results[0].boxes) > 0:
            result = results[0]
            boxes = result.boxes[0]
            x1, y1, x2, y2 = map(int, boxes.xyxy[0])
            car_img = img[y1:y2, x1:x2]

            model_name, color_name = await predict_async(car_img, detect_vehicle, recognize_color)

            return JSONResponse(status_code=200, content={
                "success": {
                    "result": {
                        "car_model": model_name,
                        "car_color": color_name,
                    }
                },
                "error": ""
            })
        else:
            return JSONResponse(status_code=400, content={"success": {}, "error": {"description": "Vehicle is not detected!"}})
    except Exception as ex:
        return JSONResponse(status_code=500, content={"success": {}, "error": {"description": f"{ex}"}})


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=3000)
