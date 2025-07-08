FROM python:3.10

WORKDIR /app

RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

COPY car_details_service_new.py .

COPY config.py .

COPY .env .

COPY vit-car-model/ vit-car-model/

COPY vit-color-model/ vit-color-model/

COPY latest-license-plate-model.pt .

RUN pip install -r requirements.txt

CMD ["uvicorn", "car_details_service_new:app", "--host", "0.0.0.0", "--port", "3060"]
