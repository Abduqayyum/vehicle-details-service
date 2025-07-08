import os
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

load_dotenv()

class Settings(BaseSettings):
    minio_endpoint: str = os.getenv("MINIO_ENDPOINT", "http://localhost:9002")
    minio_access_key: str = os.getenv("MINIO_ACCESS_KEY", "admin123")
    minio_secret_key: str = os.getenv("MINIO_SECRET_KEY", "admin123")
    minio_secure: bool = os.getenv("MINIO_SECURE", False)
    minio_bucket_name : str = os.getenv("BUCKET_NAME", "landing")
    minio_folder_name : str = os.getenv("FOLDER_NAME", "license-plates")
    minio_url : str = os.getenv("MINIO_URL", "")
    sag_url: str = os.getenv("SAG_URL", "")

settings = Settings()