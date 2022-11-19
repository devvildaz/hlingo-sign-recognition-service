from fastapi import FastAPI
from fastapi import File
from models import Coordinate
from hololingo_model import HoloLingoModel
from mediapipe_processor import MediaPipeProcessor

app = FastAPI()
hololingo = HoloLingoModel()
processor = MediaPipeProcessor()

@app.get("/status")
def status():
    return {"status": "app running"}

@app.post("/predict/coordinates")
def predict(req: Coordinate):

    '''Applies sign language detection on landmark coordinates
    obtained via processing videos with MediaPipe'''

    return hololingo.apply_model_on_coordinates(req.coordinates)

@app.post("/predict/video")
def predict(video: bytes = File()):

    '''Applies sign language detection on videos first
    processing them with MediaPipe'''

    coordinates = processor.get_coordinates(video)
    return hololingo.apply_model_on_coordinates(coordinates)