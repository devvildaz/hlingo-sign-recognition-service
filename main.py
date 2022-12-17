from typing import List, Union
from fastapi import FastAPI, Header, HTTPException
from fastapi import File
from fastapi.middleware.cors import CORSMiddleware
from models import Coordinate
from hololingo_model import HoloLingoModel
from mediapipe_processor import MediaPipeProcessor

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*']
)
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
def predict(video: bytes = File(), video_format: Union[List[str], None] = Header(default=None)):
    '''Applies sign language detection on videos first
    processing them with MediaPipe'''

    if video_format is None:
        raise HTTPException(status_code=404, detail='Video format header is empty')
    if len(video_format) != 1:
        raise HTTPException(status_code=404, detail='Video format header has more than 1 value')

    coordinates = processor.get_coordinates(video, video_format)
    return hololingo.apply_model_on_coordinates(coordinates)