from fastapi import FastAPI
from hololingo_model import HoloLingoModel

app = FastAPI()
hololingo = HoloLingoModel()

@app.get("/status")
def status():
    return {"status": "app running"}