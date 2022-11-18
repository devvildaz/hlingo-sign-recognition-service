from fastapi import FastAPI

app = FastAPI()

@app.get("/status")
def status():
    return {"status": "app running"}