from fastapi import FastAPI, File, UploadFile, Request
from fastapi.templating import Jinja2Templates
from models import predict
from pathlib import Path
from PIL import Image
import io
import numpy as np 
import uvicorn

app = FastAPI()
BASE = Path(__file__).resolve().parent
templates = Jinja2Templates(directory=str(BASE / "templates"))

@app.get("/")
def index(request: Request):
    return templates.TemplateResponse('index.html', context={'request': request})

@app.post("/upload")
async def getPrediction(file: UploadFile = File(...)):
    
    contents = await file.read()
    image = Image.open(io.BytesIO(contents))
    image = np.array(image)
    prediction = predict(image)

    return {"prediction": prediction.tolist()}
