from fastapi import FastAPI, status, Request, Response, File
from fastapi.responses import JSONResponse, PlainTextResponse
from typing import Annotated
import base64
import json
import numpy as np
from torchvision import transforms
from torch import load
from PIL import Image
import torch.nn as nn
import psycopg2
from io import BytesIO
from transformers import SegformerImageProcessor, AutoModelForSemanticSegmentation
from handle_segment import get_similar

app = FastAPI()

print('Loading function')
processor = SegformerImageProcessor.from_pretrained("preprocessor_config.json")
model = load("seg.pt")
labels ={ 0: "Background", 1: "Hat", 2: "Hair", 3: "Sunglasses", 4: "Upper-clothes", 5: "Skirt", 6: "Pants", 7: "Dress", 8: "Belt", 9: "Left-shoe", 10: "Right-shoe", 11: "Face", 12: "Left-leg", 13: "Right-leg", 14: "Left-arm", 15: "Right-arm", 16: "Bag", 17: "Scarf"}
nums = range(1,17)
labels_reverse = {"tops":4, "pants":6, "shoes": 10}
vgg16_model = load("vgg_mod.pt")
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

engine = psycopg2.connect(
    database="stealmylook",
    user="postgres",
    password="postgres",
    host="stealmylook1.ctjitqpxi57z.us-east-1.rds.amazonaws.com",
    port='5432'
)
engine.autocommit = True
cur = engine.cursor()



@app.post('/invocations')
async def invocations(request: Request):
    # model() is a hypothetical function that gets the inference output:

    image_bytes = request.encode('utf-8')
    img_b64dec = base64.b64decode(image_bytes)
    img_byteIO = BytesIO(img_b64dec)
    image = Image.open(img_byteIO)

    inputs = processor(images=image, return_tensors="pt")
    outputs = model(**inputs)
    logits = outputs.logits.cpu()
    upsampled_logits = nn.functional.interpolate(
        logits,
        size=image.size[::-1],
        mode="bilinear",
        align_corners=False,
    )
       

    pred_seg = upsampled_logits.argmax(dim=1)[0]
    orig_arr = np.asarray(image)
    # Shirt
    shirt_results = get_similar(orig_arr,pred_seg,vgg16_model,transform, 4, cur)
    # Pants
    pants_results = get_similar(orig_arr,pred_seg,vgg16_model,transform, 7, cur)
    # Shoes
    shoes_results = get_similar(orig_arr,pred_seg,vgg16_model,transform, 10, cur)
    
    output = json.dumps({
        "shirts":shirt_results,
                "pants": pants_results,
                "shoes":shoes_results
    })


    return JSONResponse(content=output)

@app.get('/ping')
async def invocations(request: Request):

    response = Response(
        content=json.dumps({"message":"allgood"}),
        status_code=status.HTTP_200_OK,
        media_type="text/plain",
    )
    return response

@app.get("/", response_class=PlainTextResponse, tags=["home"])
async def home():
    note = """
    Hi
    """
    return note