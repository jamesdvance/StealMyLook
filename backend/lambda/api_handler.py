import json
import base64
import numpy as np
from torch import load
from torchvision import  transforms
from PIL import Image
import torch.nn as nn
import psycopg2
from io import BytesIO
from transformers import SegformerImageProcessor, AutoModelForSemanticSegmentation
from handle_segment import get_similar

print('Loading function')
processor = SegformerImageProcessor.from_pretrained("preprocessor_config.json")
model = load("seg.pt")
Labels ={ 0: "Background", 1: "Hat", 2: "Hair", 3: "Sunglasses", 4: "Upper-clothes", 5: "Skirt", 6: "Pants", 7: "Dress", 8: "Belt", 9: "Left-shoe", 10: "Right-shoe", 11: "Face", 12: "Left-leg", 13: "Right-leg", 14: "Left-arm", 15: "Right-arm", 16: "Bag", 17: "Scarf"}
nums = range(1,17)
labels_reverse = {"Upper-clothes":4, "Pants":6, "shoes": (9,10)}
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

def handler(event, context):
    #print("Received event: " + json.dumps(event, indent=2))
    image_bytes = event['body'].encode('utf-8')
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

    return {
        "isBase64Encoded": False,
        "statusCode": 200,
        "body": output,
        "headers": {
            "content-type": "application/json"
        }
    }