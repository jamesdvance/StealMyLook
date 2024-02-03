import json
import base64
import numpy as np
from PIL import Image
import torch.nn as nn
from io import BytesIO
from transformers import SegformerImageProcessor, AutoModelForSemanticSegmentation

print('Loading function')
processor = SegformerImageProcessor.from_pretrained("mattmdjaga/segformer_b2_clothes")
model = AutoModelForSemanticSegmentation.from_pretrained("mattmdjaga/segformer_b2_clothes")
Labels ={ 0: "Background", 1: "Hat", 2: "Hair", 3: "Sunglasses", 4: "Upper-clothes", 5: "Skirt", 6: "Pants", 7: "Dress", 8: "Belt", 9: "Left-shoe", 10: "Right-shoe", 11: "Face", 12: "Left-leg", 13: "Right-leg", 14: "Left-arm", 15: "Right-arm", 16: "Bag", 17: "Scarf"}
nums = range(1,17)
labels_reverse = {"Upper-clothes":4, "Pants":6, "shoes": (9,10)}

def lambda_handler(event, context):
    #print("Received event: " + json.dumps(event, indent=2))
    image_bytes = event['body'].encode('utf-8')
    img_b64dec = base64.b64decode(image_bytes)
    img_byteIO = BytesIO(img_b64dec)
    image = Image.open(img_byteIO)

    # Segment 
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
    shirt_arr = orig_arr.copy()
    shirt_arr[pred_seg!=4] = [255,255,255]
    # Convert Segments to Embeddings

    # shirt 

    # pants 

    # skirt 

    # shoes

    # Retreive Similarities
    
    # Format Metadata
    
    return json.dumps({"success": 1, "image":my_string})
