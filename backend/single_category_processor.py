import json
import base64
import numpy as np
import torch
from torchvision import models, transforms
from PIL import Image
import torch.nn as nn
from io import BytesIO
from transformers import SegformerImageProcessor, AutoModelForSemanticSegmentation

print('Loading function')
processor = SegformerImageProcessor.from_pretrained("mattmdjaga/segformer_b2_clothes")
model = AutoModelForSemanticSegmentation.from_pretrained("mattmdjaga/segformer_b2_clothes")
labels ={ 0: "Background", 1: "Hat", 2: "Hair", 3: "Sunglasses", 4: "Upper-clothes", 5: "Skirt", 6: "Pants", 7: "Dress", 8: "Belt", 9: "Left-shoe", 10: "Right-shoe", 11: "Face", 12: "Left-leg", 13: "Right-leg", 14: "Left-arm", 15: "Right-arm", 16: "Bag", 17: "Scarf"}
nums = range(1,17)
labels_reverse = {"tops":4, "Pants":6, "shoes": 10}
vgg16_model = models.vgg16(pretrained=True)
vgg16_model.classifier = vgg16_model.classifier[:-1].append(nn.AvgPool1d(8, stride=3, padding=0, ceil_mode=False, count_include_pad=True)) # convert to embedding 
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def process_category(image, category):
    #print("Received event: " + json.dumps(event, indent=2))

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

    img_arr = orig_arr.copy()

    img_arr[pred_seg!=labels[labels_reverse[category]]] = [255,255,255]
    # Pants
    
    # Convert Segments to Embeddings
    img_tensor = transform(Image.fromarray(img_arr))
    img_batch = img_tensor.unsqueeze(0) 

    with torch.no_grad():
        output = vgg16_model(img_batch)

    return output.numpy().flatten()