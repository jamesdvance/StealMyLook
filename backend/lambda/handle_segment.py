import torch 
from PIL import Image

def get_similar(orig_arr,pred_seg,vgg16_model,transform, num, cur): 

    img_arr = orig_arr.copy()
    img_arr[pred_seg!=num] = [255,255,255]

    img_tensor = transform(Image.fromarray(img_arr))
    img_batch = img_tensor.unsqueeze(0) 

    with torch.no_grad():
        output = vgg16_model(img_batch)

    embed = str(list(output.numpy().flatten()))

    res = cur.execute(f"""
        SELECT * FROM tops where gender = 'Men' ORDER BY img_embedding <-> '{embed}' LIMIT 5 ;
    """)
    return cur.fetchall()