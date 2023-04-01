import torch
import requests
#from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base", torch_dtype=torch.float16).to("cuda")

#img_url = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/demo.jpg' 
#raw_image = Image.open(requests.get(img_url, stream=True).raw).convert('RGB')



def get_prompt(img):
    # Get the prompt from the image

    # unconditional image captioning
    inputs = processor(img, return_tensors="pt").to("cuda", torch.float16)

    out = model.generate(**inputs)
    return(processor.decode(out[0], skip_special_tokens=True))