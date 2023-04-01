import torch
from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler

# Generate an image from a prompt
def generate_image(prompt):
    model_id = "stabilityai/stable-diffusion-2-1"

    # Use the Euler scheduler here instead
    scheduler = EulerDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler")
    pipe = StableDiffusionPipeline.from_pretrained(model_id, scheduler=scheduler, torch_dtype=torch.float16)
    pipe = pipe.to("cuda")

    image = pipe(prompt, num_inference_steps=100, guidance_scale=7.5).images[0]
    return image

def save_image(prompt, image, output_dir):
        # Use only the text before the first comma as the image name
    filename = prompt.split(",")[0].strip()

    # Replace invalid characters in filenames
    filename = prompt.replace(" ", "_").replace(":", "").replace("/", "").replace("\\", "").replace("*", "").replace("?", "").replace("\"", "").replace("<", "").replace(">", "").replace("|", "")

    output_path = output_dir+"//"+filename+".jpg"
    image.save(output_path)