import argparse
import torch
from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler
from tqdm import tqdm

def generate_images(input_file, output_dir):
    model_id = "stabilityai/stable-diffusion-2-1"

    # Use the Euler scheduler here instead
    scheduler = EulerDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler")
    pipe = StableDiffusionPipeline.from_pretrained(model_id, scheduler=scheduler, torch_dtype=torch.float16)
    pipe = pipe.to("cuda")

    with open(input_file, "r") as file:
        lines = file.readlines()

    for prompt in tqdm(lines, desc="Generating images"):
        prompt = prompt.strip()
        prompt = prompt.replace("  ", " ")
        image = pipe(prompt, num_inference_steps=150, guidance_scale=8).images[0]


        # Use only the text before the first comma as the image name
        filename = prompt.split(",")[0].strip()

        # Replace invalid characters in filenames
        filename = prompt.replace(" ", "_").replace(":", "").replace("/", "").replace("\\", "").replace("*", "").replace("?", "").replace("\"", "").replace("<", "").replace(">", "").replace("|", "")

        output_path = output_dir+"//"+filename+".jpg"
        #output_path = f"{output_dir}/{filename}.jpg"
        image.save(output_path)
        #print(f"Image saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate images based on prompts from a text file.")
    parser.add_argument("input_file", help="Path to the text file containing prompts, one per line.")
    parser.add_argument("output_dir", help="Path to the directory where the generated images will be saved.")
    args = parser.parse_args()

    generate_images(args.input_file, args.output_dir)