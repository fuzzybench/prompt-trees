#Run this code!

import argparse
from src import promptmixer, img2prompt, prompt2img


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate images based on an initial prompt")
    parser.add_argument("starting_prompt", help="Your starting prompt")
    parser.add_argument("output_dir", help="Path to the directory where the generated images will be saved.")
    args = parser.parse_args()

    prompt2img.generate_images(args.input_file, args.output_dir)