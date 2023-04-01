#Run this code!

import argparse
import os
import csv
from tqdm import tqdm
from src import promptmixer, prompt2img

###maybe just save all text a csv, all images to a folder. Have a numerical naming convention


#concat prompts to csv. Save all the params to the header of that csv as well
#def

def append_to_csv(file_path, array):
    # Check if file exists, if not, create the file with headers
    if not os.path.exists(file_path):
        with open(file_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            header = [f'Value_{i}' for i in range(1, len(array) + 1)]
            writer.writerow(header)

    # Open the file in append mode and add the array as a row
    with open(file_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(array)


#generates the next level of images from the previous level of img prompts
def generate_level(img_array, prompt_array, prompt_expansion, steps, guidance, level, output_dir):
    new_prompt_array = []
    new_img_array = []
    for img_num, single_prompt_array in enumerate(prompt_array):
        for prompt_num, prompt in enumerate(single_prompt_array):
            img = prompt2img.generate_image(prompt, steps, guidance)
            new_img_array.append(img)
            output_path_img = str(output_dir)+"//images//"+str(level)+"_"+str(img_num)+"_"+str(prompt_num)+".jpg"

            prompt_id = str(level)+"_"+str(img_num)+"_"+str(prompt_num)
            append_to_csv(output_dir [prompt_id, prompt])

            new_prompt_array.append(promptmixer.get_prompts(img, prompt_expansion))
    return new_img_array, new_prompt_array



def generate_tree(initial_prompt, levels, prompt_expansion, steps, guidance, output_dir):
    prompt_array = initial_prompt
    img_array = []

    for level in range(levels):
        img_array, prompt_array = generate_level(img_array, prompt_array, prompt_expansion, steps, guidance, level, output_dir)
  
            

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate images based on an initial prompt")
    parser.add_argument("levels", help="How many levels of hierarical prompts to generate", type=int)
    parser.add_argument("prompt_expansion", help="How many new prompts to be generated per node", type=int)
    parser.add_argument("steps", help="Stable diffusion step count")
    parser.add_argument("guidance", help="Stable diffusion guidance scale")
    parser.add_argument("input_file", help="Path to the text file containing prompts, one per line.")
    parser.add_argument("output_dir", help="Path to the directory where the generated images will be saved.")
    args = parser.parse_args()


    with open(args.input_file, "r") as file:
        lines = file.readlines()
        #loop through each initial prompt
        for initial_prompt in tqdm(lines, desc="Generating images"):
            initial_prompt_array = [initial_prompt]

            generate_tree(initial_prompt_array, args.levels, args.prompt_expansion, args.steps, args.guidance, args.output_dir)


