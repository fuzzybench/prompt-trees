#Run this code!

import argparse
import os
import csv
from tqdm import tqdm
from src import promptmixer
from src import prompt2img

###maybe just save all text a csv, all images to a folder. Have a numerical naming convention


#concat prompts to csv. Save all the params to the header of that csv as well
#def

def append_to_csv(file_path, array):
    # Check if file exists, if not, create the file with headers
    file_path_csv = file_path+"/new-prompts.csv"
    if not os.path.exists(file_path_csv):
        with open(file_path_csv, mode='w', newline='') as file:
            writer = csv.writer(file)
            header = [f'Value_{i}' for i in range(1, len(array) + 1)]
            writer.writerow(header)

    # Open the file in append mode and add the array as a row
    with open(file_path_csv, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(array)


#generates the next level of images from the previous level of img prompts
def generate_level(img_array, prompt_array, prompt_expansion, level, output_dir, gen_next_prompts):
    new_prompt_array = []
    new_img_array = []
    print("length of prompt array-------------" + str(len(prompt_array)))
    for img_num, single_prompt_array in enumerate(prompt_array):
        print("length of single prompt array-------------" + str(len(single_prompt_array)))
        for prompt_num, prompt in enumerate(single_prompt_array):
            output_path_img = str(output_dir)+"/"+str(level)+"_"+str(img_num)+"_"+str(prompt_num)+".jpg"
            img = prompt2img.generate_image(prompt)
            img.save(output_path_img)
            new_img_array.append(img)
            
            print("---------------------The prompt being saved is: " + prompt)
            prompt_id = str(level)+"_"+str(img_num)+"_"+str(prompt_num)
            append_to_csv(output_dir,[prompt_id, str(prompt)])
            if(gen_next_prompts):
                new_prompt_array.append(promptmixer.get_prompts(img, prompt_expansion))
    print("length of prompt array being saved-------------" + str(len(prompt_array)))
    return new_img_array, new_prompt_array



def generate_tree(initial_prompt, levels, prompt_expansion, output_dir):
    prompt_array = [[initial_prompt.strip()]]
    img_array = []

    for level in range(levels):
        print("current level is: "+ str(level))
        gen_next_prompts = True
        if (level==levels-1):
            gen_next_prompts = False
        img_array, prompt_array = generate_level(img_array, prompt_array, prompt_expansion, level, output_dir, gen_next_prompts)
  
            

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate images based on an initial prompt")
    parser.add_argument("levels", help="How many levels of hierarical prompts to generate", type=int)
    parser.add_argument("prompt_expansion", help="How many new prompts to be generated per node", type=int)
    parser.add_argument("input_file", help="Path to the text file containing prompts, one per line.")
    parser.add_argument("output_dir", help="Path to the directory where the generated images will be saved.")
    args = parser.parse_args()


    with open(args.input_file, "r") as file:
        lines = file.readlines()
        #loop through each initial prompt
        for initial_prompt in tqdm(lines, desc="Generating images"):
            print("Starting prompt: " + initial_prompt)

            generate_tree(initial_prompt, args.levels, args.prompt_expansion, args.output_dir)


