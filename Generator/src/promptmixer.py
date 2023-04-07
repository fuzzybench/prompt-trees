##llm here
from src import img2prompt
import torch
from transformers import GenerationConfig, LlamaTokenizer, LlamaForCausalLM


def generate_prompt(instruction: str, input_ctxt: str = None) -> str:
    if input_ctxt:
        return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{input_ctxt}

### Response:"""
    else:
        return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Response:"""




tokenizer = LlamaTokenizer.from_pretrained("chainyo/alpaca-lora-7b")
model = LlamaForCausalLM.from_pretrained(
    "chainyo/alpaca-lora-7b",
    load_in_8bit=True,
    torch_dtype=torch.float16,
    device_map="auto",
)
generation_config = GenerationConfig(
    temperature=0.2,
    top_p=0.75,
    top_k=40,
    num_beams=4,
    max_new_tokens=128,
)

model.eval()
if torch.__version__ >= "2":
    model = torch.compile(model).to("cuda")




def generate_prompts(prompt_estimate, prompt_expansion):
    new_prompt_array = []
    
    instruction = "With this caption: "+ prompt_estimate + ", create " + str(prompt_expansion)+ " new variations of the caption that contains descriptions and artistic styles. Do not include the word variaton. Only return the new generated captions. Add an artistic style to the caption to describe what the image generated from the caption might look like. Only return each caption variation. Each new varation line should be formated as Variation: . do not call a line a Caption for the line specifier."
    #input_ctxt = ""#"you are making a unique prompt to generate an image from the prompt text. The prompt contains captions, descriptions, and styles of the image. "  # For some tasks, you can provide an input context to help the model generate a better response.

    #prompt = generate_prompt(instruction, input_ctxt)
    prompt = generate_prompt(instruction)
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    input_ids = input_ids.to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            generation_config=generation_config,
            return_dict_in_generate=True,
            output_scores=True,
        )

    response = tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
    sep_response = response.split("Response:")
        #print("total response: ---------------------" + sep_response[1])
    sep_answers = sep_response[1].strip().split("Variation")
        #print("First prompt ---------------------" + sep_answers[0])
    start_index = len(sep_answers)-prompt_expansion
    sep_answers = sep_answers[start_index:]
    #print("returned prompt answer length-------------------" + str(len(sep_answers)))
    for i in sep_answers:
        spliced = i[(i.find(":")+1):].strip().replace('"', '')
        new_prompt_array.append(spliced)
        print("-----Generated Prompts---------- " + spliced)
    return new_prompt_array



def get_prompts(img, prompt_expansion):
        prompt_estimate = img2prompt.get_prompt(img)
        print("estimated prompt----------" + str(prompt_estimate))
        #images may not be getting saved/may be using only the initial image
        #these prompts generated from the img-prompt model may be getting saved instead of the generated variations
        new_prompts = generate_prompts(prompt_estimate, prompt_expansion)
        
        #returns an array of new prompts prompts
        return new_prompts


def prompt_tester():
    instruction = "With this caption: man riding a horse, create 3 new variations of the caption that contains descriptions and artistic styles. Do not include the word variaton. Only return the new generated captions. Add an artistic style to the caption to describe what the image generated from the caption might look like. Only return each caption variation. Each new varation line should be formated as Variation: <new variation here>. do not call a line a Caption for the line specifier."

    prompt = generate_prompt(instruction)
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    input_ids = input_ids.to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            generation_config=generation_config,
            return_dict_in_generate=True,
            output_scores=True,
        )

    response = tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
    val = response.split("Response:")
    val_array = val[1].split("Variation")
    prompt_array = []
    for i in val_array:
        spliced = i[(i.find(":")+1):].strip()
        prompt_array.append(spliced)
        print(spliced)
    #print(val[1])