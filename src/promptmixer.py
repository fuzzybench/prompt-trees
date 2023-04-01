##llm here
import img2prompt
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
    model = torch.compile(model).to("cuda")#should i have to to .cuda rn?




def generate_prompts(prompt_estimate, prompt_expansion):
    instruction = "Write 2 unique descriptive image caption prompts that contain an artistic style of the image that will be used to generate an image for the following image description: " + prompt_estimate + ". format the data as follows: prompt1, prompt2. Do not return anything other than prompt1:prompt2."
    input_ctxt = "you are making 2 unique prompts to generate an image from. The prompts contain captions, descriptions, and styles of the image. You are to return the data in the described formatted way"  # For some tasks, you can provide an input context to help the model generate a better response.

    prompt = generate_prompt(instruction, input_ctxt)
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
    response = response.split(":")
    print(response)


    return response



def get_prompts(img, prompt_expansion):
        prompt_estimate = img2prompt.generate_prompt(img)
        new_prompts = generate_prompts(prompt_estimate, prompt_expansion)
        
        #returns an array of new prompts prompts
        return generate_prompts(prompt_estimate, prompt_expansion)
