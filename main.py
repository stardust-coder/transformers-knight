from time import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import pdb
import torch
import os 

cache_dir = "../.cache"
os.environ["TRANSFORMERS_CACHE"] = cache_dir
os.environ["HF_HOME"] = cache_dir

print("Execution start...")
model_name = "tokyotech-llm/Llama-3.1-Swallow-8B-Instruct-v0.3"
tokenizer = AutoTokenizer.from_pretrained(model_name,cache_dir=cache_dir)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map="auto",cache_dir=cache_dir)

prompt = "東京工業大学の主なキャンパスは、"
input_ids = tokenizer.encode(
    prompt,
    add_special_tokens=False,
    return_tensors="pt"
)

for _ in range(1):
    tokens = model.generate(
        input_ids.to(device=model.device),
        max_new_tokens=128,
        temperature=0.99,
        top_p=0.95,
        do_sample=True,
    )

    start = time()
    out = tokenizer.decode(tokens[0], skip_special_tokens=True)
    end = time()
    print(out)
    print("Inference time:", end-start, " seconds")
