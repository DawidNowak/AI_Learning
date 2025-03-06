import modal
from modal import Image

# Setup

app = modal.App("joke-service")
image = Image.debian_slim().pip_install("torch", "transformers", "bitsandbytes", "accelerate", "peft")
secrets = [modal.Secret.from_name("huggingface-secret")]
GPU = "T4"
MODEL_NAME = "meta-llama/Meta-Llama-3.1-8B" # "google/gemma-2-2b"

@app.function(image=image, secrets=secrets, gpu=GPU, timeout=1800)
def generate() -> str:
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

    # Quant Config
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4"
    )

    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, 
        quantization_config=quant_config,
        device_map="auto"
    )

    prompt = "Tell a lighthearted joke about programmers. No explanation, just the joke."
    input = tokenizer.encode(prompt, return_tensors="pt").to("cuda")
    attention_mask = torch.ones(input.shape, device="cuda")
    outputs = model.generate(input, attention_mask=attention_mask, max_new_tokens=50, num_return_sequences=1)
    return tokenizer.decode(outputs[0])