import json
import modal
from modal import Image

# Setup - define infrastructure with code!

app = modal.App("python-expert")
image = Image.debian_slim().pip_install("huggingface", "torch", "transformers", "bitsandbytes", "accelerate", "peft")
secrets = [modal.Secret.from_name("huggingface-secret")]
GPU = "T4"
MODEL_NAME = "meta-llama/Meta-Llama-3.1-8B" # "google/gemma-2-2b"
MODEL_DIR = "hf-cache/"
BASE_DIR = MODEL_DIR + MODEL_NAME

QUESTION = (
    "Please explain what does this python code do and why. "
    "Keep your response simple and concise. "
    "The code to explain: "
)

@app.cls(image=image, secrets=secrets, gpu=GPU, timeout=1800)
class Expert:
    @modal.build()
    def download_model_to_folder(self):
        from huggingface_hub import snapshot_download
        import os
        os.makedirs(MODEL_DIR, exist_ok=True)
        snapshot_download(MODEL_NAME, local_dir=BASE_DIR)

    @modal.enter()
    def setup(self):
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
        
        self.tokenizer = AutoTokenizer.from_pretrained(BASE_DIR)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"
        
        self.base_model = AutoModelForCausalLM.from_pretrained(
            BASE_DIR, 
            quantization_config=quant_config,
            device_map="auto"
        )

    @modal.method()
    def explain(self, code: str) -> str:
        import torch
    
        prompt = f"{QUESTION}\n\n{code}"
        inputs = self.tokenizer.encode(prompt, return_tensors="pt").to("cuda")
        attention_mask = torch.ones(inputs.shape, device="cuda")
        outputs = self.base_model.generate(
            inputs,
            attention_mask=attention_mask,
            num_return_sequences=1,
            max_new_tokens=256,
            eos_token_id=self.tokenizer.eos_token_id
        )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    @modal.method()
    def wake_up(self) -> str:
        return "ok"
    
