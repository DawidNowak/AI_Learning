from dotenv import load_dotenv
from transformers import AutoTokenizer

load_dotenv()

# from huggingface_hub import login
# login(HF_TOKEN)
# No need to explicitly login, HF_TOKEN will be used automatically
# The Hugging Face library automatically uses HF_TOKEN environment variable

# NOTE
# In order to use Llama 3.1, Meta does require you to sign their terms of service.
# Visit their model instructions page in Hugging Face: https://huggingface.co/meta-llama/Meta-Llama-3.1-8B
# After filling the form I got an approval in about 30 minutes
# You can check the status in your HuggingFace profile settings in Gated Repositories

text = "Tokenizers tests. I wonder how the tokens for different models looks!"
messages = [
    {"role": "system", "content": "You are a helpful assistant"},
    {"role": "user", "content": "Tell a light-hearted joke for a room of Data Scientists"}
]

def print_tokens(model, text, tokens, decoded_tokens):
    print(f"*****\nMODEL: {model}\nTEXT: {text}\nTOKENS: {tokens}\nDECODED: {decoded_tokens}\n*****\n")

def print_chat_tokens(model, text, tokens):
    print(f"*****\nMODEL: {model}\nTEXT: {text}\nTOKENS: {tokens}\n*****\n")

llama31_8b_model = 'meta-llama/Meta-Llama-3.1-8B'
llama31_8b_tokenizer = AutoTokenizer.from_pretrained(llama31_8b_model, trust_remote_code=True)
llama31_8b_tokens = llama31_8b_tokenizer.encode(text)
print_tokens(llama31_8b_model, text, llama31_8b_tokens, llama31_8b_tokenizer.batch_decode(llama31_8b_tokens))


# Many models have a variant that has been trained for use in Chats.
# These are typically labelled with the word "Instruct" at the end.
# They have been trained to expect prompts with a particular format that includes system, user and assistant prompts.

llama31_8b_instruct_model = 'meta-llama/Meta-Llama-3.1-8B-Instruct'
llama31_8b_instruct_tokenizer = AutoTokenizer.from_pretrained(llama31_8b_instruct_model, trust_remote_code=True)
llama31_8b_instruct_tokens = llama31_8b_instruct_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
print_chat_tokens(llama31_8b_instruct_model, messages, llama31_8b_instruct_tokens)


phi3_model = "microsoft/Phi-3-mini-4k-instruct"
phi3_tokenizer = AutoTokenizer.from_pretrained(phi3_model)
phi3_tokens = phi3_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
print_chat_tokens(phi3_model, messages, phi3_tokens)


qwen2_model = "Qwen/Qwen2-7B-Instruct"
qwen2_tokenizer = AutoTokenizer.from_pretrained(phi3_model)
qwen2_tokens = qwen2_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
print_chat_tokens(qwen2_model, messages, qwen2_tokens)


starcoder2_model = "bigcode/starcoder2-3b"
starcoder2_tokenizer = AutoTokenizer.from_pretrained(starcoder2_model, trust_remote_code=True)
code = """
def hello_world(person):
  print("Hello", person)
"""
starcoder2_tokens = starcoder2_tokenizer.encode(code)
print(f"*****\n{starcoder2_model}")
for token in starcoder2_tokens:
  print(f"{token} = '{starcoder2_tokenizer.decode(token)}'")
print("*****")