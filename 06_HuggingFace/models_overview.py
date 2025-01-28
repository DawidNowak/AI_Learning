import os
import torch
from dotenv import load_dotenv
from transformers import pipeline
from diffusers import DiffusionPipeline
from datasets import load_dataset
import soundfile as sf
from IPython.display import Audio
import shutil

load_dotenv()

output_dir = "./Outputs"
os.makedirs(output_dir, exist_ok=True)

# from huggingface_hub import login
# login(HF_TOKEN)
# No need to explicitly login, HF_TOKEN will be used automatically
# The Hugging Face library automatically uses HF_TOKEN environment variable

# MODELS:
# 1. Sentiment Analysis
# 2. Named Entity Recognition
# 3. Question Answering with Context
# 4. Text Summarization
# 5. Translation
# 6. Classification
# 7. Text Generation
# 8. Audio Generation

MODEL = 1

#  No gpu, can't set device="cuda"
#  Otherwise you can add optional device argument to pipeline() function

match MODEL:
    case 1:
        # Sentiment Analysis
        classifier = pipeline("sentiment-analysis")
        result = classifier("I'm super excited to be on the way to LLM mastery!")
        print(result)
        # [{'label': 'POSITIVE', 'score': 0.9993460774421692}]
    case 2:
        # Named Entity Recognition
        ner = pipeline("ner", grouped_entities=True)
        result = ner("Barack Obama was the 44th president of the United States.")
        print(result)
        # [
        #   {'entity_group': 'PER', 'score': np.float32(0.99918306), 'word': 'Barack Obama', 'start': 0, 'end': 12},
        #   {'entity_group': 'LOC', 'score': np.float32(0.9986908), 'word': 'United States', 'start': 43, 'end': 56}
        # ]
    case 3:
        # Question Answering with Context
        question_answerer = pipeline("question-answering")
        result = question_answerer(question="Who was the 44th president of the United States?", context="Barack Obama was the 44th president of the United States.")
        print(result)
        # {'score': 0.9889456033706665, 'start': 0, 'end': 12, 'answer': 'Barack Obama'}
    case 4:
        # Text Summarization
        summarizer = pipeline("summarization")
        text = """The Hugging Face transformers library is an incredibly versatile and powerful tool for natural language processing (NLP).
        It allows users to perform a wide range of tasks such as text classification, named entity recognition, and question answering, among others.
        It's an extremely popular library that's widely used by the open-source data science community.
        It lowers the barrier to entry into the field by providing Data Scientists with a productive, convenient way to work with transformer models.
        """
        summary = summarizer(text, max_length=50, min_length=25, do_sample=False)
        print(summary[0]['summary_text'])
        # The Hugging Face transformers library is an incredibly versatile and powerful tool for natural language processing (NLP)
        # It allows users to perform a wide range of tasks such as text classification, named entity recognition, and question answering .
    case 5:
        # Translation
        # All translation models https://huggingface.co/models?pipeline_tag=translation&sort=trending
        translator = pipeline("translation_en_to_pl", model="sdadas/mt5-base-translator-en-pl")
        result = translator("The Data Scientists were truly amazed by the power and simplicity of the HuggingFace pipeline API.", max_length=30)
        print(result)
        # [{'translation_text': ' Naukowcy Data Scientist byli naprawdę zaskoczeni mocą i prostotą API rurociągu HuggingFace'}]
        # (ﾐ〒﹏〒ﾐ) pretty lame translation
    case 6:
        # Classification
        classifier = pipeline("zero-shot-classification")
        result = classifier("Hugging Face's Transformers library is amazing!", candidate_labels=["technology", "sports", "politics"])
        print(result)
        # {
        #   'sequence': "Hugging Face's Transformers library is amazing!",
        #   'labels': ['technology', 'sports', 'politics'],
        #   'scores': [0.9493840336799622, 0.032250095158815384, 0.018365908414125443]
        # }
    case 7:
        # Text Generation
        generator = pipeline("text-generation")
        result = generator("If there's one thing I want you to remember about using HuggingFace pipelines, it's")
        print(result)
        # [{
        #   'generated_text': "If there's one thing I want you to remember about using HuggingFace pipelines,
        #       it's that they're not perfect and they're not exactly perfect tools for real life.
        #       They have limitations. There's been many people that have been working on this"}
        # ]
        # ( ͡• ͜ʖ ͡• ) kinda funny
    case 8:
        # Audio Generation
        synthesiser = pipeline("text-to-speech", "microsoft/speecht5_tts")
        embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
        speaker_embedding = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0)
        speech = synthesiser("Hi to an artificial intelligence engineer, on the way to mastery!", forward_params={"speaker_embeddings": speaker_embedding})
        sf.write(f"{output_dir}/speech.wav", speech["audio"], samplerate=speech["sampling_rate"])

# Locate and delete the hugging face cache models, etc.
cache_dir = os.path.expanduser("~/.cache/huggingface")
shutil.rmtree(cache_dir, ignore_errors=True)