from datasets import load_dataset
from transformers import BartTokenizer, BartForConditionalGeneration, Trainer, TrainingArguments
import torch
from transformers import get_linear_schedule_with_warmup
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from accelerate import Accelerator
from torch.utils.data import DataLoader

# Inicializar Accelerator
accelerator = Accelerator(mixed_precision='fp16')

# Carregar o tokenizer e o modelo BERT

model_name = 'facebook/bart-base'  # Você pode usar 'facebook/bart-large' se tiver recursos para isso
tokenizer = BartTokenizer.from_pretrained(model_name)
model = BartForConditionalGeneration.from_pretrained(model_name)

# Definir
model.config.decoder_start_token_id = tokenizer.cls_token_id
model.config.pad_token_id = tokenizer.pad_token_id
model.config.bos_token_id = tokenizer.cls_token_id
model.config.eos_token_id = tokenizer.sep_token_id

# Configuração opcional para evitar avisos
model.config.vocab_size = len(tokenizer) 

# Função para interagir com o modelo
def generate_response(input_text):
    # Tokenizar o texto de entrada
    input_ids = tokenizer.encode(input_text, return_tensors="pt", padding=True, truncation=True, max_length=128).to(model.device)

    # Definir os parâmetros de entrada para a geração
    decoder_start_token_id = tokenizer.cls_token_id

    # Gerar a saída
    output_ids = model.generate(
        input_ids,
        decoder_start_token_id=decoder_start_token_id, #add novo
        max_length=100,
        num_beams=4,
        no_repeat_ngram_size=2,              # Evita repetição de n-grams
        top_p=0.0,                          # Controle de amostragem top-p
        top_k=20,                            # Controle de amostragem top-k
        do_sample=True,                      # Amostragem para maior diversidade
        temperature=0.3                      # Ajuste de temperatura para controlar a aleatoriedade
    )

    # Decodificar a saída
    response = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return response

# Testar com um exemplo
input_text = "what's the description of 'How to Babysit a Grandma'?"
response = generate_response(input_text)

print(f"Input: {input_text}")
print(f"Output: {response}")

