from datasets import load_dataset
from transformers import BertTokenizer, EncoderDecoderModel, Trainer, TrainingArguments
import torch
from transformers import get_linear_schedule_with_warmup
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from accelerate import Accelerator
from torch.utils.data import DataLoader

# Inicializar Accelerator
accelerator = Accelerator(mixed_precision='fp16')

# Carregar o tokenizer e o modelo BERT
model_name = 'bert-base-multilingual-cased'

tokenizer = BertTokenizer.from_pretrained(model_name)
model = EncoderDecoderModel.from_encoder_decoder_pretrained(model_name, model_name)

# Definir
model.config.decoder_start_token_id = tokenizer.cls_token_id
model.config.pad_token_id = tokenizer.pad_token_id
model.config.bos_token_id = tokenizer.cls_token_id
model.config.eos_token_id = tokenizer.sep_token_id

# Configuração opcional para evitar avisos
model.config.vocab_size = model.config.encoder.vocab_size

file_path = './data/trn_clean.json'

all_dataset = load_dataset('json', data_files=file_path)

shuffled_dataset = all_dataset['train'].shuffle(seed=42)

# Pegar 10% do dataset
dataset = shuffled_dataset.select(range(int(0.1 * len(shuffled_dataset))))

def preprocess_function(examples):
    # Extrair inputs e targets
    inputs = examples["title"]  # Usando 'title' como entrada
    targets = examples["content"]  # Usando 'content' como alvo

    # Tokenizar inputs
    model_inputs = tokenizer(inputs, padding="max_length", truncation=True)

    # Tokenizar targets e processar labels
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, padding="max_length", truncation=True)["input_ids"]

    # Garantir que labels tenham o mesmo comprimento e formato
    model_inputs["labels"] = labels

    return model_inputs

# Tokenizar o dataset
tokenized_datasets = dataset.map(preprocess_function, batched=True)

train_dataloader = DataLoader(tokenized_datasets, batch_size=16, shuffle=True, pin_memory=True)

training_args = TrainingArguments(
    output_dir="./data/results/v2",
    learning_rate=3e-5,
    per_device_train_batch_size=3,  # Reduzir o batch size
    per_device_eval_batch_size=3,   # Reduzir o batch size
    num_train_epochs=1,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=1000,
    fp16=True,  # Desabilitar mixed precision training
    gradient_accumulation_steps=4,  # Aumentar o número de acumulações para reduzir o uso de memória
)

# Inicializar o Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets
)

torch.cuda.empty_cache()
torch.cuda.ipc_collect()
trainer.train()

trainer.save_model("./data/bert-v2")
tokenizer.save_pretrained("./data/bert-tokenizer-v2")