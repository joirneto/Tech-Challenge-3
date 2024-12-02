from datasets import load_dataset
from transformers import Trainer, TrainingArguments,  BartTokenizer, BartForConditionalGeneration
import torch
from transformers import get_linear_schedule_with_warmup
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from accelerate import Accelerator
from torch.utils.data import DataLoader

# Dataset
file_path = './data/trn_sanitized.json'
all_dataset = load_dataset('json', data_files=file_path)
shuffled_dataset = all_dataset['train'].shuffle(seed=42)

# Pegar 10% do dataset
dataset = shuffled_dataset.select(range(int(0.1 * len(shuffled_dataset))))

# Inicializar Accelerator
accelerator = Accelerator(mixed_precision='fp16')

# Carregar o tokenizer e o modelo BART
model_name = 'facebook/bart-base'  # Caso tenha mais recursos, pode usar 'facebook/bart-large'
tokenizer = BartTokenizer.from_pretrained(model_name)
model = BartForConditionalGeneration.from_pretrained(model_name)

# Definir
model.config.decoder_start_token_id = tokenizer.cls_token_id
model.config.pad_token_id = tokenizer.pad_token_id
model.config.bos_token_id = tokenizer.cls_token_id
model.config.eos_token_id = tokenizer.sep_token_id

# Configuração opcional para evitar avisos
model.config.vocab_size = len(tokenizer) 

def preprocess_function(examples):
    # Extrair inputs e targets
    inputs = examples["title"]
    targets = examples["content"]
    
    max_length = 512  # Definir tamanho máximo
    # Tokenizar inputs
    model_inputs = tokenizer(inputs, padding="max_length", truncation=True, max_length=max_length)
    
    # Tokenizar targets e processar labels
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, padding="max_length", truncation=True, max_length=max_length)["input_ids"]
    
    # Substituir tokens de padding por -100 nos labels
    labels = [[(label if label != tokenizer.pad_token_id else -100) for label in label_seq] for label_seq in labels]
    
    model_inputs["labels"] = labels
    return model_inputs

# Tokenizar o dataset
tokenized_datasets = dataset.map(preprocess_function, batched=True)

train_dataloader = DataLoader(tokenized_datasets, batch_size=16, shuffle=True, pin_memory=True)

training_args = TrainingArguments(
    output_dir="./results",
    learning_rate=3e-5,
    per_device_train_batch_size=8,  
    per_device_eval_batch_size=8, 
    num_train_epochs=1,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=1000,
    fp16=True,  
    gradient_accumulation_steps=4, 
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

trainer.save_model("./models-tuned/bart/model")
tokenizer.save_pretrained("./models-tuned/tokenizer")