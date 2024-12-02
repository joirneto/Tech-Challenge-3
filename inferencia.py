from transformers import BartTokenizer, BartForConditionalGeneration, GenerationConfig

tokenizer = BartTokenizer.from_pretrained("./models-tuned/bart/tokenizer")
model = BartForConditionalGeneration.from_pretrained('./models-tuned/bart/model')

# Definir o decoder_start_token_id e pad_token_id
model.config.decoder_start_token_id = tokenizer.cls_token_id
model.config.pad_token_id = tokenizer.pad_token_id
model.config.bos_token_id = tokenizer.cls_token_id
model.config.eos_token_id = tokenizer.sep_token_id

# Criar a configuração de geração
generation_config = GenerationConfig(
    decoder_start_token_id=tokenizer.cls_token_id,
    max_length=256,
    num_beams=4,
    no_repeat_ngram_size=2,
    top_k=20,
    top_p=0.0,
    do_sample=True,
    temperature=0.3
)

# Função de tradução
def generate_response(text, max_length=100):
    # Codificar o texto
    input_ids = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)["input_ids"]

    # Gere a tradução com a configuração de geração
    outputs = model.generate(input_ids=input_ids, generation_config=generation_config)

    # Decodifique a saída gerada
    decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return decoded_output

# Texto de exemplo para tradução
text = "what's the description of 'How to Babysit a Grandma'?"

print(f"Input:: {text}")
print(f"Output: {generate_response(text)}")
