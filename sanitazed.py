import json

# Carregar o arquivo original
file_path = "./data/trn.json"
with open(file_path, "r") as file:
    data = [json.loads(line) for line in file]

# Filtrar as colunas "title" e "content"
filtered_data = [{"title": item["title"], "content": item["content"]} for item in data]

# Contagem de vazios e nulos
empty_titles = 0
null_titles = 0
empty_contents = 0
null_contents = 0

# Contar os valores vazios e nulos
for item in data:
    # Contando Titles
    if item['title'] == "":
        empty_titles += 1
    elif item['title'] is None:
        null_titles += 1
    
    # Contando Contents
    if item['content'] == "":
        empty_contents += 1
    elif item['content'] is None:
        null_contents += 1

total_titles = len(data)
total_contents = len(data) 

# Exibir os resultados
print(f"Total de Titles vazios: {empty_titles}")
print(f"Total de Titles nulos: {null_titles}")
print(f"Total de Contents vazios: {empty_contents}")
print(f"Total de Contents nulos: {null_contents}")
print(f"Total de Titles: {total_titles}")
print(f"Total de Contents: {total_contents}")

# Retirar as colunas "title" e "content" vazias
filtered_data = [item for item in data if item['title'] and item['content']]

# Contar os valores vazios e nulos apos a limpeza
# Contagem de vazios e nulos
empty_titles = 0
null_titles = 0
empty_contents = 0
null_contents = 0

for item in filtered_data:
    # Contando Titles
    if item['title'] == "":
        empty_titles += 1
    elif item['title'] is None:
        null_titles += 1
    
    # Contando Contents
    if item['content'] == "":
        empty_contents += 1
    elif item['content'] is None:
        null_contents += 1

total_titles = len(filtered_data)
total_contents = len(filtered_data) 

print(f"------------------------Após a limpeza------------------------")
# Exibir os resultados separados
print(f"Total de Titles vazios: {empty_titles}")
print(f"Total de Titles nulos: {null_titles}")
print(f"Total de Contents vazios: {empty_contents}")
print(f"Total de Contents nulos: {null_contents}")
print(f"Total de Titles: {total_titles}")
print(f"Total de Contents: {total_contents}")

# Salvar em um novo arquivo JSON
output_path = "trn_sanitazed.json"
with open(output_path, "w") as outfile:
    json.dump(filtered_data, outfile, indent=4)

print(f"Novo arquivo salvo em: {output_path}")
