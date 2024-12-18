# POS FIAP ALURA - IA PARA DEVS
## Tech Challenge Fase 3
### Integrantes Grupo 26

- André Philipe Oliveira de Andrade(RM357002) - andrepoandrade@gmail.com
- Joir Neto (RM356391) - joirneto@gmail.com
- Marcos Jen San Hsie(RM357422) - marcosjsh@gmail.com
- Michael dos Santos Silva(RM357009) - michael.shel96@gmail.com
- Sonival dos Santos(RM356905) - sonival.santos@gmail.com

Video(Youtube): https://youtu.be/3zALZlBIz3M

Github: https://github.com/apandrade/tech-challenge3

# Fine-tuning do Modelo BART com o Dataset AmazonTitles-1.3MM

Este projeto tem como objetivo realizar o fine-tuning do modelo BART utilizando o dataset [AmazonTitles-1.3MM](https://huggingface.co/datasets/beTinti/AmazonTitles-1.3MM), explorando soluções de baixo custo para viabilizar o treinamento em ambientes de hardware limitado.

---

## 1. Escolha do Ambiente de Treinamento

Devido a limitações do Google Colab, como desconexões frequentes durante longos períodos de treinamento, a equipe optou por utilizar hardware local para explorar o menor custo viável. Apesar da possibilidade de usar ambientes de nuvem mais poderosos, a decisão foi operar em ambientes mais modestos. Os equipamentos utilizados foram:

- **GPU RTX 3060 12GB**: Capaz de treinar o modelo com 100% do dataset em **45 horas e 43 minutos**.
- **GPU RTX 3500 12GB**: Obteve desempenho similar à RTX 3060.
- **GPU RTX A1000 4GB**: Considerada inviável devido à baixa memória, impossibilitando o treinamento completo.

---

## 2. Escolha do Modelo

Após testes com os modelos **BART**, **BERT**, **RoBERTa** e **T5-small**, o modelo BART foi escolhido. Os testes com 5% e 10% do dataset demonstraram que o BART proporcionou respostas de maior qualidade, justificando sua seleção para o treinamento com 100% do dataset.

---

## 3. Limpeza dos Dados

Dois scripts foram criados para sanitização dos dados:
- **data-sanitizer.py**: Realizou uma limpeza inicial, mas a equipe identificou entradas duplicadas no dataset.
- **data-sanitizer-final.py**: Um segundo script foi desenvolvido para remover completamente duplicatas antes do treinamento final com 100% dos dados.

---

## 4. Parametrização do Treinamento

O treinamento foi conduzido com as seguintes configurações:

- **Modelo**: BART
- **Framework**: PyTorch.
- **Dataset**: AmazonTitles-1.3MM (80% treino, 20% validação).
- **Hiperparâmetros**:
  - Batch size: 14
  - Learning rate: 5e-5
  - Scheduler: Linear
  - Otimizador: AdamW ( padrão da biblioteca Hugging Face )
  - Epochs: 3
  - Gradiente acumulado: 4
  - Tokenização: Truncamento em 512 tokens.
- **Checkpoints**: Salvos a cada 1.000 passos.

---

## 5. Métricas Utilizadas

Para avaliação do modelo utilizamos as métricar padrão do BART:

- **Loss**: Monitoramento da perda no treinamento e validação.
- **Logging Steps**: Configurado para logar a cada 50 passos, registrando métricas padrão, como a perda.
---

## 6. Inferências

As inferências foram realizadas com 10 entradas de teste. O arquivo `output.txt` na pasta `inferencias` contém os resultados.

---

## 7. Análise das Inferências

### Observações sobre o arquivo `output.txt`:

- **Pontos Positivos**:
  - As respostas são coerentes e consistentes.
  - O modelo mostrou boa capacidade de generalização.
- **Pontos Negativos**:
  - Algumas respostas são genéricas ou repetitivas.


### Sugestões de melhoria

1. Melhorar o pré-processamento:

   - Adicionar manipulação de textos muito curtos ou vazios para evitar erros na tokenização.
   - Ajustar dinamicamente o max_length com base na distribuição do comprimento dos dados.

2. Aprimorar os hiperparâmetros:

   - Testar diferentes valores para learning_rate, como 3e-5 ou 1e-5, e usar 
   - agendadores de aprendizado, como cosine schedule ou linear warmup.

3. Explorar mais dados:

   - Adicionar validação cruzada com split dinâmico entre treino e validação para maior robustez.
   - Considerar um conjunto de validação para monitorar a performance do modelo durante o treinamento.

4. Avaliar métricas:

   - Implementar métricas customizadas para monitorar a qualidade da geração de texto, como BLEU, ROUGE, perplexidade, BERTScore, Coherence Score, Relevance-based Metrics, Human Evaluation.

5. Aprimorar o hardware:

   - Habilitar uso de múltiplas GPUs (se disponíveis) para acelerar o treinamento.
   - Ajustar o número de núcleos e threads (num_proc e dataloader_num_workers) para aproveitar ao máximo a CPU.

6. Augmentação de dados:

   - Utilizar técnicas de augmentação textual, como substituição de sinônimos ou parafraseamento, para enriquecer o dataset.

7. Fine-tuning mais detalhado:

   - Treinar com mais épocas e introduzir técnicas como early stopping para evitar overfitting.
   - Utilizar estratégias de regularização adicionais, como dropout dinâmico.

8. Logs e monitoramento:

   - Incorporar o wandb ou TensorBoard para um acompanhamento mais detalhado das métricas durante o treinamento.
   - Se implementadas, essas melhorias podem ajudar a obter um desempenho significativamente melhor no modelo treinado.

### Conclusão

O desempenho foi satisfatório, especialmente considerando o ambiente de hardware limitado. O modelo entrega resultados promissores e pode ser otimizado em iterações futuras. Podemos utilizar múltiplas métricas (por exemplo, BERTScore, ROUGE e Perplexidade) para capturar diferentes aspectos da qualidade das respostas. Também é possível priorizar métricas baseadas em embeddings ou aprendizado contextual, como BERTScore e COHERENCE, além de complementar avaliações automáticas com avaliações humanas regulares.

---

## Como Reproduzir

1. Clone o repositório:
   ```bash
   git clone <url-do-repositorio>
   cd <nome-do-repositorio>
   ```

2. Instale as dependências:
   ```bash
   pip install -r requirements.txt
   ```

3. Execute a limpeza dos dados:
   ```bash
   python data-sanitizer-final.py
   ```

4. Inicie o treinamento:
   ```bash
   python fine-tunning-bart-final.py
   ```

5. Realize as inferências:
   ```bash
   python inferencias/inferencia-bart-final.py
   ```

---

Este projeto reflete o compromisso da equipe em equilibrar custo e desempenho, oferecendo uma base sólida para trabalhos futuros.
