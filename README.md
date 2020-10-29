_Implementação do EP2 da matéria MAC5725 - Linguística Computacional (2020)_

Objetiva-se realizar análise de sentimentos em cinco níveis (muito negativo, negativo, neutro, positivo, muito negativo) no banco de dados [B2W-Reviews01](https://github.com/b2wdigital/b2w-reviews01) que consiste em avaliações de mais de 130k compras online no site Americanas.com e esta disponivel no github sob a licensa CC BY-NC-SA 4.01.

* Modelo:

Arquiteturas de encoder com uma camada (1) LSTM unidirecional e (2) bidirecional serão testadas, seguidas de uma rede densa para classificação de sentimentos em cinco níveis.

* Como utilizar:

Instalar todas as dependências necessárias seguindo o [este tutorial]().

- Colocar o banco de dados [B2W-Reviews01](https://github.com/b2wdigital/b2w-reviews01) em data/ .

- baixar o embedding [word2vec](http://www.nilc.icmc.usp.br/embeddings) (cbow ou skip-gram) pretreinado para português em 50 dimensões.

- criar o embedding limitado em 200k palavras com:

'''python
# Pré-processa embedding
N =  200001
with open("cbow_s50.txt", "r") as file:
    head = [next(file) for x in range(N)]

head[0] = str(N-1)+ " " + "50"+ "\n" # Conserta contagem de palavras
with open("word2vec_200k.txt", "w") as file:
    for line in head:
        file.write(line)
'''

- colocar word2vec_200k.txt em data/

_ data_prep.py _

- Processar os dados do arquivo csv, separa os campos "review_text" e "overall_rating". Separa o banco de dados em treino, validação em teste nas proporções (65%-15%-20%) Pode gerar um número limitado de exeplos N (0 < N < 130000).

Exemplo: Gerar banco de dados com 10k amostras

''' python3 data_prep.py --inpath ../data --dataset B2W-Reviews01 --outpath ../data -N 10000 '''

_ train.py _

- Definir modelo de acordo com configurações de linha de comando, treina e testa no banco de dados processado. Gráfico de acurácias de treino e validação é gerado em doc/, bem como a acurácia do teste com o melhor modelo.

Parâmetros principais:

'''
--inpath : diretório do banco de dados
--dataset : nome do dataset processado <nome>_<N>
--embedding : nome do arquivo com word2vec pretreinado (sem o .txt).
--dropout : taxa de dropout da camada densa <float ex.:0.1>
-bidirectional: se adicionar esse parametro a camada lstm vira bidirecional
'''

Parâmetros adicionais do modelo já estão definidos por default, mas podem ser alterados via linha de comando (ver as definições de argparse em train.py).

Exemplo:

''' python3 train.py --dataset B2W-Reviews01_10000 --inpath ../data --embedding word2vec_200k --dropout 0.2 -bidirectional
 '''

