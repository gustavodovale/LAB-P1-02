import numpy as np
import pandas as pd

## Passo 01: Preparação dos Dados

# Definir vocabulario
vocabulario = {"o": 0, "banco": 1, "bloqueou": 2, "cartao":3}

# Criar um Dataframe
df_vocabulario = pd.DataFrame(list(vocabulario.items()), columns=['palavra', 'id'])

print(df_vocabulario)

# Definindo uma frase e converta-a em uma lista de IDs
frase_texto = ["o", "banco", "bloqueou", "o", "cartao"]

input_ids = [vocabulario[palavra] for palavra in frase_texto]

print(f"\nFrase do texto: {frase_texto}")
print(f"Lista de IDs: {input_ids}")

# Inicializar a Tabela de Embeddings
tamanho_vocabulario = len(vocabulario)

d_model = 64

embedding_table = np.random.randn(tamanho_vocabulario, d_model)

print(f"Shape da Tabela de Embeddings: {embedding_table.shape}")

# Criar o tensor de entrada final (X)

X_embedded = embedding_table[input_ids]

print(f"Shape após a busca (SequenceLength, d_model): {X_embedded.shape}")

# o formato tridimensional

X = np.expand_dims(X_embedded, axis=0)

print(f"Shape final do Tensor X (BatchSize, SeqLen, d_model): {X.shape}")