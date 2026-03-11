import numpy as np
import pandas as pd

## Passo 01: Preparação dos Dados

# Definir vocabulario
vocabulario = {"o": 0, "banco": 1, "bloqueou": 2, "cartao":3}

# Criar um Dataframe
df_vocabulario = pd.DataFrame(list(vocabulario.items()), columns=['palavra', 'id'])

#print(df_vocabulario)

# Definindo uma frase e converta-a em uma lista de IDs
frase_texto = ["o", "banco", "bloqueou", "o", "cartao"]

input_ids = [vocabulario[palavra] for palavra in frase_texto]

#print(f"\nFrase do texto: {frase_texto}")
#print(f"Lista de IDs: {input_ids}")

# Inicializar a Tabela de Embeddings
tamanho_vocabulario = len(vocabulario)

d_model = 64

embedding_table = np.random.randn(tamanho_vocabulario, d_model)

#print(f"Shape da Tabela de Embeddings: {embedding_table.shape}")

# Criar o tensor de entrada final (X)

X_embedded = embedding_table[input_ids]

#print(f"Shape após a busca (SequenceLength, d_model): {X_embedded.shape}")

# o formato tridimensional

X = np.expand_dims(X_embedded, axis=0)

#print(f"Shape final do Tensor X (BatchSize, SeqLen, d_model): {X.shape}")


## Passo 2: Motor matematico

dk = d_model # dm_model é 64

Q_peso = np.random.randn(d_model, dk)
K_peso = np.random.randn(d_model, dk)
W_peso = np.random.randn(d_model, dk)

Q = X @ Q_peso  
K = X @ K_peso 
V = X @ W_peso  

K_transposto = K.swapaxes(-1, -2) 
scores = Q @ K_transposto

limitador = scores / np.sqrt(dk)

def Softmax(x):

    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / np.sum(e_x, axis=-1, keepdims=True)


atensao_pesos = Softmax(limitador)

Z = atensao_pesos @ V

X_residual = X + Z  # Somamos a entrada original com a saída da atenção

def Normalizar(x, epsilon=1e-6):
    media = np.mean(x, axis=-1, keepdims=True) 
    variancia = np.var(x, axis=-1, keepdims=True) 
    return (x - media) / np.sqrt(variancia + epsilon)

X_norm1 = Normalizar(X_residual)

d_expandida = 256

pesos_ff1 = np.random.randn(d_model, d_expandida)
vies_ff1 = np.zeros((1, 1, d_expandida)) 

cd_oculta = (X_norm1 @ pesos_ff1) + vies_ff1


cd_ativacao = np.maximum(0, cd_oculta)

pesos_ff2 = np.random.randn(d_expandida, d_model)
vies_ff2 = np.zeros((1, 1, d_model))

ffn_output = (cd_ativacao @ pesos_ff2) + vies_ff2

## Passo 3: Empilhando tudo

def SelfAttention(X, d_model=64):
    dk = d_model
    Q_peso = np.random.randn(d_model, dk)
    K_peso = np.random.randn(d_model, dk)
    V_peso = np.random.randn(d_model, dk)
    
    Q = X @ Q_peso
    K = X @ K_peso
    V = X @ V_peso
    
    scores = Q @ K.swapaxes(-1, -2)
    limitador = scores / np.sqrt(dk)
    
    # Usando a sua Softmax
    pesos = Softmax(limitador)
    return pesos @ V

def FFN(X_norm1, d_model=64):
    d_expandida = 256
    W1 = np.random.randn(d_model, d_expandida)
    b1 = np.zeros((1, 1, d_expandida))
    W2 = np.random.randn(d_expandida, d_model)
    b2 = np.zeros((1, 1, d_model))
    
    # Expansão + ReLU + Contraçaõ
    camada_oculta = np.maximum(0, (X_norm1 @ W1) + b1)
    return (camada_oculta @ W2) + b2

# Loop das 6 camadas:

X_atual = X 

print(f"Dimensão Inicial: {X_atual.shape}")

for i in range(6):

    X_att = SelfAttention(X_atual)
    
    X_norm1 = Normalizar(X_atual + X_att)
    
    X_ffn = FFN(X_norm1)
    
    X_out = Normalizar(X_norm1 + X_ffn)
    
    X_atual = X_out
    
    print(f"Camada {i+1} funcionou!")

print("-" * 30)
print(f"Representação Final shape: {X_atual.shape}")

