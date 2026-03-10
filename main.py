import numpy as np
import pandas as pd
# Preparação dos dados

vocabulario = {"o": 0, "banco": 1, "bloqueou": 2, "cartao":3}


df_vocabulario = pd.DataFrame(list(vocabulario.items()), columns=['palavra', 'id'])

print(df_vocabulario)