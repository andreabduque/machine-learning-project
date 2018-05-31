import pandas as pd
import sys
sys.path.append('aqui bota o caminho da pasta que o arquivo a ser importado está inserido') # o meu foi esse: /home/gustavo/avulso/machine-learning-project/app/parzen
from parzen import Parzen as p

df = pd.read_csv('../iris.data')
teste_joao = p(df)

print(teste_joao.hello_world_joao())