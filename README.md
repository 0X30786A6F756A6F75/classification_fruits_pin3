# Treinamento do Modelo e script de comparação
## Depedências necessárias para rodar modelo e script de comparação
Necessário:
python
pip

## Passo a Passo
1. Ativando o ambiente virtual
```bash 
$ cd back/
$ python3 -m venv env
$ source env/bin/activate
```
2. Instalando as depedências
```bash
$ pip install -r requirements.txt
```

3. Rodando os algoritmos
```bash
$ python tree.py
$ python forest.py
```

4. Rodando o script de comparação
```bash
$ python compare.py
```

# Instação FRONT e BACK
## VIA DOCKER 
Necessário:
docker
docker-compose

1. Buildar container
```bash
$ docker-compose build
```
2. Rodar 
```bash
$ docker-compose up
```

Alternativa para builda e roda em sequência:
```bash
$ docker-compose up --build
```
A aplicaçao Web pode ser acessada por este [link](http://localhost:3000)
Ou apenas abrindo o Localhost na porta 3000

## MANUAL

### Roda back
1. Caminhe até o projeto de back
```bash
$ cd back
```
2. Ative o ambiente virtual
```bash
$ source env/bin/activate
```
3. Instalar depedências
```bash
$ pip install -r requiremets.txt
```

4. Rodar projeto
```bash
$ flask run
```
---
### Roda front
1. Caminhe até o projeto de front
```
$ yarn install
```
2. Rode o projeto 
```bash
$ yarn start
```
