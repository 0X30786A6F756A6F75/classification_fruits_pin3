# Instalação: DOCKER
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

# Instalação: MANUAL

## Roda back
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
## Roda front
1. Caminhe até o projeto de front
```
$ yarn install
```
2. Rode o projeto 
```bash
$ yarn start
```
