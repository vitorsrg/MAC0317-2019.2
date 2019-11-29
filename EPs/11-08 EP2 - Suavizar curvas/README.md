# MAC0317 - Introdução ao Processamento de Sinais Digitais

**Prof.** Marcel Parolin Jackowski

**Aluno** Vitor Santa Rosa Gomes, 10258862, vitorssrg@usp.br

## EP2 - Suavização de curvas

O programa `main.py` é capaz de detectar, filtrar e salvar curvas em imagens.

### Funcionalidades

As funcionalidades de `main.py` estão disponíveis com `python main.py -h`:

```
usage: main.py  imgrelpath channel intensity threshold 
                ctrrelpath ctrfltrelpath

Smooth digital curves.

positional arguments:
imgrelpath      image file
channel         RGBA channel to work with (in 0..3)
intensity       pixel intensity to contour (in 0..255)
threshold       frequency threshold to filter 
                (in -100..-1 or 1..100)
ctrrelpath      contour detected from the original image
ctrfltrelpath   filtered contour
```

### Dependências

O programa `main.py` foi testado com `python3.7`. A fim de ter acesso a todos os recursos, é necessário:

```
pip install scipy numpy imageio
```

### Compilando os exemplos

Veja os comentários e respostas ao questionário em [./smoother2d.ipynb](./smoother2d.ipynb) ou [./smoother2d.html](./smoother2d.html).
