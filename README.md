Baseado em [https://github.com/espnet/espnet]

## Instalação

Seguir as instruções de [instalação](https://espnet.github.io/espnet/installation.html)

## Executar os experimentos

Executar `egs2/ptbr/asr1/run_projeto.sh`

## Resultados


 ### Environments
python version: `3.7.9 (default, Aug 31 2020, 12:42:55)  [GCC 7.3.0]`
espnet version: `espnet 0.9.5`
pytorch version: `pytorch 1.6.0`

| Modelo      | Label   | %corr Palavras |        |      |          | %corr Caracteres |        |      |          | %corr Subwords |        |      |          |
|-------------|---------|:--------------:|--------|------|----------|:----------------:|--------|------|----------|:--------------:|--------|------|----------|
|             |         | CommonVoice    | LapsBM | Sid  | Voxforge | CommonVoice      | LapsBM | Sid  | Voxforge | CommonVoice    | LapsBM | Sid  | Voxforge |
| Rnn         | char    | 81.6           | 56.3   | 63.9 | 61.7     | 92.5             | 84.2   | 88.4 | 83.9     | -              | -      | -    | -        |
|             | subword | 88.2           | 23.4   | 26.7 | 41.7     | 92.7             | 56.0   | 59.2 | 67.8     | 87.8           | 26.8   | 32.5 | 43.3     |
| ConvRNN     | char    | 84.0           | 62.6   | 67.9 | 66.2     | 93.8             | 87.7   | 90.3 | 86.5     | -              | -      | -    | -        |
|             | bpe500  | 91.5           | 23.5   | 29.7 | 42.1     | 94.9             | 82.4   | 83.5 | 77.6     | 91.3           | 34.8   | 43.2 | 47.2     |
| Transformer | char    | 83.5           | 65.3   | 69.8 | 68.9     | 94.4             | 89.6   | 91.6 | 89.1     | -              | -      | -    | -        |
|             | bpe500  | 91.5           | 62.6   | 67.7 | 71.2     | 96.4             | 87.3   | 89.3 | 88.5     | 92.7           | 71.1   | 75.7 | 75.7     |
