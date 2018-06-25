# Detecção de Fraude via Emails da Enron

## Introdução

Este é o projeto final das aulas de Introdução ao Aprendizado de Máquinas do curso de _Data Science for Business_ da
Udacity. O principal objetivo deste projeto é desenvolver um algoritimo que detecte pessoas envolvidas no escandalo
da Enron baseado num conjunto de dados finnceiros e de emails de funcionários de alto escalão e diretores da Enron.

## Seleção das _features_

## Outliers

A primeira tarefa deste projeto é identificar e remover possíveis _outliers_ da base de dados usadas, evitando
interferencia destes dados na análise.

Já em uma primeira analise é possivel identicar um outlier com base nas informações de salário e bonus, referente aos
registros de totalizadores dos dados financeiros. Por se tratar de um totalizador, este registro possui informações
discrepantes quando comparados aos demais registros dos dataset.

Este registro será excluido de nosso conjunto de dados:

```{python}
data_dict.pop("TOTAL", 0)
```