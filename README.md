# Base Sentence-Level DSRE
Baselines of sentence-level distantly supervised relation extraction models.

## Requirements
* python==3.8
* pytorch==1.6
* numpy==1.19.1
* tqdm==4.48.2
* scikit_learn==0.23.2

## Data
Download the dataset from [here](https://github.com/thunlp/HNRE/tree/master/raw_data), and unzip it under `./data/`.

## Train and Test
```
python main.py
```

## Experimental Result

| Encoder | P@100 | P@200 | P@300 | Mean | AUC |
| :-----: | :---: | :---: | :---: | :--: | :-: |
| PCNN | :---: | :---: | :---: | :--: | :-: |
| CNN | 62.0 | 60.0 | 61.7 | 61.1 | 34.8 |
| BiGRU | :---: | :---: | :---: | :--: | :-: |
