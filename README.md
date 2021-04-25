# struc2vec-airport-classifier
This repo aims at reproducing the classifier implemented in struc2vec, with the following protocols mentioned in the [paper](https://arxiv.org/pdf/1704.03165.pdf):

1. The classifier uses logistic regression with L2 regularization.
2. 80% of the nodes used for training.
3. Experiments are repeated 10 times.

## Set up
Make sure this repo is included the same directory as `struc2vec/`. Then run the following
```
python3 -m venv venv
source venv/bin/activiate
pip3 install -r requirements.txt
```

## Train and evaluate the classifier
Make sure you have the corresponding airports dataset embedding generated in directory `../struc2vec/emb/{usa/brazil/europe}-airports.emb`. Please refer to [struc2vec repo](https://github.com/leoribeiro/struc2vec) for the detailed run book.

We enable setting a few parameters in the args, for a full list you can use `python3 main.py -h`. An example usage:
```
python3 main.py --dataset brazil --epochs 20 --lr 0.01
```

## Miscellaneous
We were not able to reproduce the results for the basic struc2vec without any optimization in the paper. Feel free to play with the hyperparameters, and if you found the setting that can reproduce the results, please post in the issues :) 
