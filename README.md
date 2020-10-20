# Mini Decodable Information Bottleneck [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/YannDubs/Neural-Process-Family/blob/master/LICENSE) [![Python 3.6+](https://img.shields.io/badge/python-3.6+-blue.svg)](https://www.python.org/downloads/release/python-360/)

Note Bene: Still under construction!

This is a minimal repository for the paper [Learning Optimal Representations with the Decodable Information Bottleneck](https://arxiv.org/abs/2009.12789). The repository focuses on **practicality and simplicity**, as such there are some [differences](#differences-with-original-paper) with the original paper. For the full (and long) code see [Facebook's repository](github.com/facebookresearch/decodable_information_bottleneck).

The Decodable Information Bottleneck (DIB) is an algorithm to appproximate optimal representations, i.e.,  V-minimal V-sufficient representations. DIB is a generalization of the information bottleneck, which is simpler to estimate and provably optimal because it incorporate the classifier's architecture of interest V (e.g. linear classifier, 3 layer MLP).

## Install

0. Clone repository
1. Install [PyTorch](https://pytorch.org/)
2. `pip install -r requirements.txt`

Nota Bene: if you prefer I also provide a `Dockerfile` to install the necessary packages.

## Using DIB in Your Work
If you want to use DIB in your work, you should focus on `dib.py`. This module contains the loss `DIBLoss`, a wrapper around your encoder / model `DIBWrapper`, and a wrapper around your dataset `get_DIB_data` to add the index to the target. The wrapper can be used both in a single player game setting (i.e. to use DIB as a regularizer) or in a 2 player game setting (i.e. to pretrain an encoder using DIB). All you need is something like that:

```python
from dib import DIBWrapper, DIBLoss, get_DIB_data

V = SmallMLP # architecture of the classifier
model = DIBWrapper(V=V, Encoder=LargeMLP) # architecture of the encoder

loss = DIBLoss(V=V, n_train=50000) # needs to know training size
train(model,loss, get_DIB_data(CIFAR10))

# ------------------ CASE 1: USING DIB AS A REGULARIZER -------------------
# the model contains the encoder and classifier trained jointly 
# this corresponds to the single player game scenario 
predict(model)
# -------------------------------------------------------------------------

# ------------- CASE 2: USING DIB FOR REPRESENTATION LEARNING -------------
# the following code freezes the representation and resets the classifier. 
# This corresponds to the 2 player game scenario
model.set_2nd_player_()
# 2nd player is a usual deep learner => no more DIB (encoder is pretrained)
train(model,torch.nn.CrossEntropy(), CIFAR10)
# the model contains the DIB encoder and classifier trained disjointly
predict(model)
# -------------------------------------------------------------------------
```

The rest of the repository:
  - gives an example of how to put all together and evaluate the model
  - shows how to evaluate the model in 2 player game scenarios (including worst case, as in the paper) 

## Running DIB In This Repository

To run the minimal experiment in this repository run something along the lines of `python main.py name=test loss.beta=0,1,10 seed=123,124 -m`

Parameters:
  - `name`: name of the experiment
  - `loss.beta`: values of beta to run 
  - `seed`: seeds to ue in experiment
  - `-m`: sweeps over all the different hyperparameters, in the previous examples there were 3 beta values and 2 seeds so it will run 3*2=6 different models
  - you can modify all the parameters defined in `config.yaml`. For more information about everything you can do (e.g. running on SLURM, bayesian hyperparameter tunning, ...) check [pytorch-lighning's trainer](https://pytorch-lightning.readthedocs.io/en/latest/trainer.html#trainer-class-api) and [hydra](https://hydra.cc/docs/intro/).

Once all the model are trained / evaluated, you can plot the results using `python viz.py <name>`, this will load the results from the experiment `<name>` and save a plot in `results/<name>.png`. 


Example 1:
```
python main.py name=stochastic loss.beta=0,1e-3,1e-2,0.1,1,10,100,1000 seed=123,124,125 -m
python viz.py stochastic
```


![Simple DIB Results Stochastic](results/stochastic.png)


Example 2:
```
python main.py name=deterministic encoder.is_stochastic=False loss.beta=0,1e-3,1e-2,0.1,1,10,100,1000 seed=123,124,125 -m
python viz.py deterministic
```

![Simple DIB Results Deterministic](results/deterministic.png)

## Differences With Original Paper

As I said before, this is a simple implementation of DIB, which focuses on the concepts / simplicity / computational efficiency rather than the results. THe results will thus be a little worst than in the paper (but the trends should still hold). Here are the main differences with the full implementation:
- I use joint optimization instead of unrolling optimization (see Appx. E.2)
- I do not use y decompositions through base expansions (see Appx. E.5.)
- I share predictors to improve batch training (see Appx. E.6.)


## Cite
```
@incollection{dubois2020dib,
  title = {Learning Optimal Representations with the Decodable Information Bottleneck},
  author = {Dubois, Yann, and Kiela, Douwe  and Schwab, David J. and Vedantam, Ramakrishna},
  booktitle = {Advances in Neural Information Processing Systems 33},
  year = {2020},
  url = {https://arxiv.org/abs/2009.12789}
```