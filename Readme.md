# ConvE

Implementation of ConvE proposed by Dettmers et al. in [Convolutional 2D Knowledge Graph Embeddings](https://arxiv.org/abs/1707.01476). You can find the official repository with knowledge graph datasets [here](https://github.com/TimDettmers/ConvE).

Implementation uses [PyTorch](http://pytorch.org/).

## Usage

### Preprocessing

```
usage: preprocess.py [-h] {train,valid} ...

Preprocess knowledge graph csv train/valid (test) data.

positional arguments:
  {train,valid}  mode
    train        Preprocess a training set
    valid        Preprocess a valid or test set

optional arguments:
  -h, --help     show this help message and exit
```

#### Training set

```
python preprocess.py train ../train.tsv
```

#### Validation set

```
python preprocess.py valid ../train.pkl ../valid.tsv
```

### Training

```
python train.py ../train.pkl ../valid.pkl
```

```
usage: train.py [-h] [--name NAME] [--batch-size BATCH_SIZE] [--epochs EPOCHS]
                [--label-smooth LABEL_SMOOTH]
                train_path valid_path

Train ConvE with PyTorch.

positional arguments:
  train_path            Path to training .pkl produced by preprocess.py
  valid_path            Path to valid/test .pkl produced by preprocess.py

optional arguments:
  -h, --help            show this help message and exit
  --name NAME           name of the model, used to create a subfolder to save
                        checkpoints
  --batch-size BATCH_SIZE
  --epochs EPOCHS
  --label-smooth LABEL_SMOOTH
```
