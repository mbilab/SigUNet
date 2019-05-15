# SigUNet

A signal peptide predictor based on deep learning.

## Python version support

Python 3.6 **only**

## Installation

For CPU: `pip3 install -r requirement.cpu.txt`

For GPU (suggest): `pip3 install -r requirement.gpu.txt`

## Dataset

- SignalP dataset : [SignalP 4.0 Server](http://www.cbs.dtu.dk/services/SignalP-4.0/data.php)
- SPDS17 dataset : [DeepSig dataset](https://deepsig.biocomp.unibo.it/deepsig/default/software)

Notice : Put files of SignalP dataset into `data/SignalP` and put files of SPDS17 dataset into `data/SPDS17`

## Training

1. Preprocessing dataset:

```
./build_SignalP_dataset.py [dataset]
./build_SPDS17_dataset.py [dataset]
```

\[dataset\]: euk, gram+, gram-

2. Train models:

```
./nested_cv.py params/params.example.json [model_path]
```

For custom search space of m, n and filer size descripted in the paper, you can use your params file.
Please make sure the format of your file is the same as `params/params.example.json`.

It will show the result of SignalP dataset after training.

3. Evaluate SPDS17 dataset

```
./evaluate_SPDS17.py [model_path]
```
