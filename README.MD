Variational Inference for HDP-HMM model

## How to run

### Setting up a virtual environment

```angular2html
python -m venv venv
```

For Linux, run

```angular2html
source venv/bin/activate
```

For Windows, run

```angular2html
venv\Scripts\activate
```

### Install packages

```angular2html
pip install -r requirements.txt
```

### Code structure

+ `synthetic_dataset_hmm.ipynb` generates synthesis datasets, and the datasets are saved under `/data` directory.
+ `main.py` is the main part of VI algorithm. The variational parameters for `z` (hidden variable) are automatically
  saved every 20 steps.
+ `eval.ipynb` contains some exploration in evaluation, though not very successful.