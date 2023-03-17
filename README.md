# Symbolic Regression for HPO


## Installation

You can create an environment with the required packages using anaconda and the `environment.yml` 
file as demonstrated in the following:

```
conda env create -f environment.yml
conda activate hpo_symb
```

To install HPOBench, please run the following after activating the environment:
```
git clone https://github.com/automl/HPOBench.git
cd HPOBench 
pip install .
```

## How to run

You can run the symbolic regression for the toy functions defined in functions.py as follows:

```
python run_toy_symb.py
```