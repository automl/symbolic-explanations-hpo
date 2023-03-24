# Symbolic Explanations for Hyperparameter Optimization


## Installation

You can create an environment with the required packages using anaconda and the `environment.yml` 
file as demonstrated in the following:

```
conda env create -f environment.yml
conda activate symb_expl
```

To install HPOBench, please run the following after activating the environment:
```
git clone https://github.com/automl/HPOBench.git
cd HPOBench 
pip install .
```

## Running Experiments

In the following, we describe how to run the experiments. The overall process consists of the following steps: 
1. Run the the Bayesian Optimization-powered Hyperparameter Optimization tool SMAC and collect (a) the meta-data consisting of the evaluated configurations
and their performance and (b) the final surrogate model.
2. Learn a symbolic regression model on either (a) the collected meta-data, or (b) randomly sampled
configurations, which are evaluated using the true cost function, or (c) randomly sampled
configurations, whose performance is estimated using the Gaussian process.

### Sample Collection

Collecting the samples as described in step 1 can be run for a single model, hyperparameter-combination, and dataset, 
by running

```
python run_sampling_hpobench.py --job_id 0
```

where `job_id` is an index to iterate over a list containing all models, hyperparameter-combinations, and datasets.
By setting the option `use_random_samples` in the script to True, the script furthermore allows to collect randomly 
sampled configurations and evaluate their performance. When setting the option `evaluate_on_surrogate` in the script 
to True, the script will collect random samples, but estimated their performance using the Gaussian process. Please
note that the BO sampling needs to be run beforehand to provide the Gaussian process models.

### Symbolic Regression

Fitting the symbolic regression model as described in step 2 can be run for a single model, hyperparameter-combination, 
and dataset, by running

```
python run_symbolic_explanation_hpobench.py.py --job_id 0
```

By default, the symbolic regression will be fitted on the samples collected during Bayesian Optimization (a).
By setting the option `use_random_samples` in the script to True, the symbolic regression will be fitted on the randomly 
sampled configurations (b). When setting the option `evaluate_on_surrogate` in the script to True, the symbolic regression
will be fitted on the random samples with Gaussian process performance estimates (c). 

### Gaussian Process Baseline

Furthermore, the predictions of the Gaussian process model can be obtained by running:

```
python run_surrogate_explanation_hpobench.py.py.py --job_id 0
```
