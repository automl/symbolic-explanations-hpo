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
cd ..
```


## tl;dr: Summary of Commands to Reproduce the Results

To run the experiments for reproducing the results shown in the paper, we suggest the following
order of commands. To create the raw results, first run:
```
python run_sampling_hpobench.py --job_id 0 --run_type smac
python run_sampling_hpobench.py --job_id 0 --run_type rand
python run_sampling_hpobench.py --job_id 0 --run_type surr
python run_symbolic_explanation_hpobench.py --job_id 0 --run_type smac
python run_symbolic_explanation_hpobench.py --job_id 0 --run_type rand
python run_symbolic_explanation_hpobench.py --job_id 0 --run_type surr
python run_surrogate_explanation_hpobench.py --job_id 0
```

`job_id 0` will run the experiments for logistic regression with hyperparameters `alpha` and `eta0` on the 
dataset blood-transfusion-service-center. To reproduce the raw results for all models, hyperparameter 
combinations, and datasets showed in the paper, the above commands need to be run for `job_id` between 0-39.

After running the above commands, to calculate metrics and create plots, then run:
```
python metrics_hpobench.py
python plot_learning_curves_hpobench.py
python plot_complexity_vs_rmse.py
python plot_2d_hpobench.py
```

To limit the number of models and datasets to create plots for, you can adapt those in `utils/hpobench_utils`.

## Details on Running the Experiments

In the following, we describe how to run the experiments. The overall process consists of the following steps: 
1. Run the Bayesian optimization-powered hyperparameter optimization tool SMAC and collect (a) the meta-data consisting of the evaluated configurations
and their performance and (b) the final surrogate model.
2. Learn a symbolic regression model on either (a) the collected meta-data, or (b) randomly sampled
configurations, which are evaluated using the `True` cost function, or (c) randomly sampled
configurations, whose performance is estimated using the Gaussian process.

### Collection of Training Samples for the Symbolic Regression

Collecting the samples as described in step 1 can be run for a single model, hyperparameter-combination, and dataset, 
by running

```
python run_sampling_hpobench.py --job_id 0 --run_type smac
```

where `job_id` is an index to iterate over a list containing all models, hyperparameter-combinations, and datasets.
Which models and datasets should be included in the list can be defined in `utils/hpobench_utils`. 
By default, one hyperparameter-combination is evaluated per model and dataset. This can be adapted by modifying the 
parameter `max_hp_comb` inside the script.

By setting `run_type` to `rand`, the script furthermore allows to collect randomly sampled configurations and evaluate 
their performance. When setting `run_type` to `surr`, the script will collect random samples, but estimated their 
performance using the Gaussian process. Please  note that, in the latter case, the BO sampling needs to be run 
beforehand to provide the Gaussian process models.

### Symbolic Regression

Fitting the symbolic regression model as described in step 2 can be run for a single model, hyperparameter-combination, 
and dataset, by running

```
python run_symbolic_explanation_hpobench.py --job_id 0 --run_type smac
```

This way, the symbolic regression will be fitted on the samples collected during Bayesian Optimization (a).
By setting `run_type` to `rand`, the symbolic regression will be fitted on the randomly 
sampled configurations (b). When setting `run_type` to `surr`, the symbolic regression
will be fitted on the random samples with Gaussian process performance estimates (c). 

### Gaussian Process Baseline

Furthermore, the predictions of the Gaussian process model can be obtained by running:

```
python run_surrogate_explanation_hpobench.py --job_id 0
```

### Metrics

To average metrics over different seeds and combine them in a table for all specified models, 
hyperparameter-combinations, and datasets, run
```
python metrics_hpobench.py
```

### Plots

Plots will be created for all specified models, hyperparameter-combinations, and datasets. Thus, the experiments
described above need to be run for all of them before creating the plots.

To create plots showing the RMSE between the cost predicted by the symbolic regression and the true cost for
different numbers of samples, run
```
python plot_learning_curves_hpobench.py
```

To create plots showing the RMSE between the cost predicted by the symbolic regression and the true cost for different
values of the parsimony coefficient, run
```
python plot_complexity_vs_rmse.py
```

To create plots showing several representations of the HPO loss landscape, run
```
python plot_2d_hpobench.py
```
