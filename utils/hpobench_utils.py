from hpobench.benchmarks.ml.nn_benchmark import NNBenchmark
from hpobench.benchmarks.ml.svm_benchmark import SVMBenchmark


def get_model_name(benchmark) -> str:
    if isinstance(benchmark, NNBenchmark):
        return "NN"
    elif isinstance(benchmark, SVMBenchmark):
        return "SVM"
    else:
        raise Exception("Unknown benchmark.")
