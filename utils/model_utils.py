from utils.model_wrapper import SVM, MLP, BDT, DT, LR


def get_models():
    return ["MLP", "SVM", "BDT", "DT", "LR"]


def get_hyperparams(model_name):
    if model_name == "MLP":
        hyperparams = [
            "optimize_n_neurons",
            "optimize_n_layer",
            "optimize_learning_rate_init",
            "optimize_max_iter"
        ]
    elif model_name == "SVM":
        hyperparams = [
            "optimize_C",
            "optimize_degree",
            "optimize_coef",
            "optimize_gamma"
        ]
    elif model_name == "BDT":
        hyperparams = [
            "optimize_learning_rate", "optimize_n_estimators"
        ]
    elif model_name == "DT":
        hyperparams = [
            "optimize_max_depth", "optimize_min_samples_leaf"
        ]
    elif model_name == "LR":
        hyperparams = [
            "optimize_alpha", "optimize_eta0"
        ]
    else:
        hyperparams = None
    return hyperparams


def get_classifier_from_run_conf(model_name, run_conf):
    if model_name == "MLP":
        classifier = MLP(**run_conf)
    elif model_name == "SVM":  # set lower tolerance, iris (stopping_criteria=0.00001)
        classifier = SVM(**run_conf)
    elif model_name == "BDT":
        classifier = BDT(**run_conf)
    elif model_name == "DT":
        classifier = DT(**run_conf)
    elif model_name == "LR":
        classifier = LR(**run_conf)
    else:
        print(f"Unknown model: {model_name}")
        classifier = None
    return classifier


def get_classifier(model_name, seed=0):
    if model_name == "MLP":
        classifier = MLP(
            optimize_n_neurons=True,
            optimize_n_layer=True,
            optimize_batch_size=False,
            optimize_learning_rate_init=False,
            optimize_max_iter=False,
            seed=seed,
            data_set_name="digits"
        )
    elif model_name == "SVM":  # set lower tolerance, iris (stopping_criteria=0.00001)
        classifier = SVM(
            optimize_C=True,
            optimize_degree=False,
            optimize_coef=True,
            optimize_gamma=False,
        )
    elif model_name == "BDT":
        classifier = BDT(optimize_learning_rate=True, optimize_n_estimators=True)
    elif model_name == "DT":
        classifier = DT(optimize_max_depth=True, optimize_min_samples_leaf=True)
    else:
        classifier = None
    return classifier