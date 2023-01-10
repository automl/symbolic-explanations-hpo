import numpy as np

from symbolic_metamodeling.pysymbolic.algorithms.symbolic_metamodeling import *

class MemorizingModel:

    def __init__(self):
        self.X = None
        self.y = None
        self.classes_ = None

    def fit(self, X, y):
        self.X = X
        self.y = y
        self.classes_ = np.unique(self.y)
        return self

    def predict(self, X):
        y_hat = list()
        for x in X:
            indices = np.where((self.X == x).all(axis=1))
            if len(indices) <= 0:
                raise Exception('Unknown sample.')
            index = indices[0]
            y_hat.append(self.y[index])

        return np.array(y_hat)

    def predict_proba(self, X):
        predictions = self.predict(X)
        all_probabilities = list()
        for prediction in predictions:
            probabilities_for_instance = np.zeros(self.classes_.shape)
            index_of_predicted_class = np.where(self.classes_ == prediction)
            probabilities_for_instance[index_of_predicted_class] = 1.0
            all_probabilities.append(probabilities_for_instance)
        return np.array(all_probabilities)


class SymbolicMetaModelWrapper:
    def __init__(self):
        self.metamodel = None

    def fit(self, X, y):
        memorizing_model = MemorizingModel()
        memorizing_model.fit(X=X, y=y)
        self.metamodel = symbolic_metamodel(model=memorizing_model, X=X, mode='regression')
        self.metamodel.fit(num_iter=1, batch_size=1, learning_rate=.01)
        return self

    def predict(self, X):
        return self.metamodel.evaluate(X)

    def expression(self):
        exact_expression, approx_expression = self.metamodel.symbolic_expression()
        return exact_expression


class SymbolicMetaExpressionWrapper:
    def __init__(self, model):
        self.metaexpression = model

    def predict(self, X):
        return self.metaexpression.evaluate(X)

    def expression(self):
        return self.metaexpression.expression()
