import logging
from typing import Any, override
import polars as pl
from sklearn.neural_network import MLPClassifier
from flowcean.core import Model, SupervisedLearner
from flowcean.models.sklearn import SciKitModel

logger = logging.getLogger(__name__)

class MultiLayerPerceptronClassifier(SupervisedLearner):

    def __init__(
            self,
            *args: Any,
            **kwargs: Any,
    ) -> None:
        self.perceptron = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)

    @override
    def learn(
        self,
        inputs: pl.DataFrame,
        outputs: pl.DataFrame,
        ) -> Model:
            self.perceptron.fit(inputs, outputs)
            logger.info("Model is trained")
            return ScikitModel(self.perceptron, outputs.columns[0])