import random
from collections.abc import Iterator
from typing import Self, override

import numpy as np
import polars as pl
from numpy.typing import NDArray

from flowcean.cli.logging import initialize_logging
from flowcean.environments.hybrid_system import (
    DifferentialMode,
    HybridSystem,
    State,
)
from flowcean.environments.train_test_split import TrainTestSplit
from flowcean.learners.regression_tree import RegressionTree
from flowcean.metrics.regression import MeanAbsoluteError, MeanSquaredError
from flowcean.strategies.offline import evaluate_offline, learn_offline
from flowcean.transforms.sliding_window import SlidingWindow

class Temperature(State):
    temperature: float

    acceleration: float
    step_count: int

    def __init__(self, temperature: float) -> None:
        self.temperature = temperature

    @override
    def as_numpy(self) -> NDArray[np.float64]:
        return np.array([self.temperature])
    
    @override
    @classmethod
    def from_numpy(cls, state: NDArray[np.float64]) -> Self:
        return cls(state[0])

type targetTemperature = float

class Heating(DifferentialMode[Temperature, targetTemperature]): #Temperature not defined
        rate_constant: float = 0.0075
        overheat_timeout: float = 1.0 #what is overheat timeout??

        @override
        def flow(
            self,
            t: float,
            state: NDArray[np.float64],
        ) -> NDArray[np.float64]:
            _=t
            return np.array([self.rate_constant])
        
        @override
        def transition(
            self,
            i: targetTemperature,
        ) -> DifferentialMode[Temperature, targetTemperature]:
            if self.state.temperature > i or self.t > self.overheat_timeout:
                return Cooling(t=0.0, state=self.state) #why reset t to 0.0
            return self
        
class Cooling(DifferentialMode[Temperature, targetTemperature]):
        rate_constant: float = 0.1
        cooldown_timeout: float = 1.0

        @override
        def flow(
            self,
            t: float,
            state: NDArray[np.float64],
        ) -> NDArray[np.float64]:
            _=t
            return np.array([-self.rate_constant])
        
        @override
        def transition(
            self,
            i: targetTemperature,
        ) -> DifferentialMode[Temperature, targetTemperature]:
            if self.state.temperature < i and self.t > self.cooldown_timeout:
                return Heating(t=0.0, state=self.state)
            return self
        
def randomly_changing_values(
            change_probability: float,
            minimum: float,
            maximum: float,
    ) -> Iterator[float]:
        value = random.uniform(minimum, maximum)
        while True:
            if random.random() < change_probability:
                value = random.uniform(minimum, maximum)
            yield value

def main() -> None:
    initialize_logging()

    target_temperatures = (
        (0.1 * i, temperature)
        for i, temperature in enumerate(
            randomly_changing_values(
                change_probability = 0.002,
                minimum = 19.0,
                maximum = 21.0,
            )
        )
    )

    environment = HybridSystem(
        initial_mode=Heating(t=0.0, state=Temperature(20)),
        inputs=target_temperatures,
        map_to_dataframe=lambda times, inputs, modes: pl.DataFrame(
            {
                "time": times,
                "target": inputs,
                "temperature": [mode.temperature for mode in modes],
            }
        ),
    ).load()

    data = environment.collect(10_000)
    train, test = TrainTestSplit(ratio=0.8).split(data)

    train = train.with_transform(
        SlidingWindow(window_size=10),
    )
    test = test.with_transform(
        SlidingWindow(window_size=10),
    )

    learner = RegressionTree(max_depth=5)

    inputs = [f"temperature_{i}" for i in range(10)] + [
        f"target_{i}" for i in range(9)
    ]
    outputs = ["temperature_9"]
    model = learn_offline(
        train,
        learner,
        inputs,
        outputs,
    )
    report = evaluate_offline(
        model,
        test,
        inputs,
        outputs,
        [MeanAbsoluteError(), MeanSquaredError()],
    )
    print(report)

if __name__ == "__main__":
    main()


#where to specify which dataset to use