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

class Acceleration(State):
    acceleration: float

    def __init__(self, acceleration: float) -> None:
        self.acceleration = acceleration

    #is it necessary to convert to numpy array
    @override
    def as_numpy(self) -> NDArray[np.float64]:
        return np.array([self.acceleration])
    
    @override
    @classmethod
    def from_numpy(cls, state: NDArray[np.float64]) -> Self:
        return cls(state[0])


global step_count 
type Target_Acceleration = int

class Idle(DifferentialMode[Acceleration, Target_Acceleration]):
    threshold_detect: float = 1.0
    #threshold_walk: float = 1.2

    @override
    def flow(
        self,
        t: float,
        state: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        _ = t
        return np.array([0.0])  #the parameter that governs the continuous state?

    @override
    def transition(
        self,
        i: Target_Acceleration,
    ) -> DifferentialMode[Acceleration, Target_Acceleration]:
        if self.state.acceleration > self.threshold_detect:   
            return Detection(t=0.0, state = self.state)
        return self

class Detection(DifferentialMode[Acceleration, Target_Acceleration]):
    threshold_detect: float = 1.0
    threshold_walk: float = 1.2

    @override
    def flow(
        self,
        t: float,
        state: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        _ = t
        return np.array([0.0])  #the parameter that governs the continuous state?

    @override
    def transition(
        self,
        i: Target_Acceleration,
    ) -> DifferentialMode[Acceleration, Target_Acceleration]:
        if self.state.acceleration >  self.threshold_walk:   
            return Count(t=0.0, state = self.state)
        if self.state.acceleration  < self.threshold_detect:   
            return Idle(t=0.0, state = self.state)
        return self
    

class Count(DifferentialMode[Acceleration, Target_Acceleration]):
    omega: float = 2 * np.pi #natural oscillation frequency of the step movement
    y = 20   #initial displacement
    v = 10   #initial velocity
    y_threshold = 0.5   #displacement threshold for vertical displacement to measure the step

    @override
    def flow(
        self,
        t: float,
        state: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        self.v = self.v - (self.omega**2) * self.y * self.t
        self.y = self.y + self.v * self.t
        return np.array([self.y]) #should return 0.0 or a non-zero value. Error thrown if left empty

    @override
    def transition(
        self,
        t: float,
    ) -> DifferentialMode[Acceleration, Target_Acceleration]:
        if abs(self.y) > self.y_threshold:
            step_count += 1
            self.y = 20
        if self.state.acceleration >  self.threshold_detect and self.state.acceleration < self.threshold_walk :   
            return Detection(t=0.0, state = self.state)
        if self.state.acceleration  < self.threshold_detect:   
            return Idle(t=0.0, state = self.state)
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

    target_accelerations = (
        (0.1 * i, acc)
        for i, acc in enumerate(
            randomly_changing_values(
                change_probability=0.002,
                minimum=0.5,
                maximum=2.0,
            )
        )
    )

    environment = HybridSystem(
        initial_mode=Idle(t=0.0, state=Acceleration(0.7)),
        inputs = target_accelerations,
        map_to_dataframe=lambda times, inputs, modes: pl.DataFrame(
            {
                "time": times,
                "target": inputs,
                "acceleration": [mode.acceleration for mode in modes],
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

    inputs = [f"acceleration_{i}" for i in range(10)] + [
        f"target_{i}" for i in range(9)
    ]
    outputs = ["acceleration_9"]
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


