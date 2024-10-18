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


u1:float = 0.5
Ts:float = 0.001

L = 1**-3
C = 1**-6
RL = 0.1
R = 10
RC = 0.01
E = 10

A1 = np.array([-RL/L,0],[0, -1/(R+RC) * C])
B1 = np.array([1/L] , [0]) * E

A2 = np.array([-1/L * (RL + RC * R/(R+RC)), 1/L * (-1 + R/(R+RC))], [R/(R+RC)*C, -1(R+RC)*C])
B2 = np.array([1/L], [0]) * E

A3 = np.array([0 , 0], [0, -1/(R+RC)*C])
B3 = np.array([0], [0])

class Time(State):
    time: float
    inductor_current: float  

    def __init__(self, time: float, inductor_current: float) -> None:
        self.time = time
        self.inductor_current = inductor_current

    @override
    def as_numpy(self) -> NDArray[np.float64]:
        return np.array([self.time])

    @override
    @classmethod
    def from_numpy(cls, state: NDArray[np.float64]) -> Self:
        return cls(state[0])

class SwitchClosed(DifferentialMode[Time]):
    switching_timeout: float = (u1)*Ts

    @override
    def flow(
        self,
        t: float,
        state: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        _ = t
        return np.array([])  #continuous param - change in the inductor current
    
    @override
    def transition(
        self,
    ) -> DifferentialMode[Time]:
        if self.state.time >= self.switching_timeout:
            return SwitchOpen(t=0.0, state=self.state) #should self.time be 0?
        return self
    

class SwitchOpen(DifferentialMode[Time]):
    switching_timeout: float = Ts
    

    @override
    def flow(
        self,
        t: float,
        state: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        _ = t
        return np.array([])  #continuous param
    
    @override
    def transition(
        self,
    ) -> DifferentialMode[Time]:
        if self.state.time >= self.switching_timeout:   #self.state.time > i
            return SwitchClosed(t=0.0, state=self.state)
        if self.inductor_current == 0:
            return SwitchOpen_iL0(t=0.0, state=self.state)
        return self
    

class SwitchOpen_iL0(DifferentialMode[Time]):
    switching_timeout: float = Ts
    
    @override
    def flow(
        self,
        t: float,
        state: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        _ = t
        return np.array([])  #continuous param
    
    @override
    def transition(
        self,
    ) -> DifferentialMode[Time]:
        if self.state.time >= self.switching_timeout:   #self.state.time > i
            return SwitchClosed(t=0.0, state=self.state)
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

    target_timeframes = ((0.1 * i, time)
        for i, time in enumerate(
            randomly_changing_values(
                change_probability=0.002,
                minimum=30.0,
                maximum=60.0,
            )
        )
    ) # simulation time span : 0:1e-6:1e-3 in matlab

    environment = HybridSystem(
        initial_mode=SwitchClosed(t=0.0, state=Time(1/2000)),  #half a millisecond
        inputs=target_timeframes,
        map_to_dataframe=lambda times, inputs, modes: pl.DataFrame(
            {
                "time": times,
                "target": inputs,
                "mode_time": [mode.time for mode in modes],
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

    inputs = [f"time_{i}" for i in range(10)] + [
        f"target_{i}" for i in range(9)
    ]
    outputs = ["time_9"]
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

#where to specify the initial parameters
#include two parameters that govern the transition between the states : il and time
# how to keep checking if iL drops to 0
#where to fix the simulation time period

    

    



