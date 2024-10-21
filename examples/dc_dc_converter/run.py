import random
from collections.abc import Iterator
from typing import Self, override

import numpy as np
import polars as pl
from numpy.typing import NDArray

from flowcean.cli.logging import initialize_logging
from flowcean.environments.hybrid_system import(
    DifferentialMode,
    HybridSystem,
    State,
)
from flowcean.environments.train_test_split import TrainTestSplit
from flowcean.learners.regression_tree import RegressionTree
from flowcean.metrics.regression import MeanAbsoluteError, MeanSquaredError
from flowcean.strategies.offline import evaluate_offline, learn_offline
from flowcean.transforms.sliding_window import SlidingWindow

type sim_time = float
E: int = 20
L: float = 1e-3
RL: float = 0.1
C: float = 10e-6
RC: float = 0.06
R: int = 10


class ContinuousDynamics(State):
    inductor_current: float
    #cap_voltage: float

    def __init__(self, inductor_current: float) -> None:
        self.inductor_current = inductor_current
        #self.cap_voltage = cap_voltage

    @override
    def as_numpy(self) -> NDArray[np.float64]:
        return np.array([self.inductor_current])
    
    @override
    @classmethod
    def from_numpy(cls, state: NDArray[np.float64]) -> Self:
        return cls(state[0])

class Switch_Closed(DifferentialMode[ContinuousDynamics, sim_time]):
    Ts: float = 0.1e-3
    u: float = 0.5


    @override
    def flow(
        self,
        t: float,
        state: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        iL = state[0]
        #v = state[1]

        diL_dt = (-RL*iL + E) / L
        #dv_dt = -v / (R + RC) * C

        del_iL = diL_dt * self.Ts  # Ts is taken as the time step
        #del_v = dv_dt * self.Ts

        return np.array([diL_dt])
    
    def transition(
            self,
            i: sim_time,
    ) -> DifferentialMode[ContinuousDynamics, sim_time]:
        if i >= (self.u * self.Ts):
            return Switch_Open(t=0.0, state=self.state)  # Switching open
        return self


class Switch_Open(DifferentialMode[ContinuousDynamics, sim_time]):
    Ts: float = 0.1e-3

    @override
    def flow(
        self,
        t: float,
        state: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        iL = state[0]
        #v = state[1]

        diL_dt = 1/L * ((R*(1-RC)/(R+RC)) - (1+RL))
        #dv_dt = (R*iL - v) / ((R + RC) * v)

        return np.array([diL_dt])
    
    def transition(
            self,
            i: sim_time,
    ) -> DifferentialMode[ContinuousDynamics, sim_time]:
        if i > self.Ts:
            return Switch_Closed(t=0.0, state=self.state)
        if self.state[0] == 0:
            return Switch_Open_iL0(t=0.0, state=self.state)
        return self


class Switch_Open_iL0(DifferentialMode[ContinuousDynamics, sim_time]):
    Ts: float = 0.1e-3

    @override
    def flow(
        self,
        t: float,
        state: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        #v = state[1]

        #dv_dt = -v / (R + RC) * C

        return np.array([0])
    
    @override
    def transition(
        self,
        i: sim_time,
    ) -> DifferentialMode[ContinuousDynamics, sim_time]:
        if i > self.Ts:
            return Switch_Closed(t=0.0, state=self.state)
        return self
    

# Function to generate randomly changing values for iL and v
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

# Generator for inductor current and capacitor voltage
def target_iL_v(iL_min: float, iL_max: float):
    iL_generator = randomly_changing_values(
        change_probability=0.002,
        minimum=iL_min,
        maximum=iL_max,
    )

    #v_generator = randomly_changing_values(
     #   change_probability=0.002,
     #   minimum=v_min,
    #    maximum=v_max,
    #)

    #return (
     #   (il, v)
     #   for il, v in enumerate(zip(iL_generator, v_generator))
  #  )

# Main function to train and test the model
def main() -> None:
    initialize_logging()

    target_il_v_generator =(
        (0.1 * i, current)
        for i, current in enumerate(
            randomly_changing_values(
                change_probability=0.002,
                minimum=0.0,
                maximum=4.0,
            )
        )
    )

    for index, (scaled_index, current_value) in enumerate(target_il_v_generator):
        print(scaled_index, current_value)
    
    # Break after printing 10 tuples to avoid infinite loop
        if index >= 9:
            break

    environment = HybridSystem(
        initial_mode=Switch_Closed(t=0.0, state=ContinuousDynamics(2.0)),
        inputs=target_il_v_generator,
        map_to_dataframe=lambda times, inputs, modes: pl.DataFrame(
            {
                "time": times,
                "iL": inputs,  # Unpack two values instead of three
                "current": [mode.inductor_current for mode in modes],
            }
        ),
    ).load()


    #print(map_to_dataframe)

    data = environment.collect(10_000)

    print(data.get_data())


    train, test = TrainTestSplit(ratio=0.8).split(data)

    #print("Training Data Columns:", train.columns)

    train = train.with_transform(
        SlidingWindow(window_size=10),
    )

    test = test.with_transform(
        SlidingWindow(window_size=10),
    )

    learner = RegressionTree(max_depth=5)

    inputs = [f"current_{i}" for i in range(10)] + [
        f"iL_{i}" for i in range(9)
    ]
    outputs = ["current_9"]
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
    