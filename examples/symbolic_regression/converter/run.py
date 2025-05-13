import flowcean.cli
from flowcean.core import evaluate_offline, learn_offline
from flowcean.polars import DataFrame
from flowcean.sklearn import MeanAbsoluteError, MeanSquaredError
from flowcean.polars.transforms.derivative import Derivative

from flowcean.srlearn.symbolic_regression import SymbolicRegression

def main() -> None:
    flowcean.cli.initialize_logging()

    data = DataFrame.from_uri(uri="file:C:/Users/49157/Desktop/PA/SR_Original_code/SymbolicRegression4HA/data/converter/short_wto_zeros_data_converter_omega400e3_beta40e3_Q10_theta60.csv")

    inputs = ["t", "w1", "w2"]
    outputs = ["w2"]

    derivative = Derivative(outputs)

    learner = SymbolicRegression(
        #csv_file_path = "C:/Users/49157/Desktop/PA/SR_Original_code/SymbolicRegression4HA/data/converter/short_wto_zeros_data_converter_omega400e3_beta40e3_Q10_theta60.csv",  
        features = inputs,
        start_width = 100,
        step_width = 20, 
        target_var = outputs,
        derivative = True,
    )
    
    model = learn_offline(
        data,
        learner,
        inputs,
        outputs,
        output_transform = derivative,
    ) 

    """ report = evaluate_offline(
        model,
        data,
        inputs,
        outputs,
        [],
    ) """
    
    #print(report) 


if __name__ == "__main__":
    main() 