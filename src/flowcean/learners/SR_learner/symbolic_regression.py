import sys

import argparse
from ruamel.yaml import YAML
from pathlib import Path
import time

import numpy as np

import polars as pl
import matplotlib.pyplot as plt
import sympy
import csv

from functools import partial
from pysr import PySRRegressor
from collections import deque
#import polars as pl
#import core.processed_data as processed_data
#import criteria.segmentation_criteria as segmentation_criteria

#from functools import partial
#from pysr import PySRRegressor
#import polars as pl
from statistics import mean

#from core.processed_data import GroupedData
#import criteria.grouping_criteria as grouping_criteria


########################################################Grouping_criteria.py##########################################################################
class grouping_criteria:
    def preserving_group_loss(prev_segments_loss, concatenation_loss, factor = 1):
        print("Grouping criterion", prev_segments_loss, concatenation_loss)
        return (
            concatenation_loss < factor * prev_segments_loss
        )
    
########################################################Segmentation_criteria.py##########################################################################
class segmentation_criteria:
    def average_increase(fitness_hist, saturation = 1e-10, factor = 1):
        fitness_list = [*fitness_hist]
        return fitness_list[-1] > saturation or factor * fitness_list[-1] >= np.mean(fitness_list[0:-2])

    def average_decrease(fitness_hist, saturation = 1e-10, factor = 1):
        fitness_list = [*fitness_hist]
        return fitness_list[-1] < saturation or factor * fitness_list[-1] <= np.mean(fitness_list[0:-2])

    def increase(fitness_hist, saturation = 1e-10, factor = 1):
        fitness_list = [*fitness_hist]
        return fitness_list[-1] > saturation or factor * fitness_list[-1] >= fitness_list[-2]

    def decrease(fitness_hist, saturation = 1e-10, factor = 1):
        fitness_list = [*fitness_hist]
        return fitness_list[-1] < saturation or factor * fitness_list[-1] <= fitness_list[-2]

########################################################Processed_data.py##########################################################################
class SegmentedData:
    def __init__(self, data: pl.DataFrame, segments: pl.DataFrame, switches, target_var):
        self.data = data
        self.segments = segments
        self.switches = switches
        self.target_var = target_var

    @classmethod
    def from_file(cls, data, target_var, path):
        segments = pl.read_csv(path)
        switches = segments["window_start"].to_list()
        return cls(data, segments, switches, target_var)

    def visualize(self):
        fig, ax = plt.subplots(1, 1)
        ax.plot(self.data[self.target_var])
        for x in self.switches:
            plt.axvline(x=x, color="red")
        plt.show()

    def write_segments_csv(self, path):
        self.segments.write_csv(path)

    def write_switches_csv(self, path):
        pl.DataFrame(self.switches).write_csv(path)

    def get_segmentation_deviation(self, file, length_penalty=100):
        with open(file, 'r') as file:
            reader = csv.reader(file)
            ground_truth_switches = [float(row[0]) for row in reader]

        deviation = 0
        deviation += length_penalty*abs(len(ground_truth_switches) - len(self.switches))
        matching = [None] * len(ground_truth_switches)
        for i in range(len(ground_truth_switches)):
            matching[i] = min(range(len(self.switches)), key=lambda x: abs(self.switches[x] - ground_truth_switches[i]))
            if matching[i] in matching[:i]:
                index = matching[:i].index(matching[i])
                new_dist = abs(self.switches[matching[i]] - ground_truth_switches[i])
                old_dist = abs(self.switches[matching[index]] - ground_truth_switches[index])
                if new_dist > old_dist:
                    matching[i] = None
                else:
                    matching[index] = None
                    

        print(matching)
        for i in range(len(matching)):
            if matching[i] is not None:
                deviation += abs(self.switches[matching[i]] - ground_truth_switches[i]) / len(self.switches)

        return deviation

    

class GroupedData:
    def __init__(self, data: pl.DataFrame, target_var, groups = []):
        self.data = data
        self.groups = groups
        self.target_var = target_var

    def add_group(self, group):
        group.group_id = len(self.groups)
        self.groups.append(group)

    def create_group(self, data, equation, window, loss, segment_losses):
        group = Group(data, equation, [window], len(self.groups), loss, segment_losses)
        self.groups.append(group)

    def print_groups(self):
        for group in self.groups:
            print("Group", group.group_id)
            print(group.windows)

    def visualize(self):
        cmap = plt.colormaps.get_cmap("hsv")
        fig, ax = plt.subplots(1, 1)
        ax.plot(self.data[self.target_var])
        for i, group in enumerate(self.groups):
            alpha = 0.8 - i / len(self.groups)
            for window in group.windows:
                plt.axvspan(
                    window[0], window[1], color=cmap(i / len(self.groups)), alpha=alpha
                )
        plt.show()

    def get_mean_loss(self):
        total_length = 0
        mean_loss = 0
        for group in self.groups:
            total_length += len(group.data)
            mean_loss += group.loss * len(group.data)
        return mean_loss / total_length
    
    def write_groups_csv(self, path):
        data = pl.DataFrame({
            "group_id": [group.group_id for group in self.groups],
            "loss": [group.loss for group in self.groups],
            "equation": [sympy.sstr(group.equation) for group in self.groups],
        })
        data.write_csv(path)

    def write_windows_csv(self, path):
        data = pl.DataFrame({
            "group_id": [group.group_id for group in self.groups for _ in group.windows],
            "window_start": [window[0] for group in self.groups for window in group.windows],
            "window_end": [window[1] for group in self.groups for window in group.windows],
        })
        data.write_csv(path)

    def to_json(self):
        return {
            "groups": [
                {
                    "group_id": group.group_id,
                    "loss": group.loss,
                    "equation": sympy.sstr(group.equation),
                    "windows": group.windows,
                }
                for group in self.groups
            ]
        }


class Group:
    def __init__(self, data: pl.DataFrame, equation, windows, group_id, loss, segment_losses):
        self.data = data
        self.equation = equation
        self.windows = windows
        self.group_id = group_id
        self.loss = loss
        self.segment_losses = segment_losses

    def append_segment(self, data, equation, window, loss, segment_loss):
        self.data = pl.concat([self.data, data])
        self.equation = equation
        self.windows.append(window)
        self.loss = loss
        self.segment_losses.append(segment_loss)



########################################################Segmentor.py##########################################################################
class Segmentor:
    """
    A class that segments a data frame into segments with differing dynamics using symbolic regression.

    Args:
        config (dict): A dictionary containing the configuration parameters for segmentation.

    Attributes:
        start_width (int): The starting width of the segmentation window.
        step_width (int): The step width for enlarging the segmentation window.
        step_iterations (int): The number of iterations for each symbolic regression run on the enlarged window.
        init_iterations (int): The number of iterations for symbolic regression on the initial window.
        hist_length (int): The history of the fitness over the enlarged windows.
        criterion (function): The fitness criterion function used for segmentation.
        selection (str): The name of the selection metric.
        learner (PySRRegressor): The symbolic regression learner.
        file_prefix (str): The prefix for the log files.
        target_var (str): The name of the target variable.

    """

    def __init__(self, config):
        self.start_width = config["start_width"]
        self.step_width = config["step_width"]
        self.step_iterations = config["step_iterations"]
        self.init_iterations = config["segmentation"]["kwargs"]["niterations"]
        self.hist_length = config["hist_length"]
        self.criterion = getattr(segmentation_criteria, config["segmentation"]["criterion"]["name"])
        if "kwargs" in config["segmentation"]["criterion"]:
            self.criterion = partial(self.criterion, **config["segmentation"]["criterion"]["kwargs"])
        if "selection" not in config:
            config["selection"] = "loss"
        self.selection = config["selection"]
        self.learner = PySRRegressor(**config["segmentation"].get("kwargs", {}))
        self.learner.feature_names = config["features"]
        self.file_prefix = config["file_prefix"]
        self.target_var = config["target_var"]

    def segment(self, data_frame):
        """
        Perform segmentation on the given data frame.

        Args:
            data_frame (pandas.DataFrame): The data frame to be segmented.

        Returns:
            segmented_results (segmented_data.SegmentedData): The segmented data

        """
        fitness_hist = deque([], self.hist_length)
        switches = [0]
        window = [0, self.start_width - self.step_width]
        segments = pl.DataFrame({
            "window_start": pl.Series(dtype=pl.Int64, values=[]),
            "window_end": pl.Series(dtype=pl.Int64, values=[]),
            "extensions": pl.Series(dtype=pl.Int64, values=[]),
            "equation": pl.Series(dtype=pl.Utf8, values=[]),
            self.selection: pl.Series(dtype=pl.Float64, values=[])
        })

        while window[1] < len(data_frame):
            self.learner.warm_start = False
            fitness_hist = deque([], self.hist_length)
            extension = 0
            while len(fitness_hist) < 2 or (
                self.criterion(fitness_hist) and window[1] < len(data_frame)
            ):
                self.learner.equation_file = (
                    "C:/Users/49157/Desktop/PA/2024---Harshitha-Viswanath---Project-Work/src/flowcean/equations/"
                    + self.file_prefix
                    + "_win"
                    + str(len(switches))
                    + "_ext"
                    + str(extension)
                    + ".csv"
                )
                if hasattr(self.learner, "equations_"):
                    best_equation = self.learner.sympy()
                window[1] += self.step_width
                window[1] = min(window[1], len(data_frame))

                print(window)
                current_frame = data_frame.slice(window[0], (window[1] - window[0]))

                X_train = current_frame[self.learner.feature_names]
                y_train = current_frame[self.target_var]
                self.learner.fit(X_train, y_train)
                fitness_hist.append(self.learner.get_best()[self.selection])
                self.learner.warm_start = True
                self.learner.niterations = self.step_iterations
                extension = extension + 1

            if(window[1] >= len(data_frame)):
                result_row = pl.DataFrame({
                    "window_start": [window[0]],
                    "window_end": [len(data_frame)],
                    "extensions": [extension],
                    "equation": [str(best_equation)],
                    self.selection: [self.learner.get_best()[self.selection]]
                })
                segments = segments.vstack(result_row)
                break

            window_end = window[1] - self.step_width
            result_row = pl.DataFrame({
                "window_start": [window[0]],
                "window_end": [window_end],
                "extensions": [extension - 1],
                "equation": [str(best_equation)],
                self.selection: [self.learner.get_best()[self.selection]]
            })
            segments = segments.vstack(result_row)
            switches.append(window_end)

            window[0] = window[1] - self.step_width
            window[1] = min(window[0] + self.start_width - self.step_width, len(data_frame))
            self.learner.niterations = self.init_iterations

        segmented_results = SegmentedData(data_frame, segments, switches, self.target_var)
        
        return segmented_results
    
########################################################group_identificator.py##########################################################################

class GroupIdentificator:
    def __init__(self, config):
        """
        Initializes a GroupIdentificator object.

        Args:
            config (dict): Configuration parameters for the GroupIdentificator.

        Attributes:
            criterion (function): The grouping criterion function.
            selection (str): The selection criteria for grouping.
            learner (PySRRegressor): The PySRRegressor object for learning equations.
            file_prefix (str): The file prefix for equation files.
        """
        self.criterion = getattr(
            grouping_criteria, config["grouping"]["criterion"]["name"]
        )
        if "kwargs" in config["grouping"]["criterion"]:
            self.criterion = partial(
                self.criterion, **config["grouping"]["criterion"]["kwargs"]
            )
        if "selection" not in config:
            config["selection"] = "loss"
        self.selection = config["selection"]
        self.learner = PySRRegressor(**config["grouping"].get("kwargs", {}))
        self.learner.warm_start = False
        self.learner.feature_names = config["features"]
        self.file_prefix = config["file_prefix"]

    def _set_learner_log_file(self, window, group):
        """
        Sets the equation file path for the learner.

        Args:
            window (list): The window range.
            group (int): The group ID.
        """
        self.learner.equation_file = (
            "./equations/"
            + self.file_prefix
            + "_win"
            + str(window[1])
            + "_group"
            + str(group)
            + ".csv"
        )

    def group_segments(self, segmented_results):
        """
        Groups the segments using symbolic regression and a grouping criterion.

        Args:
            segmented_results (SegmentedResults): The segmented results.

        Returns:
            GroupedData: The grouped data.
        """
        # todo: use previous models as starting point (option:
        # 1) try previous models for both. If one of them is better than the two before, a new one is found,
        # 2) test fit first, then re-learn?)
        # alternative procedure: test against all groups and choose the smallest one, if the error is below something or the increase in accuracy is large enough

        segments = segmented_results.segments
        data_frame = segmented_results.data
        target_var = segmented_results.target_var

        group_data = GroupedData(data_frame, target_var, groups=[])
        for segment in segments.iter_rows(named=True):
            window = [segment["window_start"], segment["window_end"]]
            curr_segment_loss = segment[self.selection]
            print("Current window:", window)

            df_window = data_frame.slice(window[0], (window[1] - window[0]))
            if not group_data.groups:
                X_train = df_window[self.learner.feature_names]
                y_train = df_window[target_var]
                self.learner.fit(X_train, y_train)

                group_data.create_group(
                    df_window,
                    self.learner.sympy(),
                    window,
                    curr_segment_loss,
                    [curr_segment_loss],
                )
                continue
            else:
                found_group = False
                for group in group_data.groups:
                    print(
                        "Current group", group.group_id, "of", len(group_data.groups)
                    )
                    self._set_learner_log_file(window, group.group_id)

                    concatenation = pl.concat([group.data, df_window])
                    X_train = concatenation[self.learner.feature_names]
                    y_train = concatenation[target_var]
                    self.learner.fit(X_train, y_train)
                    equation = self.learner.sympy()
                    loss = self.learner.get_best()[self.selection]
                    if self.criterion(
                        mean(group.segment_losses),  # todo: weighted by segment length?
                        loss,
                    ):
                        print("group", window, "into", group.group_id)
                        group.append_segment(
                            df_window, equation, window, loss, curr_segment_loss
                        )
                        found_group = True
                        break

                if not found_group:
                    print("Create new group")
                    X_train = df_window[self.learner.feature_names]
                    y_train = df_window[target_var]
                    self.learner.fit(X_train, y_train)

                    group_data.create_group(
                        df_window,
                        self.learner.sympy(),
                        window,
                        curr_segment_loss,
                        [curr_segment_loss],
                    )

        group_data.print_groups()
        return group_data

########################################################main.py##########################################################################
   

def main(path):

    config = YAML(typ="safe").load(path)
    data_frame = pl.read_csv(
        config["file"], schema_overrides=[pl.Float64] * len(config["features"])
    )
    if "derivative" in config and config["derivative"]:
        data_frame = data_frame.with_columns(diff=pl.col(config["target_var"]).diff())
        data_frame[0, "diff"] = data_frame["diff"][1]
        config["target_var"] = "diff"

    # Segmentation
    # Segmentation
    segmentor = Segmentor(config)
    starttime = time.time()
    segmented_data = segmentor.segment(data_frame)
    endtime = time.time()
    segmented_data.write_segments_csv("segmentation_results.csv")
    segmented_data.write_switches_csv("switches.csv")
    segmented_data.visualize()

    print("Time for segmentation:", endtime - starttime)
    with open("time.txt", "w") as file:
        file.write("Segmentation: " + str(endtime - starttime) + "\n")


    # Grouping
    group_identificator = GroupIdentificator(config)
    starttime = time.time()
    grouped_data = group_identificator.group_segments(segmented_data)
    endtime = time.time()
    grouped_data.write_groups_csv("grouping_results.csv")
    grouped_data.write_windows_csv("grouping_windows.csv")

    print("Time for grouping:", endtime - starttime)
    with open("time.txt", "a") as file:
        file.write("Grouping: " + str(endtime - starttime))
    grouped_data.visualize() 
    

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to config file",
    )
    arguments = parser.parse_args()

    main(arguments.config)

     
