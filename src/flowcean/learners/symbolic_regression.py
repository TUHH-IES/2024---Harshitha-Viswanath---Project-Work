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
from statistics import mean

import logging
from typing import Any, override, List
import os 

from flowcean.core import Model, SupervisedLearner
from flowcean.models.srmodel import SymbolicRegressionModel

logger = logging.getLogger(__name__)

########################################################################################segmentation_criteria###############################################
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

class grouping_criteria:
    def preserving_group_loss(prev_segments_loss, concatenation_loss, factor = 1):
        print("Grouping criterion", prev_segments_loss, concatenation_loss)
        return (
            concatenation_loss < factor * prev_segments_loss
        )

########################################################################################processed_data###############################################

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
    
    def write_groups_csv(self):
        data = pl.DataFrame({
            "group_id": [group.group_id for group in self.groups],
            "loss": [group.loss for group in self.groups],
            "equation": [sympy.sstr(group.equation) for group in self.groups],
        })
        return data #directly return the dataframe instead of a csv file
        #data.write_csv(path)

    def write_windows_csv(self):
        data = pl.DataFrame({
            "group_id": [group.group_id for group in self.groups for _ in group.windows],
            "window_start": [window[0] for group in self.groups for window in group.windows],
            "window_end": [window[1] for group in self.groups for window in group.windows],
        })
        return data #return dataframe instead of dumping into csv file
        #data.write_csv(path)

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


########################################################################################segmentation#########################################################
class Segmentor:
    def __init__(
            self, 
            start_width : int,
            step_width : int,
            features : List[str],
            file_prefix : str,
            target_var : str,
            **kwargs : Any, #args given to pysr
            ) -> None:
        self.start_width = start_width
        self.step_width = step_width
        self.features = features
        self.file_prefix = file_prefix
        self.target_var = target_var

        #pysr arguments
        """ pysr_args = {
                "niterations": 40, 
                "verbosity" : 0, 
                "random_state" : 42, 
                "deterministic" : True, 
                "procs" : 0, 
                "multithreading" : False, 
                "parsimony" : 0.0032, 
                "binary_operators" : ["+", "-", "*", "/"], 
                "unary_operators" : None
        }
 """
        #keyword arguments with default values
        self.step_iterations = 5
        self.hist_length = 5
        self.criterion = getattr(segmentation_criteria, "decrease")
        self.criterion = partial(self.criterion, {"saturation": 1e-6}) 
        self.selection = "loss" #check with loss?
        self.learner = PySRRegressor(**kwargs) #if any specific args are sent else default params are used
        self.learner.feature_names = self.features

    def segment(self, data_frame):
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
                    "C:/Users/49157/Desktop/PA/2024---Harshitha-Viswanath---Project-Work/src/flowcean/learners/equations/"
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

class GroupIdentificator:

    def __init__(
            self,
            features : List[str],
            file_prefix : str,
            **kwargs : Any,) -> None:
        
        self.features = features
        self.file_prefix = file_prefix

        #pysr args
        """ pysr_args = {
            "niterations": 40,
            "verbosity": 0,
            "random_state": 42,
            "deterministic": True,
            "procs": 0,
            "multithreading": False,
            "parsimony": 0.0032,
            "binary_operators": ["+", "-", "*", "/"],
            "unary_operators": None,
            "population": 40 
        }
 """
        #keyword arguments with default values
        self.criterion = getattr(grouping_criteria, "preserving_group_loss")
        self.criterion = partial(self.criterion, {"factor":1})
        self.selection = "loss" #check with loss?
        self.learner = PySRRegressor(**kwargs) #if any specific args are sent else default params are used
        self.learner.warm_start = False
        self.learner.feature_names = self.features


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



########################################################################################SR_learner_code###############################################

class SymbolicRegression(SupervisedLearner):

    def __init__(
            self,
            csv_file_path : str, #dataframe instead of path
            features : List[str],
            start_width : int,
            step_width : int,
            target_var : str,  #should this be a list??
            derivative : bool,
    ) -> None: 
        self.csv_file_path = csv_file_path
        self.features = features
        self.start_width = start_width
        self.step_width = step_width
        self.target_var = target_var
        self.derivative = derivative
        
        self.file_prefix = os.path.splitext(os.path.basename(self.csv_file_path))[0]   


        self.data_frame = pl.read_csv(self.csv_file_path, schema_overrides=[pl.Float64] * len(self.features))     
        if derivative is not None  and self.derivative:
            self.data_frame = self.data_frame.with_columns(diff=pl.col(self.target_var).diff())
            self.data_frame[0, "diff"] = self.data_frame["diff"][1]
            self.target_var = "diff"

    @override
    def learn(self, 
              **kwargs):
        segmentor = Segmentor(start_width=self.start_width, step_width=self.step_width, features= self.features, file_prefix=self.file_prefix, target_var=self.target_var, **kwargs)
        starttime = time.time()
        segmented_data = segmentor.segment(self.data_frame) #segmented_data is an object of class SegmentedData
        endtime = time.time()
        segmented_data.write_segments_csv("segmentation_results.csv")
        segmented_data.write_switches_csv("switches.csv")
        segmented_data.visualize()

        print("Time for segmentation:", endtime - starttime)
        with open("time.txt", "w") as file:
            file.write("Segmentation: " + str(endtime - starttime) + "\n")


        group_identificator = GroupIdentificator(features=self.features, file_prefix=self.file_prefix, **kwargs)
        starttime = time.time()
        grouped_data = group_identificator.group_segments(segmented_data)
        endtime = time.time()
        grouping_results = grouped_data.write_groups_csv()
        grouping_windows = grouped_data.write_windows_csv()

        print(grouping_windows)
        print(grouping_results)

        print("Time for grouping:", endtime - starttime)
        with open("time.txt", "a") as file:
            file.write("Grouping: " + str(endtime - starttime))
        grouped_data.visualize() 

        #return SymbolicRegressionModel([grouping_windows, grouping_results])
        
def main():

    model = SymbolicRegression("C:/Users/49157/Desktop/PA/SR_Original_code/SymbolicRegression4HA/data/converter/short_wto_zeros_data_converter_omega400e3_beta40e3_Q10_theta60.csv", ["t","w1","w2"], 100, 20, "w2", True)
    model.learn()

if __name__ == "__main__":
    main()
