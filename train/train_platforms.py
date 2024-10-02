"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.
This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

import os

class TrainPlatform:
    """
    Represents a platform for training models with capabilities to report metrics and arguments, and to handle resource cleanup.

    Methods:
        __init__(self, save_dir):
            Initialize the TrainPlatform with a directory to save training outputs.

        report_scalar(self, name, value, iteration, group_name=None):
            Report a scalar metric to the training platform.

        report_args(self, args, name):
            Report the training arguments to the training platform.

        close(self):
            Perform any necessary cleanup and resource deallocation.
    """
    def __init__(self, save_dir):
        pass

    def report_scalar(self, name, value, iteration, group_name=None):
        pass

    def report_args(self, args, name):
        pass

    def close(self):
        pass


class ClearmlPlatform(TrainPlatform):
    """
         class ClearmlPlatform(TrainPlatform):

         A class that facilitates interaction with the ClearML platform for logging and task management.


         def __init__(self, save_dir):

         Initializes the ClearmlPlatform instance by creating a ClearML task.

         Parameters:
         save_dir (str): Directory path where outputs will be saved.


         def report_scalar(self, name, value, iteration, group_name):

         Reports scalar values such as metrics or losses to the ClearML logger.

         Parameters:
         name (str): The name of the scalar value being reported.
         value (float): The scalar value.
         iteration (int): The current iteration or step number.
         group_name (str): The group or category name under which the scalar will be logged.


         def report_args(self, args, name):

         Connects and reports arguments or hyperparameters to the ClearML task.

         Parameters:
         args (object): The arguments or hyperparameters to be connected.
         name (str): The name to associate with the reported arguments.


         def close(self):

         Closes the ClearML task and performs any cleanup actions needed.
    """
    def __init__(self, save_dir):
        from clearml import Task
        path, name = os.path.split(save_dir)
        self.task = Task.init(project_name='motion_diffusion',
                              task_name=name,
                              output_uri=path)
        self.logger = self.task.get_logger()

    def report_scalar(self, name, value, iteration, group_name):
        self.logger.report_scalar(title=group_name, series=name, iteration=iteration, value=value)

    def report_args(self, args, name):
        self.task.connect(args, name=name)

    def close(self):
        self.task.close()


class TensorboardPlatform(TrainPlatform):
    """
        TensorboardPlatform class for logging training metrics with TensorBoard

        __init__(save_dir)
            Initializes the TensorboardPlatform with a directory to save log files.
            Parameters:
                save_dir (str): Directory where logs will be saved.

        report_scalar(name, value, iteration, group_name=None)
            Logs a scalar value to TensorBoard.
            Parameters:
                name (str): The name of the scalar to log.
                value (float): The value of the scalar.
                iteration (int): The current training iteration.
                group_name (str, optional): The group name under which the scalar is categorized.

        close()
            Closes the TensorBoard SummaryWriter.
    """
    def __init__(self, save_dir):
        from torch.utils.tensorboard import SummaryWriter
        self.writer = SummaryWriter(log_dir=save_dir)

    def report_scalar(self, name, value, iteration, group_name=None):
        self.writer.add_scalar(f'{group_name}/{name}', value, iteration)

    def close(self):
        self.writer.close()


class NoPlatform(TrainPlatform):
    """

        NoPlatform is a subclass of TrainPlatform that serves as a placeholder with no implemented functionality.

        Methods:
            __init__(save_dir): Initializes the NoPlatform instance.

        Parameters:
            save_dir (str): Directory path for saving data.
    """
    def __init__(self, save_dir):
        pass


