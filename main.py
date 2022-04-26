# author: Laurens Devos
# Copyright BDAP team, DO NOT REDISTRIBUTE

###############################################################################
#                                                                             #
# THIS FILE IS NOT GRADED, MAKE SURE YOUR CODE WORKS WITH THE ORIGINAL FILE!  #
#                                                                             #
###############################################################################

import joblib, sys, os
import numpy as np
import argparse
import pprint

import util
from nn import *

def parse_arguments():
    tasks = ["time_and_accuracy", "distance_error", "plot",
            "retrieval", "hyperparam"]
    parser = argparse.ArgumentParser(
            description="Execute the code and experiments for BDAP "
                        "assignment 3.")
    parser.add_argument("dataset", choices=list(util.DATASETS.keys()),
            default="spambase", nargs="?")
    parser.add_argument("--task", choices=tasks, default="time_and_accuracy")
    parser.add_argument("-k", type=int, default=1,
            help="number of neighbors")
    parser.add_argument("-n", type=int, default=10,
            help="size of test set sample")
    parser.add_argument("--seed", type=int, default=1,
            help="seed for random sample selection")
    args = parser.parse_args()

    return args

if __name__ == "__main__":
    args = parse_arguments()
    dataset, task = args.dataset, args.task
    k, n, seed = args.k, args.n, args.seed

    if task == "time_and_accuracy":
        print("ACCURACY TASK")
        res = functions.time_and_accuracy_task(dataset, k, n, seed)
        pprint.pprint(res)

    elif task == "distance_error":
        print("DISTANCE_ABSOLUTE_ERROR TASK")
        res = functions.distance_absolute_error_task(dataset, k, n, seed)
        print(res)

    elif task == "retrieval":
        print("RETRIEVAL TASK")
        res = functions.retrieval_task(dataset, k, n, seed)
        print(res)

    elif task == "hyperparam":
        print("HYPERPARAM TASK")
        if n != 1000:
            n = 1000
            print("Using n=1000")
        if k != 10:
            k = 10
            print("Using k=10")
        functions.hyperparam_task(dataset, k, n, seed)

    elif task == "plot":
        print("PLOT TASK")
        functions.plot_task(dataset, k, n, seed)

    else:
        print(f"Invalid task '{task}'")
        args.print_help()



