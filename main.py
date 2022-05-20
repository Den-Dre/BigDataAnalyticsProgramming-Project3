# author: Laurens Devos
# Copyright BDAP team, DO NOT REDISTRIBUTE

###############################################################################
#                                                                             #
# THIS FILE IS NOT GRADED, MAKE SURE YOUR CODE WORKS WITH THE ORIGINAL FILE!  #
#                                                                             #
###############################################################################
import linecache
import tracemalloc
from datetime import datetime
from queue import Empty, Queue
from resource import RUSAGE_SELF, getrusage
from threading import Thread

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

def memory_monitor(command_queue: Queue, poll_interval=1):
    tracemalloc.start()
    old_max = 0
    snapshot = None
    while True:
        try:
            command_queue.get(timeout=poll_interval)
            if snapshot is not None:
                print(datetime.now())
                display_top(snapshot)

            return
        except Empty:
            max_rss = getrusage(RUSAGE_SELF).ru_maxrss
            if max_rss > old_max:
                old_max = max_rss
                snapshot = tracemalloc.take_snapshot()
                print(datetime.now(), 'max RSS', max_rss)


def display_top(snapshot, key_type='lineno', limit=3):
    snapshot = snapshot.filter_traces((
        tracemalloc.Filter(False, "<frozen importlib._bootstrap>"),
        tracemalloc.Filter(False, "<unknown>"),
    ))
    top_stats = snapshot.statistics(key_type)

    print("Top %s lines" % limit)
    for index, stat in enumerate(top_stats[:limit], 1):
        frame = stat.traceback[0]
        # replace "/path/to/module/file.py" with "module/file.py"
        filename = os.sep.join(frame.filename.split(os.sep)[-2:])
        print("#%s: %s:%s: %.1f KiB"
              % (index, filename, frame.lineno, stat.size / 1024))
        line = linecache.getline(frame.filename, frame.lineno).strip()
        if line:
            print('    %s' % line)

    other = top_stats[limit:]
    if other:
        size = sum(stat.size for stat in other)
        print("%s other: %.1f KiB" % (len(other), size / 1024))
    total = sum(stat.size for stat in top_stats)
    print("Total allocated size: %.1f KiB" % (total / 1024))

if __name__ == "__main__":
    args = parse_arguments()
    dataset, task = args.dataset, args.task
    k, n, seed = args.k, args.n, args.seed

    queue = Queue()
    poll_interval = 0.1
    monitor_thread = Thread(target=memory_monitor, args=(queue, poll_interval))
    monitor_thread.start()

    try:
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
    finally:
        queue.put('stop')
        monitor_thread.join()


