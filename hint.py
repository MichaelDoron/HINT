#!/usr/bin/python

import argparse
import sys
from scoop import futures
import pandas as pd
import numpy as np
import utils
import time

from importlib import import_module

parser = argparse.ArgumentParser(description='HINT')
parser.add_argument('--model_path', '-m', type=str, metavar='model path', help='The path to the model python wrapper')
parser.add_argument('--output_path', '-o', type=str, metavar='output path', help='The path to the output directory')
parser.add_argument('--number_of_core_samples', '-n', type=int, metavar='core sample number', default=1000, help='The number of randomly sampled feature vectors, before perturbations')
parser.add_argument('--step_size', '-dx', type=float, metavar='step size', default=0.01, help='The step size used for the finite differences calculation, written as a percentage of the feature range')
parser.add_argument('--threshold', '-t', type=float, metavar='threshold', default=3, help='The threshold for the local interaction feature pair ranking, in units of standard deviations of the Hessian matrix')
parser.add_argument('--name', '-i', type=str, metavar='unique_name', default='0', help='Saves the run under a specific name')
parser.add_argument('--create_data', action='store_true', help='Whether to sample new data from the model, in the structure of N core samples and (m^2 + m) perturbations around each core sample')
parser.add_argument('--create_unstructured_data', action='store_true', help='Whether to sample new data from the model, in the structure of N * (m^2 + m + 1) samples uniformly sampled from the feature space')
parser.add_argument('--rank_global', action='store_true', help='Whether to rank the global pairwise interactions')
parser.add_argument('--rank_local', action='store_true', help='Whether to rank the local pairwise interactions')
parser.add_argument('--measure_global_accuracy', action='store_true', help='Whether to measure the global ranking')
parser.add_argument('--measure_local_accuracy', action='store_true', help='Whether to measure the local sample accuracy')
parser.add_argument('--plot', action='store_true', help='Whether to plot the global / local interactions')

def main():
  start = time.time()
  args = parser.parse_args()
  model = import_module(args.model_path)
  if (args.create_data):
    utils.create_data(model, args.number_of_core_samples, args.step_size, args.name, args.output_path)
  if (args.create_unstructured_data):
    model.create_unstructured_data(model, args.number_of_core_samples, args.name, args.output_path)
  if (args.rank_global):
    utils.rank_global(model, args.number_of_core_samples, args.step_size, args.name, args.plot, args.output_path)
  if (args.rank_local):
    print(utils.rank_local(model, args.number_of_core_samples, args.step_size, args.name, args.threshold, args.plot, args.output_path))
  if (args.measure_global_accuracy):
    print(utils.measure_global_accuracy(model, args.number_of_core_samples, args.step_size, args.name, args.output_path))
  if (args.measure_local_accuracy):
    print(utils.measure_local_accuracy(model, args.number_of_core_samples, args.step_size, args.name, args.output_path))
  print('this took {} seconds'.format(time.time() - start))

if __name__== "__main__":
  main()
