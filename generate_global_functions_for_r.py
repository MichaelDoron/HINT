#!/usr/bin/python
import scoop
from scoop import futures
import numpy as np
from scipy.stats import ortho_group
import copy
import dill as pickle
import itertools
import time
import marshal
import types

import sys
import os

func = None
true_pairs = []
number_of_variables = {'easy' : 10, 'hard' : 100}
number_of_functions = {'easy' : 25, 'hard' : 1000} 

def generate_gaussian(z):
  if (len(z) == 1):
    U = np.array(np.random.normal(0, 1))
    D = np.diag([np.power((np.random.rand() * (2 - 0.1)) + 0.1, 2) for ind in range(len(z))])
  else:
    U = ortho_group.rvs(dim=len(z))
    D = np.diag([np.power((np.random.rand() * (2 - 0.1)) + 0.1, 2)  for ind in range(len(z))])
  def g(x): 
    input_preparation = x[:, z]
    func = lambda k : np.exp(-0.5 * np.dot(np.dot(input_preparation[k,:], np.dot(np.dot(U.T, D), U)), input_preparation[k,:].T))
    result = np.array(map(func, range(x.shape[0])))
    return np.array(result).flatten()
  g.__name__ = str(np.random.rand())
  return g

def generate_super_function(number_of_samples, iteration, task):
  global true_pairs
  true_pairs = []
  coeffs = (np.random.rand(number_of_functions[task]) * 2) - 1
  variable_subset_sizes = np.min(np.array([np.array(1.5 + np.random.exponential(scale = 1.0, size=number_of_functions[task])).astype(int), [number_of_variables[task]] * number_of_functions[task]]), axis=0)
  variables = np.arange(number_of_variables[task])
  Z = [np.random.choice(variables, variable_subset_sizes[ind], replace=False) for ind in range(number_of_functions[task])]
  [true_pairs.extend(list(itertools.permutations(z, 2))) for z in Z]
  functions = [generate_gaussian(z) for z in Z]
  def super_function(inp):
    result = 0
    for function in functions:
      result += function(inp)
    return result
  super_function.__name__ = str(np.random.rand())
  return super_function

def set_func(iteration, number_of_samples, task):
  global func
  func = generate_super_function(number_of_samples, iteration, task)

def get_data(data):
  return func(np.array(data))

def get_true_pairs(iteration, number_of_samples, task):
  return np.array(true_pairs)


if __name__ == '__main__':
	func = None