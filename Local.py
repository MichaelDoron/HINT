#!/usr/bin/python

import dill as pickle
import itertools
import pandas as pd
import numpy as np
from scoop import futures
from scipy.stats import ortho_group
import time

number_of_variables = 10
number_of_phenomena = 3
sort_key = lambda x : int(x[1:])
feature_names = sorted(['X{}'.format(ind) for ind in range(1, number_of_variables + 1)], key=sort_key)

m = len(feature_names)

feature_pairs = sorted([sorted(pair) for pair in itertools.combinations(range(len(feature_names)), 2)])
feature_pairs = ['{} and {}'.format(feature_names[p[0]], feature_names[p[1]]) for p in feature_pairs]

normalization_feature_pairs = []
for feature_ind_1 in range(len(feature_names)):
  for feature_ind_2 in range(feature_ind_1 + 1, len(feature_names)):
    normalization_feature_pairs.append('{} and {}'.format(feature_names[feature_ind_1],feature_names[feature_ind_2]))

perturbation_feature_pairs = []
for feature_ind_1 in range(len(feature_names)):
  for feature_ind_2 in range(feature_ind_1 + 1, len(feature_names)):
    perturbation_feature_pairs.append('{} and {}'.format(feature_names[feature_ind_1],feature_names[feature_ind_2]))

perturbation_status_columns = []
perturbation_status_columns.append('core')
for feature_ind_1 in range(len(feature_names)):
  perturbation_status_columns.append(feature_names[feature_ind_1])
for feature_ind_1 in range(len(feature_names)):
  for feature_ind_2 in range(feature_ind_1 + 1, len(feature_names)):
    perturbation_status_columns.append('{} and {}'.format(feature_names[feature_ind_1],feature_names[feature_ind_2]))


feature_limits = {
  'X{}'.format(ind) : (0, 1) for ind in range(1, number_of_variables + 1)
  }

output_names = ['Result']

def generate_local_function(local_dimensions, local_centers, local_magnitudes):
  def f(x): 
    input_preparation = x[:, local_dimensions]
    func = lambda k : np.power(local_magnitudes[k] * (input_preparation[:, k] - local_centers[k]), 2)
    result = np.exp(-np.sum(np.array(map(func, range(input_preparation.shape[1]))), axis=0))
    return np.array(result)
  f.__name__ = str(np.random.rand())
  return f

def set_ground_truth(number_of_core_samples, step_size, name, output_path):
  found = 0
  while found < 1:
    variables = np.arange(number_of_variables)
    variable_subset_sizes = np.min(np.array([np.array(2.0 + np.random.exponential(scale = 1.0, size=number_of_phenomena)).astype(int), [number_of_variables] * number_of_phenomena]), axis=0)

    dimensions = [np.random.choice(variables, variable_subset_sizes[ind], replace=False) for ind in range(number_of_phenomena)]
    centers = [np.random.rand(variable_subset_sizes[ind]) for ind in range(number_of_phenomena)]
    magnitudes = [10 + (10 * np.random.rand(variable_subset_sizes[ind])) for ind in range(number_of_phenomena)]

    old_true_pairs = []
    old_true_pairs = [[sorted(a) for a in (sorted(list(itertools.permutations(z, 2))))] for z in dimensions]
    true_pairs = [np.unique(a, axis=0) if len(a) > 0 else a for a in old_true_pairs]
    if np.sum(np.sum(np.array(old_true_pairs).flatten())) == 0:
      continue
    else:
      found = 1
    
    pickle.dump(centers, open('{}/centers_{}_{}_{}.pickle'.format(output_path,number_of_core_samples, step_size, name),'wb'))
    pickle.dump(magnitudes, open('{}/magnitudes_{}_{}_{}.pickle'.format(output_path,number_of_core_samples, step_size, name),'wb'))
    pickle.dump(dimensions, open('{}/dimensions_{}_{}_{}.pickle'.format(output_path,number_of_core_samples, step_size, name),'wb'))

    functions = [generate_local_function(dimensions[dim_ind], centers[dim_ind], magnitudes[dim_ind]) for dim_ind in range(len(dimensions))]

    def super_function(inp):
      result = 0
      for function in functions:
        result += function(inp)
      for var in range(number_of_variables):
        result += np.random.randint(low=-5,high=5) * np.power(inp[:, var], np.random.randint(low=1,high=3))
      return result
    
    super_function.__name__ = str(np.random.rand())
  pickle.dump(true_pairs, open('{}/true_pairs_{}_{}_{}.cPickle'.format(output_path,number_of_core_samples, step_size, name),'wb'))
  pickle.dump(super_function, open('{}/model_{}_{}_{}.cPickle'.format(output_path,number_of_core_samples, step_size, name),'wb')) 

def get_ground_truth(output_path,number_of_core_samples, step_size, name):
  pairs = pickle.load(open('{}/true_pairs_{}_{}_{}.cPickle'.format(output_path,number_of_core_samples, step_size, name),'rb'))
  return np.concatenate(pairs, axis=0)
  

def get_local_ground_truth(output_path,number_of_core_samples, step_size, name):
  centers = pickle.load(open('{}/centers_{}_{}_{}.pickle'.format(output_path,number_of_core_samples, step_size, name),'rb'))
  magnitudes = pickle.load(open('{}/magnitudes_{}_{}_{}.pickle'.format(output_path,number_of_core_samples, step_size, name),'rb'))
  dimensions = pickle.load(open('{}/dimensions_{}_{}_{}.pickle'.format(output_path,number_of_core_samples, step_size, name),'rb'))
  return (centers, magnitudes, dimensions)

def generate_feature_vectors(number_of_core_samples, step_size):
  start = time.time()
  x = (np.random.rand(number_of_core_samples, len(feature_names)))
  lower_limits = np.array([feature_limits[f][0] for f in feature_names])
  upper_limits = np.array([feature_limits[f][1] for f in feature_names])
  perturbation_status_columns = []
  perturbation_status_columns.append('core')
  for feature_ind_1 in range(len(feature_names)):
    perturbation_status_columns.append(feature_names[feature_ind_1])
    for feature_ind_2 in range(feature_ind_1 + 1, len(feature_names)):
      perturbation_status_columns.append('{} and {}'.format(feature_names[feature_ind_1],feature_names[feature_ind_2]))
  
  data = []
  for ind in range(number_of_core_samples):
    data.append([])
    data[-1].append([x[ind, feature_names.index(key)] for key in feature_names])
    for feature_ind_1 in range(len(feature_names)):
      data[-1].append([x[ind, feature_names.index(key)] for key in feature_names])    
      data[-1][-1][feature_ind_1] += (upper_limits[feature_ind_1] - lower_limits[feature_ind_1]) * step_size
      for feature_ind_2 in range(feature_ind_1 + 1, len(feature_names)):
        data[-1].append([x[ind, feature_names.index(key)] for key in feature_names])
        data[-1][-1][feature_ind_1] += (upper_limits[feature_ind_1] - lower_limits[feature_ind_1]) * step_size
        data[-1][-1][feature_ind_2] += (upper_limits[feature_ind_2] - lower_limits[feature_ind_2]) * step_size
  data = np.array(data)
  feature_vectors = pd.DataFrame(data.reshape(data.shape[0], data.shape[1] * data.shape[2]), index = np.arange(number_of_core_samples), columns = pd.MultiIndex.from_product([perturbation_status_columns, feature_names], names=['perturbation_status','features']))
  end = time.time()
  print('Sampling features took {}'.format(end - start))  
  return feature_vectors, []

def simulate_model(feature_vectors, supplemental_data, number_of_core_samples, step_size, name, output_path):
  start = time.time()
  stacked_feature_vectors = pd.DataFrame(feature_vectors.stack(0).to_records())
  features = np.array(stacked_feature_vectors.loc[:, feature_names])
  indices = stacked_feature_vectors.loc[:, ['level_0','perturbation_status']]
  super_function = pickle.load(open('{}/model_{}_{}_{}.cPickle'.format(output_path,number_of_core_samples, step_size, name),'rb'))
  raw_results = super_function(features)
  individual_outputs = extract_outputs(raw_results)
  outputs = pd.concat((indices, individual_outputs), axis=1)  
  outputs = outputs.pivot(index = 'level_0', columns = 'perturbation_status')
  cols = [(out, pert) for out in output_names for pert in ['core']+feature_names+feature_pairs]
  outputs = outputs.loc[:, cols]
  end = time.time()
  print('Calculating outputs took {}'.format(end - start))
  return outputs

def extract_outputs(raw_results):
  outputs = pd.DataFrame(0, index = np.arange(raw_results.shape[0]), columns = output_names)
  outputs['Result'] = raw_results
  return outputs
