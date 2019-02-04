#!/usr/bin/python

import dill as pickle
import itertools
import pandas as pd
import numpy as np
from scoop import futures
from scipy.stats import ortho_group
import time

sort_key = lambda x : int(x[1:])
feature_names = sorted(['X{}'.format(ind) for ind in range(1, 10 + 1)], key=sort_key)

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
  'X{}'.format(ind) : (-5,5) for ind in range(1, 10 + 1)
  }

output_names = ['Result']

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

def set_ground_truth(number_of_core_samples, step_size, name, output_path):
  true_pairs = []
  while ((len(true_pairs) == 0) or (len(true_pairs) == (len(feature_names) * (len(feature_names) - 1)))):
    true_pairs = []
    variable_subset_sizes = np.min(np.array([np.array(1.5 + np.random.exponential(scale = 1.0, size=25)).astype(int), [len(feature_names)] * 25]), axis=0)
    variables = np.arange(len(feature_names))
    Z = [np.random.choice(variables, variable_subset_sizes[ind], replace=False) for ind in range(25)]
    [true_pairs.extend(list(itertools.permutations(z, 2))) for z in Z]
    true_pairs = list(set(true_pairs))
  functions = [generate_gaussian(z) for z in Z]
  
  def super_function(inp):
    result = 0
    for function in functions:
      result += function(inp)
    return result
  super_function.__name__ = str(np.random.rand())
  pickle.dump(list(set(true_pairs)), open('{}/true_pairs_{}_{}_{}.cPickle'.format(output_path,number_of_core_samples, step_size, name),'wb'))  
  pickle.dump(super_function, open('{}/model_{}_{}_{}.cPickle'.format(output_path,number_of_core_samples, step_size, name),'wb'))  

def get_ground_truth(output_path,number_of_core_samples, step_size, name):
  return pickle.load(open('{}/true_pairs_{}_{}_{}.cPickle'.format(output_path,number_of_core_samples, step_size, name),'rb'))

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
