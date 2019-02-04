#!/usr/bin/python

import cPickle as pickle
import itertools
import pandas as pd
import numpy as np
from scoop import futures

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
  'X{}'.format(ind) : (0,1) for ind in range(1, 10 + 1)
  }
feature_limits['X4'] = (0.6, 1.0)
feature_limits['X5'] = (0.6, 1.0)
feature_limits['X8'] = (0.6, 1.0)
feature_limits['X10'] = (0.6, 1.0)

output_names = ['Result']

def f(x1, x2, x3, x4, x5, x6, x7, x8, x9, x10):
  result = np.power(np.pi, (x1 * x2)) * np.sqrt(2 * x3) - (1.0 / np.sin(x4)) + np.log(x3 + x5) - (x9 / x10) * np.sqrt(x7 / x8) - x2 * x7
  return np.array(result)

def set_ground_truth(number_of_core_samples, step_size, name, output_path):
  pickle.dump([(0,1),(0,2),(1,2),(6,7),(6,8),(6,9),(7,8),(7,9),(8,9),(1,6),(2,4)], open('{}/VIN_true_pairs_{}_{}_{}.pickle'.format(output_path, number_of_core_samples, step_size, name),'wb'))

def simulate_model(feature_vectors, supplemental_data, number_of_core_samples, step_size, name, output_path):
  stacked_feature_vectors = pd.DataFrame(feature_vectors.stack(0).to_records())
  features = np.array(stacked_feature_vectors.loc[:, feature_names])
  indices = stacked_feature_vectors.loc[:, ['level_0','perturbation_status']]
  raw_results = np.array(list(futures.map(simulate_single_sample, features)))
  individual_outputs = extract_outputs(raw_results)
  outputs = pd.concat((indices, individual_outputs), axis=1)  
  outputs = outputs.pivot(index = 'level_0', columns = 'perturbation_status')
  cols = [(out, pert) for out in output_names for pert in ['core']+feature_names+feature_pairs]
  outputs = outputs.loc[:, cols]
  return outputs

def generate_feature_vectors(number_of_core_samples, step_size):
  feature_vectors = pd.DataFrame(0, index = np.arange(number_of_core_samples), columns=pd.MultiIndex.from_product([['core']+feature_names+feature_pairs, feature_names], names=['perturbation_status','features']))
  for feature_ind in range(len(feature_names)):
    lower_limit = feature_limits[feature_names[feature_ind]][0]
    upper_limit = feature_limits[feature_names[feature_ind]][1]
    feature_vectors.loc[:, [('core',feature_names[feature_ind])]] = (lower_limit + (np.random.rand(number_of_core_samples) * (upper_limit - lower_limit)))
  for feature_ind in range(len(feature_names)):
    first_feature = feature_names[feature_ind]
    feature_vectors.loc[:, [(first_feature)]] = np.copy(feature_vectors.loc[:, [('core')]])
    feature_vectors.loc[:, [(first_feature, first_feature)]] += ((feature_limits[first_feature][1] - feature_limits[first_feature][0]) * step_size)
    for second_feature_ind in range(len(feature_names[(feature_ind + 1):])):
      second_feature = feature_names[(feature_ind + 1):][second_feature_ind]
      feature_vectors.loc[:, [('{} and {}'.format(first_feature, second_feature))]] = np.copy(feature_vectors.loc[:, [(first_feature)]])
      feature_vectors.loc[:, [('{} and {}'.format(first_feature, second_feature), second_feature)]] += ((feature_limits[second_feature][1] - feature_limits[second_feature][0]) * step_size)
  return feature_vectors, []

def get_ground_truth(output_path,number_of_core_samples, step_size, name):
  return [(0,1),(0,2),(1,2),(6,7),(6,8),(6,9),(7,8),(7,9),(8,9),(1,6),(2,4)]

def simulate_single_sample(feature_vector):
  return f(*feature_vector)

def extract_outputs(raw_results):
  outputs = pd.DataFrame(0, index = np.arange(raw_results.shape[0]), columns = output_names)
  outputs['Result'] = raw_results
  return outputs
