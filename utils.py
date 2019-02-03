import cPickle as pickle
import numpy as np
import pandas as pd
import functools
from scoop import futures
from scipy.interpolate import griddata
from scipy.signal import convolve2d
from sklearn.metrics import average_precision_score, roc_auc_score, precision_recall_curve

def calculate_hessian(model, data, step_size):
  """
  Computes the mixed derivative using finite differences mathod
  
  :param model: The imported model module
  :param data: The sampled data in structured form
  :param step_size: The dx time step taken between each 
  :returns: mixed derivative
  """
  hessian = pd.DataFrame(0, index = np.arange(data.shape[0]), columns=pd.MultiIndex.from_product([model.output_names, model.feature_pairs], names=['model.output_names','model.feature_pairs']))
  for output_name in model.output_names:
    hessian_calculation_helpers = create_hessian_calculation_columns(model, output_name)
    mixed_derivative = (data.loc[:, hessian_calculation_helpers[0]].values - data.loc[:, hessian_calculation_helpers[1]].values - data.loc[:, hessian_calculation_helpers[2]].values + data.loc[:, hessian_calculation_helpers[3]].values) / (step_size * step_size)
    mixed_derivative *= np.sign(data.loc[:, hessian_calculation_helpers[1]].values + data.loc[:, hessian_calculation_helpers[2]].values - 2 * data.loc[:, hessian_calculation_helpers[0]].values)
    hessian.loc[:, (output_name)] = mixed_derivative
  return hessian

def create_hessian_calculation_columns(model, output_name):
  hessian_calculation_helpers = []
  hessian_calculation_helpers.append([(output_name, 'core') for p in range(len(model.feature_pairs))])
  hessian_calculation_helpers.append([(output_name, p[:p.find(' ')]) for p in model.feature_pairs])
  hessian_calculation_helpers.append([(output_name, p[p.find(' and ') + 5:]) for p in model.feature_pairs])
  hessian_calculation_helpers.append([(output_name, p) for p in model.feature_pairs])
  return hessian_calculation_helpers

def max_filter_activation(matrix, filter_size):
  kernel = np.ones((filter_size,filter_size)) / np.power(filter_size, 2)
  out = convolve2d(matrix, kernel, mode='valid')
  return out.max()

def get_max_filters(matrix, num_filters = 90, threshold = 3):
  matrix_size = matrix.shape[0]
  filter_sizes = np.linspace(10, matrix_size, num_filters).astype(int)
  filter_results = list(futures.map(functools.partial(max_filter_activation, np.abs(matrix)), filter_sizes))
  if len(np.where(np.array(filter_results) >= threshold)[0]) == 0:
    return -1
  else:
    return np.where(np.array(filter_results) < threshold)[0][0] - 1

def create_interaction_map(hessian, feature_limits, core_feature_vectors, output_name, pair):
  first_feature = pair[:pair.find(' ')]
  second_feature = pair[pair.find(' and ') + 5:]
  coordinates = core_feature_vectors.loc[:, (first_feature, second_feature)]
  grid_x, grid_y = np.mgrid[feature_limits[first_feature][0]:feature_limits[first_feature][1]:(100j), feature_limits[second_feature][0]:feature_limits[second_feature][1]:(100j)]
  values = hessian.loc[:,(output_name, pair)] / hessian.loc[:, output_name].values.std()
  grid_z0 = griddata(coordinates, values, (grid_x, grid_y), method='linear', fill_value=0)
  return(grid_z0)


def rank_local(model, number_of_core_samples, step_size, name, threshold, plot, output_path):
  """
  Computes the mixed derivative for each sample, using finite differences mathod
  
  :param model: The imported model module
  :param data: The sampled data in structured form
  :param step_size: The dx time step taken between each 
  :returns: hessian matrix, with the core sample index as rows and feature pair as column name
  """
  feature_vectors = pd.DataFrame(np.load('{}/feature_vectors_{}_{}_{}.npy'.format(output_path, number_of_core_samples, step_size, name)), index = np.arange(number_of_core_samples), columns=pd.MultiIndex.from_product([['core']+model.feature_names+model.feature_pairs, model.feature_names], names=['perturbation_status','features']))
  outputs = pd.DataFrame(np.load('{}/outputs_{}_{}_{}.npy'.format(output_path, number_of_core_samples, step_size, name)), index = np.arange(number_of_core_samples), columns=pd.MultiIndex.from_product([model.output_names, ['core']+model.feature_names+model.feature_pairs], names=['outputs','perturbation_status']))
  hessian = calculate_hessian(model, outputs, step_size)
  core_feature_vectors = feature_vectors.loc[:, 'core']
  
  ranking = []
  for output_name in model.output_names:
    interaction_maps = list(futures.map(functools.partial(create_interaction_map, hessian, model.feature_limits, core_feature_vectors, output_name), model.feature_pairs))
    local_ranking = list(futures.map(functools.partial(get_max_filters, num_filters = 90, threshold = 3), interaction_maps))
    ranking_indices = np.argsort(local_ranking)[::-1]
    ranking.append((output_name, list(np.array(model.feature_pairs)[np.array(ranking_indices)]), np.array(local_ranking)[ranking_indices]))
  pickle.dump(obj = ranking, file = open('{}/local_ranking_{}_{}_{}.pickle'.format(output_path,number_of_core_samples, step_size, name),'wb'))
  return ranking

def rank_global(model, number_of_core_samples, step_size, name, plot, output_path):
  """
  Computes the mixed derivative for each sample, using finite differences mathod
  
  :param model: The imported model module
  :param data: The sampled data in structured form
  :param step_size: The dx time step taken between each 
  :returns: hessian matrix, with the core sample index as rows and feature pair as column name
  """
  outputs = pd.DataFrame(np.load('{}/outputs_{}_{}_{}.npy'.format(output_path, number_of_core_samples, step_size, name)), index = np.arange(number_of_core_samples), columns=pd.MultiIndex.from_product([model.output_names, ['core']+model.feature_names+model.feature_pairs], names=['outputs','perturbation_status']))
  hessian = calculate_hessian(model, outputs, step_size)
  ranked_hessian = hessian.abs().mean(axis=0)
  ranking = []
  for output_name in model.output_names:
    sorted_pairs = ranked_hessian.loc[output_name].sort_values()[::-1]
    ranking.append((output_name, list(sorted_pairs.index), sorted_pairs.values))
  pickle.dump(obj = ranking, file = open('{}/global_ranking_{}_{}_{}.pickle'.format(output_path,number_of_core_samples, step_size, name),'wb'))
  return ranking

def create_data(model, number_of_core_samples, step_size, name, output_path):
  model.set_ground_truth(number_of_core_samples, step_size, name, output_path)
  feature_vectors, supplemental_data = model.generate_feature_vectors(number_of_core_samples, step_size)
  outputs = model.simulate_model(feature_vectors, supplemental_data, number_of_core_samples, step_size, name, output_path)
  np.save('{}/outputs_{}_{}_{}'.format(output_path,number_of_core_samples, step_size, name), outputs)
  np.save('{}/feature_vectors_{}_{}_{}'.format(output_path,number_of_core_samples, step_size, name), feature_vectors)
  np.save('{}/supplemental_data_{}_{}_{}'.format(output_path,number_of_core_samples, step_size, name), supplemental_data)

def measure_global_accuracy(model, number_of_core_samples, step_size, name, output_path):
  y_predicted = pickle.load(open('{}/global_ranking_{}_{}_{}.pickle'.format(output_path,number_of_core_samples, step_size, name),'rb'))[0][2]
  predicted_pairs = pickle.load(open('{}/global_ranking_{}_{}_{}.pickle'.format(output_path,number_of_core_samples, step_size, name),'rb'))[0][1]
  true_pairs = ['{} and {}'.format(model.feature_names[pair[0]], model.feature_names[pair[1]]) for pair in model.get_ground_truth(output_path,number_of_core_samples, step_size, name)]
  y_actual = np.zeros(len(predicted_pairs))
  for pair_ind in range(len(predicted_pairs)):
    if (predicted_pairs[pair_ind] in true_pairs):
      y_actual[pair_ind] = 1
  np.save('{}/AUC_{}_{}'.format(output_path,number_of_core_samples, step_size), average_precision_score(y_actual, y_predicted))
  np.save('{}/PR_{}_{}'.format(output_path,number_of_core_samples, step_size), precision_recall_curve(y_actual, y_predicted))
  return(average_precision_score(y_actual, y_predicted))
