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
  hessian = pd.DataFrame(0, index = np.arange(data.shape[0]), columns=pd.MultiIndex.from_product([model.output_names, model.perturbation_feature_pairs + model.feature_names], names=['model.output_names','model.feature_pairs']))
  for output_name in model.output_names:
    hessian_calculation_helpers = create_hessian_calculation_columns(model, output_name)
    mixed_derivative = (data.loc[:, hessian_calculation_helpers[0]].values - data.loc[:, hessian_calculation_helpers[1]].values - data.loc[:, hessian_calculation_helpers[2]].values + data.loc[:, hessian_calculation_helpers[3]].values) / (step_size * step_size)
    mixed_derivative *= np.sign(data.loc[:, hessian_calculation_helpers[1]].values + data.loc[:, hessian_calculation_helpers[2]].values - 2 * data.loc[:, hessian_calculation_helpers[0]].values)
    hessian.loc[:, zip([output_name] * len(model.perturbation_feature_pairs), model.perturbation_feature_pairs)] = mixed_derivative
    hessian.loc[:, zip([output_name] * len(model.feature_names), model.feature_names)]  = np.array([(data.loc[:, (output_name,f)] - data.loc[:, (output_name,'core')]) / (step_size) for f in model.feature_names]).T
  return hessian

def create_hessian_calculation_columns(model, output_name):
  hessian_calculation_helpers = []
  hessian_calculation_helpers.append([(output_name, 'core') for p in range(len(model.perturbation_feature_pairs))])
  hessian_calculation_helpers.append([(output_name, p[:p.find(' ')]) for p in model.perturbation_feature_pairs])
  hessian_calculation_helpers.append([(output_name, p[p.find(' and ') + 5:]) for p in model.perturbation_feature_pairs])
  hessian_calculation_helpers.append([(output_name, p) for p in model.perturbation_feature_pairs])
  return hessian_calculation_helpers

def max_filter_activation(matrix, filter_size):
  kernel = np.ones((filter_size,filter_size)) / np.power(filter_size, 2)
  out = convolve2d(matrix, kernel, mode='valid')
  return out.max()

def get_max_filters(matrix, num_filters = 100, threshold = 3):
  matrix_size = matrix.shape[0]
  filter_sizes = np.linspace(5, matrix_size, num_filters).astype(int)
  filter_results = list(futures.map(functools.partial(max_filter_activation, np.abs(matrix)), filter_sizes))
  if len(np.where(np.array(filter_results) >= threshold)[0]) == 0:
    return -1
  else:
    return np.where(np.array(filter_results) < threshold)[0][0] - 1

def create_interaction_map(model, hessian, core_feature_vectors, output_name, method, pair):
  first_feature = pair[:pair.find(' ')]
  second_feature = pair[pair.find(' and ') + 5:]
  coordinates = core_feature_vectors.loc[:, (first_feature, second_feature)].values * 99 
  grid_x, grid_y = np.mgrid[0:100:(100j), 0:100:(100j)]
  if len(hessian) == 1:
    values = hessian.loc[:,(output_name, pair)]
  else:
    values = hessian.loc[:,(output_name, pair)] / hessian.loc[:, zip([output_name] * len(model.normalization_feature_pairs), model.normalization_feature_pairs)].values.std()
  grid_z0 = griddata(coordinates, values.values, (grid_x, grid_y), method=method, fill_value=0)
  return(grid_z0)

def rank_samples_in_pair(model, centers, magnitudes, dimensions, interaction_map_and_pair):
  interaction_map, pair = interaction_map_and_pair
  first_feature = pair[:pair.find(' ')]
  second_feature = pair[pair.find(' and ') + 5:]
  grid_x, grid_y = np.mgrid[0:1:(100j), 0:1:(100j)]
  y_true = np.zeros(interaction_map.shape)
  for dim_ind in range(len(dimensions)):
    if (model.feature_names.index(first_feature) in dimensions[dim_ind]) and (model.feature_names.index(second_feature) in dimensions[dim_ind]):
      first_v = np.where(dimensions[dim_ind] == model.feature_names.index(first_feature))[0]
      second_v = np.where(dimensions[dim_ind] == model.feature_names.index(second_feature))[0]
      y_true += (np.array((np.power(magnitudes[dim_ind][first_v] * (grid_x - centers[dim_ind][first_v]), 2) + np.power(magnitudes[dim_ind][second_v] * (grid_y - centers[dim_ind][second_v]), 2))) < 5.9).astype(int)
  return(interaction_map.flatten(), np.clip(y_true,0,1).flatten())

def measure_local_accuracy(model, number_of_core_samples, step_size, name, output_path):
  """
  Computes the mixed derivative for each sample, using finite differences mathod
  
  :param model: The imported model module
  :param data: The sampled data in structured form
  :param step_size: The dx time step taken between each 
  :returns: hessian matrix, with the core sample index as rows and feature pair as column name
  """
  feature_vectors = pd.DataFrame(np.load('{}/feature_vectors_{}_{}_{}.npy'.format(output_path, number_of_core_samples, step_size, name)), index = np.arange(number_of_core_samples), columns=pd.MultiIndex.from_product([model.perturbation_status_columns, model.feature_names], names=['perturbation_status','features']))
  outputs = pd.DataFrame(np.load('{}/outputs_{}_{}_{}.npy'.format(output_path, number_of_core_samples, step_size, name)), index = np.arange(number_of_core_samples), columns=pd.MultiIndex.from_product([model.output_names, model.perturbation_status_columns_output], names=['outputs','perturbation_status']))
  hessian = calculate_hessian(model, outputs, step_size)
  (centers, magnitudes, dimensions) = model.get_local_ground_truth(output_path,number_of_core_samples, step_size, name)

  core_feature_vectors = feature_vectors.loc[:, 'core']
  
  output_name = model.output_names[0]
  interaction_maps = list(futures.map(functools.partial(create_interaction_map, model, hessian, core_feature_vectors, output_name, 'nearest'), model.feature_pairs))
  local_ranking = list(futures.map(functools.partial(rank_samples_in_pair, model, centers, magnitudes, dimensions), zip(interaction_maps, model.feature_pairs)))
  ranking = np.concatenate(np.array(local_ranking), axis=1)
  accuracies = average_precision_score(ranking[1,:], np.abs(ranking[0,:]))
  ROCs = np.array(precision_recall_curve(ranking[1,:], np.abs(ranking[0,:])))

  pickle.dump(obj = accuracies, file = open('{}/local_accuracies_{}_{}_{}.pickle'.format(output_path,number_of_core_samples, step_size, name),'wb'))
  pickle.dump(obj = ROCs, file = open('{}/local_ROCs_{}_{}_{}.pickle'.format(output_path,number_of_core_samples, step_size, name),'wb'))
  return accuracies

def denoise_hessian(hessian):
  """
  Rectifies the uppermost and bottommost 0.1% of the hessian to remove noises
  """
  new_hessian = hessian.copy()
  s = new_hessian.values.shape
  c = new_hessian.columns
  new_hessian = new_hessian.values.flatten()
  new_hessian[np.argsort(new_hessian.flatten())[int(len(new_hessian.flatten()) * 0.999):]] = np.sign(new_hessian[np.argsort(new_hessian.flatten())[int(len(new_hessian.flatten()) * 0.999):]]) * np.abs(new_hessian.flatten()[np.argsort(new_hessian.flatten())][int(len(new_hessian.flatten()) * 0.999)])
  new_hessian[np.argsort(new_hessian.flatten())[::-1][int(len(new_hessian.flatten()) * 0.999):]] = np.sign(new_hessian[np.argsort(new_hessian.flatten())[::-1][int(len(new_hessian.flatten()) * 0.999):]])  * np.abs(new_hessian.flatten()[np.argsort(new_hessian.flatten())][::-1][int(len(new_hessian.flatten()) * 0.999)])
  return pd.DataFrame(new_hessian.reshape(s), columns = c)

def normalize_outputs(model, outputs):
  new_outputs = outputs.copy()
  for output_name in model.output_names:
    if (outputs.loc[:, output_name].max().max() == outputs.loc[:, output_name].min().min()):
      new_outputs.loc[:, output_name] = outputs.loc[:, output_name].values
    else:
      new_outputs.loc[:, output_name] = ((new_outputs.loc[:, output_name] - new_outputs.loc[:, output_name].min().min()) / np.abs((new_outputs.loc[:, output_name].max().max() - new_outputs.loc[:, output_name].min().min()))).values
  return new_outputs

def normalize_inputs(model, feature_vectors):
  new_feature_vectors = feature_vectors.copy()
  for feature in model.feature_names:
    if (new_feature_vectors.loc[:, (feature)].max().max() == new_feature_vectors.loc[:, (feature)].min().min()):
      new_feature_vectors.loc[:, (feature)] = new_feature_vectors.loc[:, (feature)]
    else:
      new_feature_vectors.loc[:, (feature)] = ((new_feature_vectors.loc[:, (feature)] - new_feature_vectors.loc[:, (feature)].min()) / (new_feature_vectors.loc[:, (feature)].max() - new_feature_vectors.loc[:, (feature)].min())).values
  return new_feature_vectors

def plot_interaction_map(model, name, matrix, output_name, first_variable, second_variable, x_coord, y_coord, output_path):
  """
  Plots a map of the parameter space for two input parameters, with the areas with more nonlinearity colored white
  
  :param ax: The axes on which to plot
  :param args: The arguments for the plot - 
                 The matrix to plot,
                 the name of the first variable
                 The name of the second variable,
                 The name of the first variable, as a key to the parameter limits dictionary
                 The name of the second variable, as a key to the parameter limits dictionary
                 the x coordinate of the sample being studied
                 the y coordinate of the sample being studied
  :returns: The axes with the plotted sample
  """  
  import matplotlib
  import matplotlib.cm as cm
  import matplotlib.pyplot as plt

  font = {'size'   : 14}

  matplotlib.rc('font', **font)
  fig = plt.figure(figsize=(5,5))
  ax = plt.subplot()

  maxValue = np.max(np.abs(matrix))
  img = ax.imshow((matrix), cmap = cm.bwr, origin='lower', vmin = -min(maxValue, 6), vmax = min(maxValue, 6), interpolation='spline16')

  first_variable = '{}'.format(first_variable)
  second_variable = '{}'.format(second_variable)
  ax.set_ylabel(r'$x_i$ = ' + first_variable)
  ax.set_xlabel(r'$y_i$ = ' + second_variable)
  ax.axes.set_xticks([0, 50, 99])
  ax.axes.set_yticks([0, 50, 99])
  xticks = np.linspace(np.array(model.feature_limits[first_variable]).min(), np.array(model.feature_limits[first_variable]).max(), 3)
  yticks = np.linspace(np.array(model.feature_limits[second_variable]).min(), np.array(model.feature_limits[second_variable]).max(), 3)
  ax.scatter([x_coord], [y_coord], marker='o', color='white', s = 250, edgecolors='black', linewidth=3)

  ax.set_yticklabels([xticks[tind] for tind in range(3)])
  ax.set_xticklabels([yticks[tind] for tind in range(3)])
  ax.axis([0, (100) - 1, 0, (100) - 1])

  # ax.scatter([x_coord_linear], [y_coord_linear], marker='o', color='blue', s = 250, edgecolors='black', linewidth=3)
  t = ax.set_title(r'$\mathregular{\frac{\delta ^2 F(\bar{x})}{\delta x_i \delta x_j}}$')
  # t = ax.set_title('{} and {} - '.format(first_variable, second_variable) + r'$\mathregular{\frac{\delta ^2 F(\bar{x})}{\delta x_i \delta x_j}}$')
  t.set_position([.5, 1.025])
  from mpl_toolkits.axes_grid1 import make_axes_locatable
  divider = make_axes_locatable(ax)
  cax = divider.append_axes("right", size="5%", pad=0.05)
  cb = plt.colorbar(img, cax=cax)
  cb.set_label("Nomralized mixed derivative", rotation=90)
  plt.savefig('{}/{}_{}_{}_{}_nonlinear_map.pdf'.format(output_path, name, output_name, first_variable, second_variable), transparent=True, bbox_inches='tight', format='pdf', dpi=600)
  # plt.close('all')
  
def rank_local(model, number_of_core_samples, step_size, name, threshold, output_path, top_k_to_plot):
  """
  Computes the mixed derivative for each sample, using finite differences mathod
  
  :param model: The imported model module
  :param data: The sampled data in structured form
  :param step_size: The dx time step taken between each 
  :returns: hessian matrix, with the core sample index as rows and feature pair as column name
  """
  feature_vectors = pd.DataFrame(np.load('{}/feature_vectors_{}_{}_{}.npy'.format(output_path, number_of_core_samples, step_size, name)), index = np.arange(number_of_core_samples), columns=pd.MultiIndex.from_product([model.perturbation_status_columns, model.feature_names], names=['perturbation_status','features']))
  core_feature_vectors = feature_vectors.loc[:, 'core'].copy()
  core_feature_vectors = normalize_inputs(model, core_feature_vectors)
  raw_outputs = pd.DataFrame(np.load('{}/outputs_{}_{}_{}.npy'.format(output_path, number_of_core_samples, step_size, name)), index = np.arange(number_of_core_samples), columns=pd.MultiIndex.from_product([model.output_names, model.perturbation_status_columns_output], names=['outputs','perturbation_status']))
  outputs = normalize_outputs(model, pd.DataFrame(raw_outputs))
  hessian = calculate_hessian(model, outputs, step_size)
  hessian = denoise_hessian(hessian)
  np.save('{}/hessian_{}_{}_{}'.format(output_path,number_of_core_samples, step_size, name), hessian)
  
  ranking = []
  for output_name in model.output_names:
    interaction_maps = list(futures.map(functools.partial(create_interaction_map, model, hessian, core_feature_vectors, output_name, 'linear'), model.feature_pairs))
    for ind in range(len(model.feature_pairs)):
      first_variable, second_variable = model.feature_pairs[ind].split(' and ')
      if ind < top_k_to_plot:
        model.set_ground_truth(number_of_core_samples, step_size, name, output_path)
        most_nonlinear_sample = hessian[output_name][model.feature_pairs[ind]].abs().idxmax()
        y_coord = 100 * (feature_vectors.loc[most_nonlinear_sample, 'core'][first_variable] - model.feature_limits[first_variable][0]) / (model.feature_limits[first_variable][1] - model.feature_limits[first_variable][0])
        x_coord = 100 * (feature_vectors.loc[most_nonlinear_sample, 'core'][second_variable] - model.feature_limits[second_variable][0]) / (model.feature_limits[second_variable][1] - model.feature_limits[second_variable][0])
        plot_interaction_map(model, name, interaction_maps[ind], output_name, first_variable, second_variable, x_coord, y_coord, output_path)
        features = []
        features.append(feature_vectors.loc[most_nonlinear_sample, 'core'])
        features.append(feature_vectors.loc[most_nonlinear_sample, first_variable])
        features.append(feature_vectors.loc[most_nonlinear_sample, second_variable])
        features.append(feature_vectors.loc[most_nonlinear_sample, model.feature_pairs[ind]])
        features = np.array(features)
    # local_ranking = list(futures.map(functools.partial(get_max_filters), interaction_maps))
    # ranking_indices = np.argsort(local_ranking)[::-1]
    # ranking.append((output_name, list(np.array(model.feature_pairs)[np.array(ranking_indices)]), np.array(local_ranking)[ranking_indices]))
  pickle.dump(obj = ranking, file = open('{}/local_ranking_{}_{}_{}.pickle'.format(output_path,number_of_core_samples, step_size, name),'wb'))
  return ranking

def rank_global(model, number_of_core_samples, step_size, name, output_path, top_k_to_plot):
  """
  Computes the mixed derivative for each sample, using finite differences mathod
  
  :param model: The imported model module
  :param data: The sampled data in structured form
  :param step_size: The dx time step taken between each 
  :returns: hessian matrix, with the core sample index as rows and feature pair as column name
  """
  outputs = pd.DataFrame(np.load('{}/outputs_{}_{}_{}.npy'.format(output_path, number_of_core_samples, step_size, name)), index = np.arange(number_of_core_samples), columns=pd.MultiIndex.from_product([model.output_names, model.perturbation_status_columns], names=['outputs','perturbation_status']))
  outputs = normalize_outputs(model, outputs)
  hessian = calculate_hessian(model, outputs, step_size)
  hessian = denoise_hessian(hessian)
  ranked_hessian = hessian.abs().mean(axis=0)
  ranking = []
  for output_name in model.output_names:
    sorted_pairs = ranked_hessian.loc[output_name].loc[model.normalization_feature_pairs].sort_values()[::-1]
    ranking.append((output_name, list(sorted_pairs.index), sorted_pairs.values))
  if top_k_to_plot:
    feature_vectors = pd.DataFrame(np.load('{}/feature_vectors_{}_{}_{}.npy'.format(output_path, number_of_core_samples, step_size, name)), index = np.arange(number_of_core_samples), columns=pd.MultiIndex.from_product([model.perturbation_status_columns, model.feature_names], names=['perturbation_status','features']))
    core_feature_vectors = feature_vectors.loc[:, 'core'].copy()
    core_feature_vectors = normalize_inputs(model, core_feature_vectors)
    interaction_maps = list(futures.map(functools.partial(create_interaction_map, model, hessian, core_feature_vectors, output_name, 'linear'), model.feature_pairs))
    ranked_feature_pairs = np.array(ranking)[:, 1][0][:top_k_to_plot]
    for pair_name in ranked_feature_pairs:
      ind = model.feature_pairs.index(pair_name)
      first_variable, second_variable = model.feature_pairs[ind].split(' and ')
      most_nonlinear_sample = hessian[output_name][model.feature_pairs[ind]].abs().idxmax()
      y_coord = 100 * (feature_vectors.loc[most_nonlinear_sample, 'core'][first_variable] - model.feature_limits[first_variable][0]) / (model.feature_limits[first_variable][1] - model.feature_limits[first_variable][0])
      x_coord = 100 * (feature_vectors.loc[most_nonlinear_sample, 'core'][second_variable] - model.feature_limits[second_variable][0]) / (model.feature_limits[second_variable][1] - model.feature_limits[second_variable][0])
      plot_interaction_map(model, name, interaction_maps[ind], output_name, first_variable, second_variable, x_coord, y_coord, output_path)

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
