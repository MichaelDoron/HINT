#!/usr/bin/python

import dill as pickle
import itertools
import pandas as pd
import numpy as np
from scoop import futures
from scipy.stats import ortho_group
import time

dt = 0.025
number_of_models = 10
simulation_length = 200
ap_time = 150
v_init = -90

# Input parameters

feature_names = ['NMDA','GABA','Delay']
# input_param_names = sorted(input_param_names)

feature_name_coverter = {'NMDA' : r'$\mathregular{{g_{NMDA}}_{Max}}$',
 'GABA': r'$\mathregular{{g_{{GABA}_{A}}}_{Max}}$',
 'Delay': r'$\mathregular{Delay}$'}

original_input_param = {
'NMDA' : 0 ,
'GABA' : 0 ,
'Delay' : 0}

feature_limits = {'NMDA' : [0, 0.008], 'GABA' : [0, 0.001], 'Delay' : [-50, 150]}

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


output_names = ['Integral']

### the cell structure and basic model

def attenuate_action_potential(voltage_vector, percentage):
  """
  Attenuates the action potential to a certain percentage, and adds it to the voltage vector 
  
  :param voltage_vector: The voltage vector of the action potential
  :param percentage: The percentage of the action potential to attenuate
  :returns: voltage_vector, the attentuated action potential voltage vector
  """   
  prev_min_voltage = np.min(voltage_vector)
  voltage_vector += np.abs(prev_min_voltage)
  voltage_vector *= percentage
  voltage_vector -= np.abs(prev_min_voltage)
  return voltage_vector

def create_presynaptic_spike_trains(input_param_limits):
  """
  Creates poisson distributed presynaptic spike traces for the exitation and inhibition
  
  :returns: voltage_vector, the attentuated action potential voltage vector
  """     
  number_of_E_synapses = 1
  number_of_I_synapses = 1
  full_E_events = np.random.poisson(input_param_limits['rate_E'][1] / 1000.0, size=(number_of_E_synapses, int(simulation_length - 1)))[0]
  full_I_events = np.random.poisson(input_param_limits['rate_I'][1] / 1000.0, size=(number_of_I_synapses, int(simulation_length - 1)))[0]
  
  E_events_list = []
  I_events_list = []
  AP_events_list = []
  
  E_events_list.append(np.copy(full_E_events))
  while (np.sum(full_E_events)):
    ind = np.random.choice(np.where(full_E_events)[0], 1)
    full_E_events[ind] = full_E_events[ind] - 1
    E_events_list.append(np.copy(full_E_events))
    print('E synapses remain: {}'.format(np.sum(full_E_events)))
  
  I_events_list.append(np.copy(full_I_events))
  while (np.sum(full_I_events)):
    ind = np.random.choice(np.where(full_I_events)[0], 1)  
    full_I_events[ind] = full_I_events[ind] - 1
    I_events_list.append(np.copy(full_I_events))
    print('I synapses remain: {}'.format(np.sum(full_I_events)))
  
  return E_events_list, I_events_list


def set_ground_truth(number_of_core_samples, step_size, name, output_path):
  """
  Prepares the simulator for receiving input parameter vectors and outputing results
  
  :param number_of_center_samples: The number of samples to be sampled
  :param input_param_names: The names of the parameters
  :param input_param_dx: The steps to take for derivative calculation
  :param input_param_limits: The maximum and minumum parameters values for sampling
  :param number_of_models: The number of models to build (a model is a set of parameters which are constant and don't change in the derivative steps)
  :returns: center_sample_param_dicts, a dictionary with the sample parameters
            all_sample_param_dicts, a dictionary with the parameters of all samples and their derivative steps
            supplemental_data, the constant parameters for the model of each trial
  """
  supplemental_data = []
  for model in range(number_of_models):
    E_events_lists, I_events_lists = create_presynaptic_spike_trains(input_param_limits)
    for trial in range(max(1, int(number_of_center_samples / number_of_models))):
      supplemental_data.append([E_events_lists, I_events_lists])
  pickle.dump(supplemental_data, open('{}/supplemental_data_{}_{}_{}.cPickle'.format(output_path,number_of_core_samples, step_size, name),'wb'))  

def extract_outputs(raw_results):
  outputs = pd.DataFrame(0, index = np.arange(raw_results.shape[0]), columns = output_names)
  outputs['Integral'] = np.sum(raw_results + 90) / 40.0
  return outputs

def simulate_single_param(args):  
  """
  Simulates a specific input parameter vector
  
  :param args: The parameters for the simulation: 
                 The list of excitatory presynpatic inputs,
                 The list of inhibitory presynpatic inputs,
                 and the input parameter dictionary
  :returns: The voltage trace of the simulation
  """   
  from neuron import h
  from neuron import gui
  h.load_file("nrngui.hoc")
  h.load_file("import3d.hoc")
  
  param_dict = args[0]  
  
  h.dt = 0.025
  h("create soma")
  h("access soma")
  h("nseg = 1")
  h("L = 20")
  h("diam = 20")
  h("insert pas")
  h("cm = 1")
  h("Ra = 100")
  h("forall nseg = 1")
  h("g_pas = 0.00005")
  h("forall e_pas = -70")
  exec('h("tstop = {}")'.format(simulation_length))
  (e_ns, e_pc, e_syn) = (None,None,None)
  (i_ns, i_pc, i_syn) = (None,None,None)

  e_ns = h.NetStim()
  e_ns.interval = 1
  e_ns.number = 1
  e_ns.start = 100
  e_ns.noise = 0
  e_syn = h.ProbAMPANMDA2_RATIO(0.5)
  e_syn.gmax = 1
  e_syn.mgVoltageCoeff = 0.08
  e_pc = h.NetCon(e_ns, e_syn)
  e_pc.weight[0] = 1
  e_pc.delay = 0

  i_ns = h.NetStim()
  i_ns.interval = 1
  i_ns.number = 1
  i_ns.start = 100
  i_ns.noise = 0                   
  
  i_syn = h.ProbUDFsyn2_lark(0.5)
  i_syn.tau_r = 0.18
  i_syn.tau_d = 5
  i_syn.e = - 80
  i_syn.Dep = 0
  i_syn.Fac = 0
  i_syn.Use = 0.6
  i_syn.u0 = 0
  i_syn.gmax = 1
  
  i_pc = (h.NetCon(i_ns, i_syn))
  i_pc.weight[0] = 1
  i_pc.delay = 0

  delaysVoltageVector = {}

  delayDiff = 1

  h.finitialize()

  nmda_cond = param_dict['NMDA']
  gaba_cond = param_dict['GABA']
  delay = param_dict['Delay']

  start = time.time()
  e_syn.gmax = nmda_cond
  i_syn.gmax = gaba_cond
  i_ns.start = 100 + delay
  voltageVector = h.Vector()
  timeVector = h.Vector()
  timeVector.record(h._ref_t)
  voltageVector.record(eval("h.soma(0.5)._ref_v"))
  h.run()

  timeVector = np.array(timeVector)
  voltageVector = np.array(voltageVector)
  
  trace = {}
  trace['T'] = timeVector
  trace['V'] = np.array(voltageVector)

  del voltageVector, timeVector
  return trace


def get_ground_truth(output_path,number_of_core_samples, step_size, name):
  print('This is a currently studied system, there is still no ground truth')

def generate_feature_vectors(number_of_core_samples, step_size):
  start = time.time()
  lower_limits = np.array([feature_limits[f][0] for f in feature_names])
  upper_limits = np.array([feature_limits[f][1] for f in feature_names])
  x = lower_limits + (np.random.rand(number_of_core_samples, len(feature_names))) * (upper_limits - lower_limits)
  perturbation_status_columns = []
  perturbation_status_columns.append('core')
  for feature_ind_1 in range(len(feature_names)):
    perturbation_status_columns.append(feature_names[feature_ind_1])
    for feature_ind_2 in range(len(feature_names)):
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


