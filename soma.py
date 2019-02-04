#!/usr/bin/python

import dill as pickle
import itertools
import pandas as pd
import numpy as np
from scoop import futures
from scipy.stats import ortho_group
import time


dt = 0.025
number_of_models = 1
simulation_length = 2000
v_init = -82
extract_output_args = []
result_indices_parameters = True

# Input parameters

feature_names = ['gamma_CaDynamics_E2', 
'decay_CaDynamics_E2', 
'gCa_LVAstbar_Ca_LVAst',
'gCa_HVAbar_Ca_HVA',
'gSKv3_1bar_SKv3_1',
'gSK_E2bar_SK_E2',
'gNap_Et2bar_Nap_Et2',
'gNaTa_tbar_NaTa_t',
'gK_Pstbar_K_Pst',
'gK_Tstbar_K_Tst',
'g_pas']
feature_names = sorted(feature_names)

original_input_param = {
'g_pas' : 0.0000338 ,
'gSK_E2bar_SK_E2' : 0.0441 ,
'gK_Tstbar_K_Tst' : 0.0812 ,
'gK_Pstbar_K_Pst' : 0.00223 ,
'gNap_Et2bar_Nap_Et2' : 0.00172,
'gCa_LVAstbar_Ca_LVAst' : 0.00343 ,
'gCa_HVAbar_Ca_HVA' : 0.000992,
'gSKv3_1bar_SKv3_1' : 0.693 ,
'gNaTa_tbar_NaTa_t' : 2.04 ,
'gamma_CaDynamics_E2' : 0.000501 ,
'decay_CaDynamics_E2' : 460.0 }

feature_limits = {}
for input_param in original_input_param.keys():
  feature_limits[input_param] = (original_input_param[input_param] - original_input_param[input_param] * 1.00, original_input_param[input_param] + original_input_param[input_param] * 1.00)

m = len(feature_names)

feature_pairs = sorted([sorted(pair) for pair in itertools.combinations(range(len(feature_names)), 2)])
feature_pairs = ['{} and {}'.format(feature_names[p[0]], feature_names[p[1]]) for p in feature_pairs]

normalization_feature_pairs = []
for feature_ind_1 in range(len(feature_names)):
  for feature_ind_2 in range(feature_ind_1 + 1, len(feature_names)):
    normalization_feature_pairs.append('{} and {}'.format(feature_names[feature_ind_1],feature_names[feature_ind_2]))

perturbation_feature_pairs = []
for feature_ind_1 in range(len(feature_names)):
  for feature_ind_2 in range(feature_ind_1, len(feature_names)):
    perturbation_feature_pairs.append('{} and {}'.format(feature_names[feature_ind_1],feature_names[feature_ind_2]))

perturbation_status_columns = []
perturbation_status_columns.append('core')
for feature_ind_1 in range(len(feature_names)):
  perturbation_status_columns.append(feature_names[feature_ind_1])
  for feature_ind_2 in range(feature_ind_1, len(feature_names)):
    perturbation_status_columns.append('{} and {}'.format(feature_names[feature_ind_1],feature_names[feature_ind_2]))


output_names = ['ISI_CV']

def set_ground_truth(number_of_core_samples, step_size, name, output_path):
  print('This is a currently studied system, there is still no ground truth')

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
  
  hoc_code = '''h("create soma")
h("access soma")
h("nseg = 1")
h("L = 20")
h("diam = 20")
h("insert pas")
h("cm = 1")
h("Ra = 150")
h("forall nseg = 1")
h("forall e_pas = -90")

h("soma insert Ca_LVAst ")
h("soma insert Ca_HVA ")
h("soma insert SKv3_1 ")
h("soma insert SK_E2 ")
h("soma insert K_Tst ")
h("soma insert K_Pst ")
h("soma insert Nap_Et2 ")
h("soma insert NaTa_t")
h("soma insert CaDynamics_E2")
h("soma insert Ih")

h("ek = -85")
h("ena = 50")
h("gIhbar_Ih = 0.0002")
h("g_pas = 1.0 / 12000 ")
h("celsius = 36")
'''
  
  exec(hoc_code)
  exec('h("tstop = {}")'.format(simulation_length))
  exec('h("v_init = {}")'.format(v_init))

  for key in param_dict.keys():
    h(key + " = " + str(param_dict[key]))
  
  iclamp = h.IClamp(0.5)
  iclamp.delay = 500
  iclamp.dur = 1400

  amp = 0.05
  iclamp.amp = amp
  im = h.Impedance()
  h("access soma")
  h("nseg = 1")
  im.loc(0.5)
  im.compute(0)
  Ri = im.input(0.5)
  
  semitrial_start = time.time()
  
  ### Set the protocol
  
  h.finitialize()
  
  ### Simulate!
  
  voltageVector = h.Vector()
  voltageVector.record(eval("h.soma(0.5)._ref_v"))
  timeVector = h.Vector()
  timeVector.record(h._ref_t)

  h.run()
  
  timeVector = np.array(timeVector)
  voltageVector = np.array(voltageVector)

  trace = {}
  trace['T'] = timeVector[4000:]
  trace['V'] = np.array(voltageVector)[4000:]
  trace['stim_start'] = [500]
  trace['stim_end'] = [1900]
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


