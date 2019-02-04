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
extract_output_args = [ap_time]

# Input parameters

feature_names = ['gamma_CaDynamics_E2', 
'decay_CaDynamics_E2', 
'gCa_LVAstbar_Ca_LVAst',
'gCa_HVAbar_Ca_HVA',
'gSKv3_1bar_SKv3_1',
'gSK_E2bar_SK_E2',
'gImbar_Im',
'gNaTa_tbar_NaTa_t',
'g_pas',
'rate_E',
'rate_I',
'percentage_AP']  
feature_names = sorted(feature_names)
m = len(feature_names)

feature_name_coverter = {'decay_CaDynamics_E2' : r'Calcium Decay $\mathregular{\tau}$',
 'gCa_HVAbar_Ca_HVA': r'High Voltage activated Calcium',
 'gCa_LVAstbar_Ca_LVAst': r'Low Voltage activated Calcium',
 'gImbar_Im': r'M type Potassium',
 'gNaTa_tbar_NaTa_t': r'Fast Sodium',
 'gSK_E2bar_SK_E2': r'Calcium dependent Potassium',
 'gSKv3_1bar_SKv3_1': r'Voltage dependent Potassium',
 'g_pas': r'Leak current',
 'rate_E': r'Excitatory input rate',
 'rate_I': r'Inhibitory input rate',
 'percentage_AP': r'BAP height',
 'gamma_CaDynamics_E2': r'Calcium Gamma constant'}

original_input_param = {
'g_pas' : 5.89e-05,
'gSK_E2bar_SK_E2' : 0.0012,
'gCa_LVAstbar_Ca_LVAst' : 0.000187,
'gCa_HVAbar_Ca_HVA' : 5.55e-05,
'gSKv3_1bar_SKv3_1' : 0.000261,
'gNaTa_tbar_NaTa_t' : 0.0213,
'gImbar_Im' : 6.75e-05,
'gamma_CaDynamics_E2' : 0.000509,
'decay_CaDynamics_E2' : 122}

feature_limits = {}
for input_param in original_input_param.keys():
  feature_limits[input_param] = (original_input_param[input_param] - original_input_param[input_param] * 0.1, original_input_param[input_param] + original_input_param[input_param] * 0.1)

feature_limits['rate_E']                        = (300.0, 2400.0)
feature_limits['rate_I']                        = (300.0, 2400.0)
feature_limits['percentage_AP']                 = (0.0, 1.0)

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


def remove_spikes(voltage_vector, ap_time):
  """
  removes the action potential from the voltage trace
  
  :param voltage_vector: The voltage vector of the simulator output
  :param ap_time: the timing of the action potential
  :returns: result, the voltage trace without the action potential
  """     
  result = np.copy(voltage_vector['V'])
  result[int(ap_time / 0.025) : int((ap_time + 4) / 0.025)] = -90
  return result

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
  outputs['Integral'] = np.sum(remove_spikes(raw_results, ap_time[0]) + 90) / 40.0
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
  
  E_events_list = args[0]
  I_events_list = args[1]
  param_dict = args[2]  
  
  action_potential_at_soma = pickle.load(open('action_potential_at_soma.pickle', 'rb'))
  
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
h("soma insert Ca_LVAst")
h("soma insert Ca_HVA")
h("soma insert SKv3_1") 
h("soma insert SK_E2 ")
h("soma insert NaTa_t")
h("soma insert CaDynamics_E2")
h("soma insert Im ")
h("soma insert Ih")
h("ek = -85")
h("ena = 50")
h("gIhbar_Ih = 0.0128597")
h("g_pas = 1.0 / 12000 ")
h("celsius = 36")
'''
  
  exec(hoc_code)
  exec('h("tstop = {}")'.format(simulation_length))
  exec('h("v_init = {}")'.format(v_init))
  
  im = h.Impedance()
  h("access soma")
  im.loc(0.5)
  im.compute(0)
  Ri = im.input(0.5)

  h("access soma")
  h("nseg = 1")
  eSynlist = []
  eNetconlist = []
  iSynlist = []
  iNetconlist = []
  E_vcs = []
  I_vcs = []

  eSynlist.append(h.ProbAMPANMDA2_RATIO(0.5))  
  eSynlist[-1].gmax = 0.0004
  eSynlist[-1].mgVoltageCoeff = 0.08

  iSynlist.append(h.ProbUDFsyn2_lark(0.5))
  iSynlist[-1].tau_r = 0.18
  iSynlist[-1].tau_d = 5
  iSynlist[-1].e = - 80
  iSynlist[-1].gmax = 0.001
    
  
  semitrial_start = time.time()
  
  for key in param_dict.keys():
    if key is not "rate_E" and key is not "rate_I":
      h(key + " = " + str(param_dict[key]))
  
  E_events = np.array(E_events_list)[np.argmin(np.abs(np.array(E_events_list).mean(axis=1) - param_dict["rate_E"] / 1000.0))]
  I_events = np.array(I_events_list)[np.argmin(np.abs(np.array(I_events_list).mean(axis=1) - param_dict["rate_I"] / 1000.0))]

  E_vcs_events = []
  I_vcs_events = []
  E_vcs = h.VecStim()
  I_vcs = h.VecStim()

  I_vcs_events.append(h.Vector())
  events = np.where(I_events[:])[0] + 100
  for event in events:
    I_vcs_events[-1].append(event)

  E_vcs_events.append(h.Vector())
  events = np.where(E_events[:])[0] + 100
  for event in events:
    E_vcs_events[-1].append(event)
  
  eNetconlist = h.NetCon(E_vcs, eSynlist[-1])
  eNetconlist.weight[0] = 1
  eNetconlist.delay = 0
  E_vcs.play(E_vcs_events[-1])
  
  iNetconlist = h.NetCon(I_vcs, iSynlist[-1])
  iNetconlist.weight[0] = 1
  iNetconlist.delay = 0
  I_vcs.play(I_vcs_events[-1])

  ### Set the protocol
  
  h.finitialize()
  
  ### Simulate!
  
  prev_voltageVector = h.Vector()
  prev_voltageVector.record(eval("h.soma(0.5)._ref_v"))
  prev_timeVector = h.Vector()
  prev_timeVector.record(h._ref_t)

  h.tstop = 160
  h.run()
  h.tstop = 200

  voltageVector = h.Vector()
  voltageVector.record(eval("h.soma(0.5)._ref_v"))
  timeVector = h.Vector()
  timeVector.record(h._ref_t)

  # Simulate AP

  h('''objref clamp
  soma clamp = new SEClamp(.5)
  {clamp.dur1 = 1e9 clamp.rs=1e9}
  ''')
  
  active_timeVector = h.Vector(np.arange(0, h.tstop, 0.025))
  active_ones = np.array([float(1e9)] * len(active_timeVector))
  play_vector = np.array([0.0] * len(active_timeVector))
  
  action_potential_neuron_vector = h.Vector(list(np.maximum(np.array(prev_voltageVector)[int(150 / 0.025) : int(154 / 0.025)], attenuate_action_potential(np.array(np.copy(action_potential_at_soma)), param_dict["percentage_AP"]))))
  
  active_ones[int(ap_time / 0.025) : int((ap_time + 4) / 0.025)] = 0.00001
  play_vector[int(ap_time / 0.025) : int((ap_time + 4) / 0.025)] = action_potential_neuron_vector
  
  active_ones = h.Vector(active_ones)
  play_vector = h.Vector(play_vector)
  
  play_vector.play(h.clamp._ref_amp1, active_timeVector, 0)
  active_ones.play(h.clamp._ref_rs, active_timeVector, 0)

  h.run()
  
  timeVector = np.array(timeVector)
  voltageVector = np.array(voltageVector)
  
  trace = {}
  trace['T'] = timeVector[4000:]
  trace['V'] = np.array(voltageVector)[4000:]

  h.clamp = None
  active_ones = None
  play_vector = None
  del voltageVector, timeVector, prev_voltageVector, prev_timeVector, action_potential_neuron_vector
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
