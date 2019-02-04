#!~/anaconda2/bin/python
import sys
import os
import time
number_of_samples = sys.argv[1]
iteration = sys.argv[2]
task = sys.argv[3]

if task == 'ga2m':
	dataset = "ga2m_synt_{}_{}.train".format(number_of_samples, iteration)
	datasetDisc = "ga2m_synt_{}_{}_disc.train".format(number_of_samples, iteration)
	attset = "ga2m_synt.attr"
	attsetDisc = "ga2m_synt_disc_{}_{}_disc.attr".format(number_of_samples, iteration)
	residualPath = "random_function_{}_{}.res".format(number_of_samples, iteration)
	modelPath = "gam_ga2m_{}_{}.model".format(number_of_samples, iteration)
	FASTOutput = "random_function_{}_{}.out".format(number_of_samples, iteration)
else:
	dataset = "random_function_{}_{}_{}.train".format(iteration, number_of_samples, task)
	datasetDisc = "random_function_{}_{}_disc_{}.train".format(iteration, number_of_samples, task)
	attset = "random_function_{}.attr".format(task)
	attsetDisc = "random_function_disc_{}_{}_{}.attr".format(task, iteration, number_of_samples)
	residualPath = "random_function_{}_{}_{}.res".format(iteration, number_of_samples, task)
	modelPath = "gam_{}_{}_{}.model".format(task, iteration, number_of_samples);
	FASTOutput = "random_function_{}_{}_{}.out".format(iteration, number_of_samples, task)	


start = time.time()
os.system("/usr/bin/java mltk.core.processor.Discretizer -i {} -o {} -t {} -r {} -m {}".format(dataset, datasetDisc, dataset, attset, attsetDisc))
os.system("/usr/bin/java mltk.predictor.tree.ensemble.brt.LSBoostLearner -t {} -m {} -o {} -r {} -V False".format(datasetDisc, 1000, modelPath, attsetDisc))
os.system("/usr/bin/java mltk.predictor.evaluation.Predictor -d {} -m {} -r {} -R {}".format(datasetDisc, modelPath, attsetDisc, residualPath))
os.system("/usr/bin/java mltk.predictor.gam.interaction.FAST -d {} -m {} -r {} -R {} -o {} -b 8".format(dataset, modelPath, attset, residualPath, FASTOutput))
end = time.time()

print(end - start)