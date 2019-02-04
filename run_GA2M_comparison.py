#!~/anaconda2/bin/python
from scoop import futures
import os
import time
def f(args):
    number_of_samples, iteration, task = args
    params = '%d %d %s' % (number_of_samples, iteration, task)
    jobName = 'job_create_data_per_parameter_' + params.replace(' ','_') + '.txt'
    pythonScriptAndParams = 'single_run_GA2M.py ' + params
    os.system('python ' + pythonScriptAndParams)


if __name__ == '__main__':
    args = []
    times = []
    for task in ['hard']:
      for number_of_samples in [1]:
        for iteration in range(30):
          args.append((number_of_samples, iteration, task))
          allstart = time.time()
          f((number_of_samples, iteration, task))
          times.append(time.time() - allstart)
          print(times[-1])
    print(np.mean(times))
    print(np.std(times))
