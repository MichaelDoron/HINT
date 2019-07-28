# HINT
HINT: Discovering Unexpected Local Nonlinear Interactions in Scientific Black-box Models

## To run HINT on browser (with limited functionality)
https://hub.gke.mybinder.org/user/michaeldoron-hint-zyoamyhc/notebooks/HINT.ipynb

## To reproduce paper results:

### Synthetic models:

#### Gaussian 10:

python hint.py -m Gaussian_10 -o results -n 1 --create_data --rank_global --measure_global_accuracy


#### Gaussian 100:

python hint.py -m Gaussian_100 -o results -n 1 --create_data --rank_global --measure_global_accuracy


#### Complex function:

python hint.py -m Complex_function -o results -n 1 --create_data --rank_global --measure_global_accuracy

python hint.py -m Complex_function -o results -n 10 --create_data --rank_global --measure_global_accuracy

python hint.py -m Complex_function -o results -n 100 --create_data --rank_global --measure_global_accuracy

python hint.py -m Complex_function -o results -n 1000 --create_data --rank_global --measure_global_accuracy



#### Local (m=2):


python hint.py -m Local -o results -n 10 --create_data --measure_local_accuracy --name Local

python hint.py -m Local -o results -n 100 --create_data --measure_local_accuracy --name Local

python hint.py -m Local -o results -n 1000 --create_data --measure_local_accuracy --name Local

python hint.py -m Local -o results -n 10000 --create_data --measure_local_accuracy --name Local



###  Computational models:



python hint.py -m NMDA -o results -n 10000 --step_size 0.01 --rank_local --name 'NMDA'

python hint.py -m BAP -o results -n 3000 --step_size 0.1 --rank_local --name 'BAP'

python hint.py -m SOMA -o results -n 1000 --step_size 0.1 --rank_local --name 'SOMA'


