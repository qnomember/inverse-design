from scipy.optimize import minimize
import numpy as np
from meep_flower_3D_dict import trans_flower

Nf = 3

def fom(params):
    """
    Calculates the figure of merit (FOM) to be minimized.

    Args:
        params : a dictiona 

    Returns:
        float: The negative FOM value to be minimized.
    """
    rmin, rmax, alpha, s= params
    params_dic ={'rmin':rmin,
                'rmax':rmax,
                'alpha':alpha,
                's':s,
                'Nf': Nf
    }
    t_rcp, t_lcp = trans_flower(params_dic)    
    #t_rcp = np.sin(rmax) 
    #t_lcp = np.cos(rmin)
    fmax = np.abs(t_rcp - t_lcp) / np.abs(t_rcp + t_lcp)
    fmin = - fmax
    return fmin

#####################
with open('opt_chiral_ev.txt', 'a') as f:
    f.write(f'Simulation of an Nf = {Nf} flower\n')
#####################

from scipy.optimize import differential_evolution
bounds =[(0,0.2),(0,0.5),(0,30),(0,20)]

# Define initial guesses (list of parameter sets within bounds)
init_guesses = [
    np.array([1.114698e-01, 2.192695e-01, 1.318258e+01, 1.154396e+01]) *  (1 + np.random.rand()/3 - 1/6)  for _ in range(20) 
]

## to print the process
####
from datetime import datetime

# Get current time and date
current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

def callback_func(xk, convergence=None):
    """
    Callback function to print intermediate results during optimization.

    Args:
        xk : Current parameters
        convergence : Convergence value (optional)
    """
    rmin, rmax, alpha, s = xk
    fom_value = fom(xk)
    print(f"Intermediate results: rmin={rmin:.2e}, rmax={rmax:.2e}, alpha={alpha:.2e}, s={s:.2e}, fom={fom_value:.2e}")
    with open('opt_chiral_ev.txt', 'a') as f:
        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        str1 = f'fom={fom_value:.6e}'
        array_str = ' '.join([f'{x:.6e}' for x in xk])
        f.write(f'{current_time}   {str1}  {array_str}\n')


# Run Differential Evolution
result = differential_evolution(
    fom,    # Function to minimize
    bounds,                # Parameter bounds
    strategy='best1bin',   # DE strategy (can adjust: 'best1bin', 'rand1exp', etc.)
    maxiter=40,          # Maximum iterations
    popsize=5,            # 
    tol=0.1e-9,              # Convergence tolerance
    mutation=(0.5, 1),     # Mutation factor range
    recombination=0.7,     # Crossover probability
 #  seed=42,               # Random seed for reproducibility
    disp=True,              # Print progress
    workers = 10,
   # updating='deferred',
  #init = init_guesses,
    callback = callback_func
    #x0 = [1.114698e-01, 2.192695e-01, 1.318258e+01, 1.154396e+01]
)

# Extract results
best_params = result.x
best_value = result.fun

# Print results
print("\nOptimization Results:")
print("Best parameters :")
rmin, rmax, alpha, s= best_params
print(f"rmin = {rmin:.2e}, rmax = {rmax:.2e}, alpha = {alpha:.2e}, s = {s:.2e}")
print(f"Minimum value of f: {best_value:.2e}")


# # Open file in append mode ('a')
with open('opt_chiral_ev.txt', 'a') as f:
    f.write(f"Number of function evaluations: {result.nfev}")


