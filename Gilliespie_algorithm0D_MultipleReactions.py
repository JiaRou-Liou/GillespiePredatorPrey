"""
Simulate a 0D predator-prey reaction using Gillespie algorithm (stochastic simulation algorithm). 

The simulation setup and the parameters are inspired by McKane A. j., Newman T. J., 2005, Physical review letters, 94, 218102. 
"""
import numpy as np
import matplotlib.pyplot as py
from pathlib import Path
import random
import time
import matplotlib
import math
import time

from functools import partial #ref: https://stackoverflow.com/questions/15331726/how-does-functools-partial-do-what-it-does

class reactants:
    """
    A class for storing the names and the number of reactants. 

    Attributes: 
        reactant_IDs (str; iterable object): A 1D array of the names (identifiers) of the reactants.

        number_of_particles (int; iterable object): A 1D array of the number of the particles of each reactant. 
        Should have the same index as reactant_IDs.

        data (dict): Dictionary of reactants and their current number of particles.

    """
    def __init__(self,reactant_IDs,number_of_particles):
        """
        Initialise an reactants object. 

        Parameters: 
            reactant_IDs (str; iterable object): A 1D array of the names (identifiers) of the reactants.

            number_of_particles (int; iterable object): A 1D array of the number of the particles of each reactant. 
            Should have the same index as reactant_IDs.

            data (dict): Dictionary of reactants and their current number of particles.
        """
        self.reactant_IDs = reactant_IDs
        self.number_of_particles = number_of_particles
        self.data = dict(zip(reactant_IDs, number_of_particles))
    
class reaction_channels:
    """
    A class for computing and storing the propensities of each reaction in situ. 

    Attributes:
        reactions (dict): A dictionary that stores the reaction channels, number of particles for each reactants, 
        the number of particles of the products, etc. 

        particle_dict (dict): Current number of particles of each species. 

        nonlinear_cf (dict): Perform nonlinear mathematical processes. 

        propensities: compute the propensities each time reaction_channels is called. 
    """
    def __init__(self, reactions, particle_dict, nonlinear_cf):
        """
        Initialise an reaction_channels object. 

        Parameters: 
            reactions (dict): A dictionary that stores the reaction channels, number of particles for each reactants, 
        the number of particles of the products, etc. 

            particle_dict (dict): Current number of particles of each species. 

            nonlinear_cf (dict): Perform nonlinear mathematical processes. 
        """

        self.reactions = reactions
        self.particle_dict = particle_dict
        self.nonlinear_cf = nonlinear_cf ## Optional nonlinear function
        
        self.propensities = self.compute_all_propensities() #Call the method compute_all_propensities() at each run.
        


    def compute_all_propensities(self):
        """
        Compute the propensities of each reaction channel. 
            
            Returns: 
                dict: propensities of each reaction channel with the name R{index of each reaction channel}
        """
        prop_dict = {}
        for idx, reaction in enumerate(self.reactions):
            a = 1
            valid = True
            for i, r in enumerate(reaction['reactants']):
                
                N = self.particle_dict[r]
                n_r = reaction['num_r'][i]
                if N < n_r:
                    valid = False
                    break  # Not enough particles to react
                    
                if self.nonlinear_cf:
                    modifier = self.nonlinear_cf.get(r, lambda N: 1)(N)
                else:
                    modifier = 1
                    
                if n_r > 1:
                    a *= math.comb(N, n_r) * modifier
                else:
                    a *= N*modifier
            if valid:
                a *= reaction['rate']
                name = f"R{idx}"
                prop_dict[name] = {'a': a, 'index': idx}
        return prop_dict

class processes:
    """
    A class that executes the selected reaction. 

    Attributes: 
        reactions (dict): A dictionary that stores the reaction channels, number of particles for each reactants, 
        the number of particles of the products, etc. 

        particle_dict (dict): A dictionary that records the current number of particles of each species. 
    """
    def __init__(self, reactions, particle_dict):
        """
        Initialise the processes object. 

        Parameters: 
            reactions (dict): A dictionary that stores the reaction channels, number of particles of each reactants, 
        the number of particles of the products, etc. 

            particle_dict (dict): A dictionary that records the current number of particles of each species. 

        """
        self.reactions = reactions
        self.particle_dict = particle_dict

    def execute(self, selected_index):
        """
        Add or subtract the particle numbers of the reactants and the products according to the selected reaction. 

        Paramerters: 
            selected_index (int): Index representing the selected reaction. 

        Returns: 
            dict: Current particle numbers of each species after the reaction. 
        """
        reaction = self.reactions[selected_index]

        # Subtract reactants
        for i, r in enumerate(reaction['reactants']):
            self.particle_dict[r] -= reaction['num_r'][i]

        # Add products
        for i, p in enumerate(reaction['products']):
            self.particle_dict[p] += reaction['num_p'][i]

        return self.particle_dict

#Nonlinear reaction coefficient        
class NonlinearCoefficients: 
    @staticmethod
    def allee_effect(N, A, h=2):
        """Smooth Allee effect (Hill-type function)""" #saturation function
        return N**h / (A**h + N**h)

    @staticmethod
    def inhibition(N, K, h=1):
        """Inhibition function"""
        return K**h / (K**h + N**h)

    @staticmethod
    def michaelis_menten(N, Km):
        """Michaelis-Menten kinetics"""
        return N / (Km + N)

    @staticmethod
    def logistic_growth(N, K):
        """Logistic coefficient (self-limiting growth)"""
        return 1 - (N / K)

    @staticmethod
    def prey_birth_with_space(n, m, N, b):
        E = N - n - m
        return 2 * b * (m / N) * E if E > 0 else 0
    
        
        
class record:
    """
    A class that records the non-uniform time steps and the evolution of the particle dictionary. 

    Attributes: 
        non_uniform_time_steps (float; 1D numpy array): the record of the non-uniform time (t_i = t_{i-1}+non-uniform time step)

        dictinary (dict): the current state of particle number of each species. 

        record_dict (dict): a record log of the states of the particles at each time step. 

        time_evol: update record_dict every time the records object is called.  
    """
    
    def __init__(self,non_uniform_time_steps,dictionary,record_dict):
        """
        Initialise the records object. 

        Parameters: 
            non_uniform_time_steps (float; 1D numpy array): the record of the non-uniform time (t_i = t_{i-1}+non-uniform time step)

            dictinary (dict): the current state of particle number of each species. 

            record_dict (dict): a record log of the states of the particles at each time step.
        """
        self.t_gill = non_uniform_time_steps        
        self.dictionary = dictionary
        self.record_dict = record_dict
        self.time_evol = self.update()
        
    def update(self):
        """
        Update the time and the particle number distribution. 

        Returns: 
            dict: the updated record of the particle dictionary. 
        """
        
        for keys, values in (self.dictionary.items()):
            self.record_dict[keys].append(values)
        self.record_dict['t_gill'].append(self.t_gill)
        return self.record_dict
    
    @staticmethod
    def uniform_time_grid(non_uniform_time, value):
        """
        One-dimensional linear interpolation for monotonically increasing sample points. 

        Parameters: 
            t_gill (float; 1D array): non-uniform time steps. 

            value (float; 1D array): 1D array to be interpolated. 

        Returns: 
            float: 1D array with discrete time step. 

            float: 1D array with one-dimensional piecewise linear interpolant 
            to a function with given discrete data points (t_gill[i], value[i]), evaluated at t_gill[i]
        """
        t_gill = non_uniform_time
        t_uniform = np.linspace(t_gill[0], t_gill[-1], num=len(t_gill))
        x_uniform = np.interp(t_uniform, t_gill, value)
        return t_uniform, x_uniform
        
#---Take the microscopic predator-prey interactions (individual level model (ILM)) as an example.---     

#Parameters
reactant_IDs = np.array(['A', 'B', 'E'])        #A: predator, B: prey, E: vacant space (used to conserve the number of total site)
V = 3200        #Total number of all species.
b, p1, p2, d1, d2 = 0.1/V, 0.25/V, 0.05/V, 0.1, 0.001       #Birth rate, 2 different types of predation types, ...
                                                                # mortality rates of the predator A, and prey B

number_of_particles = np.ones(len(reactant_IDs))*V*0.2      #Initial number of predator A, prey B, and the empty spaces E
number_of_particles[2] = V-(number_of_particles[0]+number_of_particles[1])
dictionary = reactants(reactant_IDs, number_of_particles).data

# Define reactions
reactions = [
    {"reactants": ['A'], "products": ['E'], "num_r": [1], "num_p": [1], "rate": d1},
    {"reactants": ['B'], "products": ['E'], "num_r": [1], "num_p": [1], "rate": d2},
    {"reactants":['A','B'], "products":['A'], "num_r":[1,1], "num_p":[2], 'rate':2*p1},
    {"reactants":['A','B'], "products":['A','E'], "num_r":[1,1], "num_p":[1,1], "rate":2*p2},
    {"reactants":['B','E'], "products":['B','B'], "num_r":[1,1], "num_p":[1,1], "rate":2*b}
]

# (Optional) Define only "species-specific" nonlinear behavior 
#nonlinear_cf = {
#    'A': partial(NonlinearCoefficients.allee_effect, A=5, h=2),
#    'B': partial(NonlinearCoefficients.inhibition, K=80)
#}

# (Optional) Manually add special (non-linear) prey birth reaction which is based on total population: m -> m + 1
#n = dictionary['A']
#m = dictionary['B']
#special_rate = prey_birth_with_space(n, m, N=V, b=b)  # define total_population and birth_rate
#propensities['special_birth'] = {'a': special_rate, 'index': 'special'}


start = time.time()         #Start time of the simulation. 
length_of_time = 250000         #Number of steps of the simulation. 

    
total=[]        #Population checker. Stays constant throughout the simulation.
total.append(V)   
t = 0
values = [[] for i in range(len(reactant_IDs))]

record_dict = dict(zip(reactant_IDs,values))
for key, value in dictionary.items():
    record_dict[key].append(value)
record_dict['t_gill']=[0]           #Set up a dictionary to record the time evolution of the particles. 


channel = reaction_channels(reactions, dictionary, nonlinear_cf=None)
propensities = channel.propensities

for i in range(length_of_time):
    r1 = np.random.random()
    r2 = np.random.random()

    
    a_0 = sum(item['a'] for item in propensities.values())

    if a_0 == 0:
        break  # No more reactions

    tau = (1.0 / a_0) * np.log(1.0 / r1)
    t += tau
    

    # Select reaction
    cumulative = 0
    threshold = r2*a_0
    selected_index = None
    for name, data in propensities.items():
        prev = cumulative
        cumulative += data['a']
        if prev <= threshold < cumulative:
            selected_index = data['index']
            break

    dictionary = processes(reactions, dictionary).execute(selected_index)       ## Execute the reaction selected
    
    channel = reaction_channels(reactions, dictionary, nonlinear_cf=None)
    propensities = channel.propensities
    
    
    evolution = record(t,dictionary,record_dict).time_evol
    
    total.append(sum(np.sum(dictionary[key]) for key in reactant_IDs))

        
    

    
t_uniform, x_uniform = record.uniform_time_grid(evolution['t_gill'], np.array(evolution['A'])/V)
t_uniform,x_uniformB = record.uniform_time_grid(evolution['t_gill'], np.array(evolution['B'])/V)


end = time.time()           
print(end - start)          #The duration of the Gillespie algorithm written 


#---Plot the figure---
freq = np.fft.fftfreq(len(t_uniform),d=t_uniform[1] - t_uniform[0])
FFTA = np.fft.fft(x_uniform).real

py.plot(freq, FFTA, label='A')
py.xlim(-0.05,0.05)
py.savefig('Power_spectra.png')

#py.plot(t_uniform,x_uniform,label='species A (predator)')
#py.plot(t_uniform,x_uniformB,label='species B (prey)')
#py.legend(loc='best')
#py.savefig('Time_evolution.png')




    
        












        



        
