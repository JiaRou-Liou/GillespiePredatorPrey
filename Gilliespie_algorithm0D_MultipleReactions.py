import numpy as np
import matplotlib.pyplot as py
from pathlib import Path
import random
import time
import matplotlib
import math
import time
#from scipy.interpolate import interp1d #插值法


from functools import partial #https://stackoverflow.com/questions/15331726/how-does-functools-partial-do-what-it-does

class reactants:
    def __init__(self,reactant_IDs,number_of_particles):#initialization
        self.reactant_IDs = reactant_IDs
        self.number_of_particles = number_of_particles
        self.data = dict(zip(reactant_IDs, number_of_particles))
    
class reaction_channels:
    def __init__(self, reactions, particle_dict, nonlinear_cf):
        self.reactions = reactions
        self.particle_dict = particle_dict
        self.nonlinear_cf = nonlinear_cf ## Optional nonlinear function
        
        self.propensities = self.compute_all_propensities()
        
        #self.propensities = {}
        #self.update()  # Initialize

    #def update(self):
        #self.propensities = self.compute_all_propensities()

    def compute_all_propensities(self):
        prop_dict = {}
        for idx, reaction in enumerate(self.reactions):
            a = 1
            valid = True
            for i, r in enumerate(reaction['reactants']):
                #print(r)
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
    def __init__(self, reactions, particle_dict):
        self.reactions = reactions
        self.particle_dict = particle_dict

    def execute(self, selected_index):
        reaction = self.reactions[selected_index]

        # Subtract reactants
        for i, r in enumerate(reaction['reactants']):
            self.particle_dict[r] -= reaction['num_r'][i]

        # Add products
        for i, p in enumerate(reaction['products']):
            self.particle_dict[p] += reaction['num_p'][i]

        return self.particle_dict
        
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

    #@staticmethod
    #def prey_birth_with_space(n, m, N, b):
    #    E = N - n - m
    #return 2 * b * (m / N) * E if E > 0 else 0
    

        
def prey_birth_with_space(n, m, N, b):
    E = N - n - m
    return 2 * b * (m / N) * E if E > 0 else 0        
        
        
class record:
    
    def __init__(self,non_uniform_time_steps,dictionary,record_dict):
        self.t_gill = non_uniform_time_steps
        #self.t_uniform = np.linspace(non_uniform_time_steps[0], non_uniform_time_steps[-1], num=len(non_uniform_time_steps))        
        self.dictionary = dictionary
        #self.index = index
        self.record_dict = record_dict
        self.time_evol = self.update()
        
    def update(self):
        
        #self.record_dict['t_gill'][self.index] = self.t_gill
        for keys, values in (self.dictionary.items()):
            self.record_dict[keys].append(values)
        self.record_dict['t_gill'].append(self.t_gill)
        return self.record_dict
    
    @staticmethod
    def uniform_time_grid(non_uniform_time, value):
        t_gill = non_uniform_time
        t_uniform = np.linspace(t_gill[0], t_gill[-1], num=len(t_gill))
        x_uniform = np.interp(t_uniform, t_gill, value)
        return t_uniform, x_uniform
        
            
        
    
        
reactant_IDs = np.array(['A', 'B', 'E'])
V = 3200
b, p1, p2, d1, d2 = 0.1/V, 0.25/V, 0.05/V, 0.1, 0.001

number_of_particles = np.ones(len(reactant_IDs))*V*0.2
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

# Define only "species-specific" nonlinear behavior
nonlinear_cf = {
    'A': partial(NonlinearCoefficients.allee_effect, A=5, h=2),
    'B': partial(NonlinearCoefficients.inhibition, K=80)
}
start = time.time() 



# Manually add special prey birth reaction: m -> m + 1
#n = dictionary['A']
#m = dictionary['B']
#special_rate = prey_birth_with_space(n, m, N=V, b=b)  # define total_population and birth_rate
#propensities['special_birth'] = {'a': special_rate, 'index': 'special'}
runs = 0
times = 1
intervals=1000

many_runs = {}
length_of_time = 250000

#while runs < times:
    # Simulation setup
    
    
total=[]
total.append(V)   
t = 0
values = [[] for i in range(len(reactant_IDs))]

record_dict = dict(zip(reactant_IDs,values))
for key, value in dictionary.items():
    record_dict[key].append(value)
#record_dict['t_gill']=[]
record_dict['t_gill']=[0]
#record_dict['t_gill'].append(t)

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

    dictionary = processes(reactions, dictionary).execute(selected_index)
    
    #channel.update()
    #propensities = channel.propensities

    #if selected_index == 'special':
    # Special case: m -> m + 1 (prey birth)
        #dictionary['B'] += 1
    #else:
        #dictionary = processes(reactions, dictionary).execute(selected_index)
    
    channel = reaction_channels(reactions, dictionary, nonlinear_cf=None)
    propensities = channel.propensities
    # Execute selected reaction
    

    # Manually add special prey birth reaction: m -> m + 1
    #n = dictionary['A']
    #m = dictionary['B']
    #special_rate = prey_birth_with_space(n, m, N=V, b=b)  # define total_population and birth_rate
    #propensities['special_birth'] = {'a': special_rate, 'index': 'special'}

    #if i%intervals==0:

    #T.append(t)
    #record_A.append(dictionary['A']/V)
    #record_B.append(dictionary['B']/V)
    #record_E.append(dictionary['E']/V)
    
    evolution = record(t,dictionary,record_dict).time_evol
    
    total.append(sum(np.sum(dictionary[key]) for key in reactant_IDs))

        
    

    
t_uniform, x_uniform = record.uniform_time_grid(evolution['t_gill'], np.array(evolution['A'])/V)
t_uniform,x_uniformB = record.uniform_time_grid(evolution['t_gill'], np.array(evolution['B'])/V)

#py.plot(evolution['t_gill'],total,'.')




end = time.time()
print(end - start)





freq = np.fft.fftfreq(len(t_uniform),d=t_uniform[1] - t_uniform[0])
FFTA = np.fft.fft(x_uniform).real

py.plot(freq, FFTA, label='A')

py.xlim(-0.05,0.05)

#py.legend()
#py.title('modified predator-prey_FFT')
#py.show()


    
        












        



        
