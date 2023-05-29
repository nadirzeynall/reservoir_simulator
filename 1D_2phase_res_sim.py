import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve

class OneDTwoPhaseReservoir:
    def __init__(self, nx, dx, porosity, permeability, viscosity, compressibility):
        self.nx = nx
        self.dx = dx
        self.porosity = porosity
        self.permeability = permeability
        self.viscosity = viscosity
        self.compressibility = compressibility
        self.pressure = np.ones(nx) * 5000  # Initial pressure 
        
        self.saturation = np.zeros((nx, 2))
        self.saturation[:, 0] = 0.0  # Initial water saturation
        self.saturation[:, 1] = 1 - self.saturation[:, 0]  # Initial oil saturation
        

    def rel_perm(self, phase, saturation):
        # Check for valid phase
        if phase not in ['water', 'oil']:
            raise ValueError(f"Invalid phase: {phase}")
            
        # Assume linear relative permeability functions for simplicity
        if phase == 'water':
            saturation_water = np.clip(saturation[:, 0], 0, 1)
            return saturation_water
        elif phase == 'oil':
            saturation_oil = np.clip(saturation[:, 1], 0, 1)
            return saturation_oil

    def rel_perm_deriv(self, phase, saturation):
        # Derivative of linear relative permeability is constant.
        # For water, it's 1. For oil, it's -1.
        if phase == 'water':
            return 1
        elif phase == 'oil':
            return -1
        
    def capillary_pressure(self, phase, saturation):
        # Assume zero capillary pressure for simplicity
        return 0
    
    def capillary_pressure_deriv(self, phase, saturation):
        # Derivative of zero capillary pressure is zero.
        return 0

    def transmissibility(self, phase, i, j):
        k_rel = self.rel_perm(phase, self.saturation)
        eps = 1e-10  # Small constant to avoid division by zero
        k_rel_i = k_rel[i] + eps
        k_rel_j = k_rel[j] + eps
        return (2 * self.permeability * k_rel_i * k_rel_j) / (self.viscosity[phase] * self.dx * max(k_rel_i + k_rel_j, eps))

    def accumulation(self, i):
        return self.porosity * self.compressibility

    def jacobian(self):
        n = self.nx
        J = np.zeros((2*n, 2*n))

        J[0, 0] = 1
        J[self.nx, self.nx] = -1
        J[self.nx, self.nx-1] = 1
        
        J[self.nx + 1, self.nx + 1] = -1
        J[-1, -1] = 1

        for i in range(1, n - 1):
            T_oil_left = self.transmissibility('oil', i, i - 1)
            T_water_left = self.transmissibility('water', i, i - 1)
            T_oil_right = self.transmissibility('oil', i, i + 1)
            T_water_right = self.transmissibility('water', i, i + 1)
            B = self.accumulation(i)
            
            # Pressure derivatives
            J[i, i - 1] = -(T_oil_left + T_water_left)
            J[i, i] = T_oil_left + T_water_left + T_oil_right + T_water_right + B
            J[i, i + 1] = -(T_oil_right + T_water_right)

            # Saturation derivatives
            J[n + i, n + i - 1] = -T_water_left
            J[n + i, n + i] = T_water_left + T_water_right
            J[n + i, n + i + 1] = -T_water_right
            
            # Coupling terms
            J[i, n + i] = -(self.transmissibility('oil', i, i - 1) * self.rel_perm_deriv('oil', self.saturation[i, :])
                          + self.transmissibility('water', i, i - 1) * self.rel_perm_deriv('water', self.saturation[i, :]))
            J[n + i, i] = -self.transmissibility('water', i, i - 1) * self.capillary_pressure_deriv('water', self.saturation[i, :])

        return J

    def residual(self):
        n = self.nx
        R = np.zeros(2*n)

        R[0] = self.pressure[0] - 4000  # for left boundary
        R[self.nx] = self.pressure[-1] - self.pressure[-2]  # for right boundary
        
        R[self.nx + 1] = self.saturation[0, 0] - self.saturation[1, 0] # for left boundary
        R[-1] = self.saturation[-1, 0] # for right boundary

        for i in range(1, n - 1):
            T_oil_left = self.transmissibility('oil', i, i - 1)
            T_water_left = self.transmissibility('water', i, i - 1)
            T_oil_right = self.transmissibility('oil', i, i + 1)
            T_water_right = self.transmissibility('water', i, i + 1)

            R[i] += (T_oil_left + T_water_left) * (self.pressure[i - 1] - self.pressure[i])
            R[i] += self.accumulation(i) * self.pressure[i]
            R[i] += (T_oil_right + T_water_right) * (self.pressure[i + 1] - self.pressure[i])

            R[n + i] += T_water_left * (self.pressure[i - 1] - self.pressure[i])
            R[n + i] += T_water_right * (self.pressure[i + 1] - self.pressure[i])

        return R

    def solve(self, time_steps, max_iterations, tolerance, damping_factor=0.5):
        result_dict = dict()
        I = csr_matrix(np.eye(self.nx * 2))  # Identity matrix
        for t in range(time_steps):
            J = csr_matrix(self.jacobian())
            R = self.residual()
            # Regularization
            J += 1e-10 * I

            dX = spsolve(J, -R)

            self.pressure += damping_factor * dX[:self.nx]  # Update pressure
            self.pressure = np.maximum(self.pressure, 0)  # Clip to atmospheric pressure

            # Compute changes to saturation
            delta_saturation = damping_factor * dX[self.nx:]

            # Limit changes to [0, 1]
            delta_saturation = np.clip(delta_saturation, -self.saturation[:, 0], 1 - self.saturation[:, 0])

            self.saturation[:, 0] += damping_factor * dX[self.nx:]  # Update water saturation
            self.saturation[:, 1] = 1 - self.saturation[:, 0]  # Update oil saturation

            self.saturation = np.clip(self.saturation, 0, 1)  # Ensure saturations are within the valid range

            temp_dict = {'Saturation': np.copy(self.saturation), 'Pressure': np.copy(self.pressure)}
            result_dict[f"Time step {t + 1}"] = temp_dict

            # Print the saturation and pressure values for the current time step
            print(f"Time step {t + 1}:")
            print("Water Saturation:")
            print(self.saturation[:, 0])
            print("Oil Saturation:")
            print(self.saturation[:, 1])
            print("Pressure:")
            print(self.pressure)
            print("\n")
            
        return result_dict

reservoir = OneDTwoPhaseReservoir(
    nx=100,
    dx=10,
    porosity=0.2,
    permeability=100,
    viscosity={
        'oil': 3,
        'water': 1
    },
    compressibility=1e-5
)

result = reservoir.solve(time_steps=20, max_iterations=10, tolerance=1e-6, damping_factor=0.01)

sol_pres = list()
sol_wtr = list()
sol_oil = list()

for i in range(reservoir.nx):
    sol_pres.append([[result[key]['Pressure'][:][99-i] for key in result.keys()]])
    sol_wtr.append([[result[key]['Saturation'][:,0][i] for key in result.keys()]])
    sol_oil.append([[result[key]['Saturation'][:,1][i] for key in result.keys()]])
    
sol_pres = np.array(sol_pres)
sol_wtr = np.array(sol_wtr)
sol_oil = np.array(sol_oil)

trans_pres = np.transpose(sol_pres, (2,1,0))
# trans_pres = np.flip(trans_pres, axis=1)
trans_wtr = np.transpose(sol_wtr, (2,1,0))
trans_oil = np.transpose(sol_oil, (2,1,0))

"""""""""""
INTERACTIVE ANIMATION
"""""""""""

from __future__ import print_function
from ipywidgets import interact, interactive, fixed, interact_manual, ToggleButtons
import ipywidgets as widgets
import matplotlib.pyplot as plt 

min_pres, max_pres = np.amin(trans_pres), np.amax(trans_pres)
min_wtr, max_wtr = np.amin(trans_wtr), np.amax(trans_wtr)
min_oil, max_oil = np.amin(trans_oil), np.amax(trans_oil)

@interact
def display_pressure(day=(0, 19)):
    fig, ax = plt.subplots(figsize = (30,10))
    im = ax.imshow(trans_pres[day], vmin=min_pres, vmax=max_pres,aspect=5)
    ax.set_title('Pressure at day {}'.format(day+1))
#     cax = fig.add_axes([1, 0.1, 1, 1])
    fig.colorbar(im, orientation='horizontal')

#     # add values on the grid blocks
#     for (j,i),label in np.ndenumerate(np.round(solution[day], 3)):
#         ax.text(i,j,label,ha='center',va='center')  

    plt.show()
    
@interact
def display_pressure(day=(0, 19)):
    fig, ax = plt.subplots(figsize = (30,10))  
    im = ax.imshow(trans_wtr[day], vmin=min_wtr, vmax=max_wtr,aspect=5)
    ax.set_title('Pressure at day {}'.format(day+1))
#     cax = fig.add_axes([1, 0.1, 1, 1])
    fig.colorbar(im, orientation='horizontal')

#     # add values on the grid blocks
#     for (j,i),label in np.ndenumerate(np.round(solution[day], 3)):
#         ax.text(i,j,label,ha='center',va='center')  

    plt.show()
    
@interact
def display_pressure(day=(0, 19)):
    fig, ax = plt.subplots(figsize = (30,10))  
    im = ax.imshow(trans_oil[day], vmin=min_oil, vmax=max_oil,aspect=5)
    ax.set_title('Pressure at date {}'.format(day+1))
#     cax = fig.add_axes([1, 0.1, 1, 1])
    fig.colorbar(im, orientation='horizontal')

#     # add values on the grid blocks
#     for (j,i),label in np.ndenumerate(np.round(solution[day], 3)):
#         ax.text(i,j,label,ha='center',va='center')  

    plt.show()