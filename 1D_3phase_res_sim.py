class OneDThreePhaseReservoir:
    def __init__(self, nx, dx, porosity, permeability, viscosity, compressibility):
        self.nx = nx
        self.dx = dx
        self.porosity = porosity
        self.permeability = permeability
        self.viscosity = viscosity
        self.compressibility = compressibility
        self.pressure = np.ones(nx) * 5000  # Initial pressure 
        self.saturation = np.zeros((nx, 3))
        self.saturation[:, 0] = 0.7  # Initial water saturation
        self.saturation[:, 1] = 0.2  # Initial oil saturation
        self.saturation[:, 2] = 1 - self.saturation[:, 0] - self.saturation[:, 1]  # Initial gas saturation
        self.gas_compressibility = 1e-4  # Specific to gas, higher than for oil or water
        self.oil_gas_solubility = 0.01  # Fraction of gas that dissolves in oil
        
    def rel_perm(self, phase, saturation):
        if phase not in ['water', 'oil', 'gas']:
            raise ValueError(f"Invalid phase: {phase}")
        if phase == 'water':
            saturation_water = np.clip(saturation[:, 0], 0, 1)
            return saturation_water
        elif phase == 'oil':
            saturation_oil = np.clip(saturation[:, 1], 0, 1)
            return saturation_oil
        elif phase == 'gas':
            saturation_gas = np.clip(saturation[:, 2], 0, 1)
            return saturation_gas

    def capillary_pressure(self, phase, saturation):
        return 0

    def transmissibility(self, phase, i, j):
        k_rel = self.rel_perm(phase, self.saturation)
        eps = 1e-10
        k_rel_i = k_rel[i] + eps
        k_rel_j = k_rel[j] + eps
        return (2 * self.permeability * k_rel_i * k_rel_j) / (self.viscosity[phase] * self.dx * max(k_rel_i + k_rel_j, eps))

    def accumulation(self, phase):
        if phase == 'gas':
            return self.porosity * self.gas_compressibility
        else:
            return self.porosity * self.compressibility

    def jacobian(self):
        n = self.nx
        J = np.zeros((3*n, 3*n))
        
        J[0, 0] = 1
        J[self.nx, self.nx] = 1
        J[2*self.nx, 2*self.nx] = 1
        J[self.nx + 1, self.nx + 1] = 1
        J[2*self.nx + 1, 2*self.nx + 1] = 1
        J[-1, -1] = 1

#         # Left boundary (Dirichlet for Pressure and Saturation)
#         J[0, 0] = 1
#         J[self.nx, self.nx] = 1
#         J[2*self.nx, 2*self.nx] = 1

#         # Right boundary (Neumann for Pressure and Saturation)
#         J[n - 1, n - 1] = 1
#         J[n - 1, n - 2] = -1
#         J[2*n - 1, 2*n - 1] = 1
#         J[2*n - 1, 2*n - 2] = -1

        for i in range(1, n - 1):
            T_oil_left = self.transmissibility('oil', i, i - 1)
            T_water_left = self.transmissibility('water', i, i - 1)
            T_gas_left = self.transmissibility('gas', i, i - 1)
            T_oil_right = self.transmissibility('oil', i, i + 1)
            T_water_right = self.transmissibility('water', i, i + 1)
            T_gas_right = self.transmissibility('gas', i, i + 1)
            B = self.accumulation(i)
            
            # Pressure derivatives
            J[i, i - 1] = -(T_oil_left + T_water_left + T_gas_left)
            J[i, i] = T_oil_left + T_water_left + T_gas_left + T_oil_right + T_water_right + T_gas_right + B
            J[i, i + 1] = -(T_oil_right + T_water_right + T_gas_right)

            # Saturation derivatives
            J[n + i, n + i - 1] = -T_water_left
            J[n + i, n + i] = T_water_left + T_water_right
            J[n + i, n + i + 1] = -T_water_right
            J[2*n + i, 2*n + i - 1] = -T_gas_left
            J[2*n + i, 2*n + i] = T_gas_left + T_gas_right
            J[2*n + i, 2*n + i + 1] = -T_gas_right

        return J

    def residual(self):
        n = self.nx
        R = np.zeros(3*n)
        
        
        R[0] = self.pressure[0] - 4500  # left boundary
        R[self.nx] = self.pressure[-1] - self.pressure[-2]  # right boundary
        R[2*self.nx] = self.saturation[1, 0] - self.saturation[0, 0]  # left boundary
        R[-1] = self.saturation[-1, 0] - self.saturation[-2, 0]  # right boundary
        
#         # Left boundary (Dirichlet for Pressure and Saturation)
#         R[0] = self.pressure[0] - 4500
#         R[n] = self.saturation[0, 0] - 0.2  # Assuming initial water saturation
#         R[2*n] = self.saturation[0, 2] - 0.1  # Assuming initial gas saturation

#         # Right boundary (Neumann for Pressure and Saturation)
#         R[n - 1] = self.pressure[n - 1] - self.pressure[n - 2] 
#         R[2*n - 1] = self.saturation[n - 1, 0] - self.saturation[n - 2, 0]
#         R[3*n - 1] = self.saturation[n - 1, 2] - self.saturation[n - 2, 2]

        for i in range(1, n - 1):
            T_oil_left = self.transmissibility('oil', i, i - 1)
            T_water_left = self.transmissibility('water', i, i - 1)
            T_gas_left = self.transmissibility('gas', i, i - 1)
            T_oil_right = self.transmissibility('oil', i, i + 1)
            T_water_right = self.transmissibility('water', i, i + 1)
            T_gas_right = self.transmissibility('gas', i, i + 1)

            R[i] += (T_oil_left + T_water_left + T_gas_left) * (self.pressure[i - 1] - self.pressure[i])
            R[i] += self.accumulation(i) * self.pressure[i]
            R[i] += (T_oil_right + T_water_right + T_gas_right) * (self.pressure[i + 1] - self.pressure[i])

            R[n + i] += T_water_left * (self.pressure[i - 1] - self.pressure[i])
            R[n + i] += T_water_right * (self.pressure[i + 1] - self.pressure[i])

            R[2*n + i] += T_gas_left * (self.pressure[i - 1] - self.pressure[i])
            R[2*n + i] += T_gas_right * (self.pressure[i + 1] - self.pressure[i])
                    
        return R
    
    def solve(self, time_steps, dt, max_iterations, tolerance, damping_factor=0.5):
        result_dict = dict()
        I = csr_matrix(np.eye(self.nx * 3))  # Identity matrix
        for t in range(time_steps):
            for iteration in range(max_iterations):
                J = csr_matrix(self.jacobian())
                R = self.residual()
                
                # Regularization
                J += 1e-10 * I

                dX = spsolve(J, -R)

                if np.linalg.norm(dX, np.inf) < tolerance:
                    break

            self.pressure += damping_factor * dX[:self.nx]  # Update pressure
            self.pressure = np.maximum(self.pressure, 0)  # Clip to atmospheric pressure

            # Compute changes to saturation
            delta_saturation_water = damping_factor * dX[self.nx: 2 * self.nx]
            delta_saturation_gas = damping_factor * dX[2 * self.nx:]

            # Limit changes to [0, 1]
            delta_saturation_water = np.clip(delta_saturation_water, -self.saturation[:, 0], 1 - self.saturation[:, 0])
            delta_saturation_gas = np.clip(delta_saturation_gas, -self.saturation[:, 1], 1 - self.saturation[:, 1] - self.saturation[:, 0])

            # Apply changes
            self.saturation[:, 0] += delta_saturation_water
            self.saturation[:, 1] += delta_saturation_gas
            self.saturation[:, 2] = 1 - self.saturation[:, 0] - self.saturation[:, 1]

            # Account for gas solubility in oil
            average_pressure = sum(self.pressure) / len(self.pressure)
            
            if average_pressure < 3500:
                dissolved_gas = self.saturation[:, 1] * self.oil_gas_solubility
                self.saturation[:, 1] -= dissolved_gas
                self.saturation[:, 2] += dissolved_gas
                self.saturation[:, 2] = np.minimum(self.saturation[:, 2], 1)

            temp_dict = {'Saturation': np.copy(self.saturation), 'Pressure': np.copy(self.pressure)}
            result_dict[f"Time step {t + 1}"] = temp_dict

            # Print the saturation and pressure values for the current time step
            print(f"Time step {t + 1}:")
            print("Water Saturation:")
            print(self.saturation[:, 0])
            print("Oil Saturation:")
            print(self.saturation[:, 1])
            print("Gas Saturation:")
            print(self.saturation[:, 2])
            print("Pressure:")
            print(self.pressure)
            print("\n")
            
        return result_dict


reservoir = OneDThreePhaseReservoir(
    nx=100,
    dx=10,
    porosity=0.2,
    permeability=200,
    viscosity={
        'oil': 3,
        'water': 1,
        'gas': 0.1
    },
    compressibility=1e-5
)

result = reservoir.solve(time_steps=20, dt=1e-3, max_iterations=10, tolerance=1e-5, damping_factor=0.1)

sol_pres = list()
sol_wtr = list()
sol_oil = list()
sol_gas = list()

for i in range(reservoir.nx):
    sol_pres.append([[result[key]['Pressure'][:][99-i] for key in result.keys()]])
    sol_wtr.append([[result[key]['Saturation'][:,0][i] for key in result.keys()]])
    sol_oil.append([[result[key]['Saturation'][:,1][i] for key in result.keys()]])
    sol_gas.append([[result[key]['Saturation'][:,2][i] for key in result.keys()]])
    
    
sol_pres = np.array(sol_pres)
sol_wtr = np.array(sol_wtr)
sol_oil = np.array(sol_oil)
sol_gas = np.array(sol_gas)


trans_pres = np.transpose(sol_pres, (2,1,0))
trans_wtr = np.transpose(sol_wtr, (2,1,0))
trans_oil = np.transpose(sol_oil, (2,1,0))
trans_gas = np.transpose(sol_gas, (2,1,0))


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
min_gas, max_gas = np.amin(trans_gas), np.amax(trans_gas)


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
    
@interact
def display_pressure(day=(0, 19)):
    fig, ax = plt.subplots(figsize = (30,10))  
    im = ax.imshow(trans_gas[day], vmin=min_gas, vmax=max_gas,aspect=5)
    ax.set_title('Pressure at date {}'.format(day+1))
#     cax = fig.add_axes([1, 0.1, 1, 1])
    fig.colorbar(im, orientation='horizontal')

#     # add values on the grid blocks
#     for (j,i),label in np.ndenumerate(np.round(solution[day], 3)):
#         ax.text(i,j,label,ha='center',va='center')  

    plt.show()