#!/usr/bin/env python
# coding: utf-8

# In[1]:part 1 import all we need


import meep as mp
import meep.adjoint as mpa
import numpy as np
from autograd import numpy as npa
from autograd import tensor_jacobian_product, grad
import nlopt
import matplotlib.colors as mcolors
from matplotlib import pyplot as plt
from matplotlib.patches import Circle
from meep.materials import Ag
import os
import math

# In[2]:part 2 create path and source setting


dir_path = "3-D/img/LCPRCP0626_250nm_20nm_grad_1500_f600_tol7_rota"  
os.makedirs(f'{dir_path}/s_change',exist_ok=True) # Create Folder
os.makedirs(f'{dir_path}/final_data',exist_ok=True) # Create Folder
os.makedirs(f'{dir_path}/v_data',exist_ok=True)
mp.verbosity(0)
TiO2 = mp.Medium(index=2.6)
SiO2 = mp.Medium(index=1.44)
Si = mp.Medium(index=3.4)
Air = mp.Medium(index=1)
resolution = 100
design_region_resolution = int(resolution)

design_region_x_width  = 0.25   #250 nm
design_region_y_height = 0.25   #250 nm
design_region_z_height = 0.02  #20  nm

pml_size = 1.0
pml_layers = [mp.PML(pml_size,direction=mp.Z)]

Sz_size = 0.2
Sx = design_region_x_width
Sy = design_region_y_height 
Sz = 2 * pml_size + design_region_z_height + Sz_size
cell_size = mp.Vector3(Sx, Sy, Sz)

wavelengths = np.array([0.6])     # wavelengths = np.array([1.5 ,1.55, 1.6])
frequencies = np.array([1 / 0.6])

nf = 1                 #3 #wavelengths Number

minimum_length = 0.02  # minimum length scale (microns)
eta_i = 0.5            # blueprint (or intermediate) design field thresholding point (between 0 and 1)
eta_e = 0.55           # erosion design field thresholding point (between 0 and 1)
eta_d = 1 - eta_e      # dilation design field thresholding point (between 0 and 1)
filter_radius = mpa.get_conic_radius_from_eta_e(minimum_length, eta_e)

Source_distance = -0.05

fcen   = 1 / 0.6
width  = 0.2  
fwidth = width * fcen
df=fwidth
src_z = -0.05
source_center = mp.Vector3(0, 0, src_z)
source_size   = mp.Vector3(design_region_x_width, design_region_y_height, 0)

lcp_sources = [
    mp.Source(
        src=mp.GaussianSource(frequency=fcen, fwidth=fwidth, is_integrated=True),  
        component=mp.Ex,
        center=source_center,
        size=source_size,
        amplitude=1.0
    ),
    mp.Source(
        src=mp.GaussianSource(frequency=fcen, fwidth=fwidth, is_integrated=True),
        component=mp.Ey,
        center=source_center,
        size=source_size,
        amplitude=1.0j  # (LCP) 
    )
]

rcp_sources = [
    mp.Source(
        src=mp.GaussianSource(frequency=fcen, fwidth=fwidth, is_integrated=True),
        component=mp.Ex,
        center=source_center,
        size=source_size,
        amplitude=1.0
    ),
    mp.Source(
        src=mp.GaussianSource(frequency=fcen, fwidth=fwidth, is_integrated=True),
        component=mp.Ey,
        center=source_center,
        size=source_size,
        amplitude=-1.0j  # (RCP)
    )
]



#sources = lcp_sources + rcp_sources


Nx = int(design_region_resolution * design_region_x_width) + 1
Ny = int(design_region_resolution * design_region_y_height) + 1
Nz = 1

design_variables = mp.MaterialGrid(mp.Vector3(Nx, Ny, Nz), Air, Ag, grid_type="U_MEAN")
design_region    = mpa.DesignRegion(
            design_variables,
            volume=mp.Volume(
            center=mp.Vector3(0,0,0),
            size=mp.Vector3(design_region_x_width, design_region_y_height, design_region_z_height),
            ),
)


def mapping(x, eta, beta):
    # filter
    filtered_field = mpa.conic_filter(
        x,
        filter_radius,
        design_region_x_width,
        design_region_y_height,
        design_region_resolution,
    )
    # projection
    projected_field = mpa.tanh_projection(filtered_field, beta, eta)
   # projected_field = (npa.flipud(projected_field) + projected_field) / 2  # left-right symmetry    
    
    return projected_field.flatten()


geometry = [mp.Block(center=design_region.center, size=design_region.size, material=design_variables)]

kpoint = mp.Vector3()
# LCP sim
sim_LCP = mp.Simulation(
    cell_size        = cell_size,
    boundary_layers  = pml_layers,
    geometry         = geometry,
    sources          = lcp_sources,
    default_material = Air,
    k_point          = kpoint,
   # symmetries       = [mp.Mirror(direction=mp.X)]  
    resolution       = resolution,
    extra_materials  = [Ag],                       
)

# RCP sim
sim_RCP = mp.Simulation(
    cell_size        = cell_size,
    boundary_layers  = pml_layers,
    geometry         = geometry,
    sources          = rcp_sources,
    default_material = Air,
    k_point          = kpoint,
   # symmetries       = [mp.Mirror(direction=mp.X)],
    resolution       = resolution,
    extra_materials  = [Ag],
)


# In[3]:part 3 sim normalize(no structure and define J(poynter vector))


# Incident flux simulation (empty, LCP source)
sim_incident = mp.Simulation(
    cell_size=cell_size,
    boundary_layers=pml_layers,
    geometry=[],  # Empty
    sources=lcp_sources,  # LCP sources (same incident flux for RCP)
    default_material=Air,
    k_point=mp.Vector3(0, 0, 0),
    resolution=resolution,
    extra_materials=[Ag],
)

# Define monitor for incident flux
monitor_position = mp.Vector3(0, 0, 0.03)
monitor_size = mp.Vector3(Sx, Sy, 0)
incident_dft = sim_incident.add_dft_fields(
    [mp.Ex, mp.Ey, mp.Hx, mp.Hy],
    fcen,
    fwidth,
    1,  # Single frequency
    where=mp.Volume(center=monitor_position, size=monitor_size)
)

# Run simulation
sim_incident.run(until_after_sources=mp.stop_when_fields_decayed(50, mp.Ex, mp.Vector3(0, 0, 0), 1e-3))

# Get field components
Ex = sim_incident.get_dft_array(incident_dft, mp.Ex, 0)
Ey = sim_incident.get_dft_array(incident_dft, mp.Ey, 0)
Hx = sim_incident.get_dft_array(incident_dft, mp.Hx, 0)
Hy = sim_incident.get_dft_array(incident_dft, mp.Hy, 0)

# Get metadata for weights
x, y, z, w = sim_incident.get_array_metadata(dft_cell=incident_dft)
w = w.reshape(Ex.shape)  # Reshape weights to match field shape

# Compute Poynting flux
flux_density = 0.5 * npa.real(npa.conj(Ex) * Hy - npa.conj(Ey) * Hx)
#flux_density = 0.5 * npa.real(Ex * npa.conj(Hy) - Ey * npa.conj(Hx))
incident_flux = np.sum(w * flux_density)

# Validate flux
if incident_flux <= 0:
    raise ValueError(f"Invalid incident flux: {incident_flux}. Must be positive.")

# Save incident flux and weights
np.save(f'{dir_path}/v_data/incident_flux_poynting.npy', incident_flux)
np.save(f'{dir_path}/v_data/weights_monitor.npy', w)
print(f"[✓] Incident flux (Poynting): {incident_flux:.4f}")
print(f"[✓] Weights saved to {dir_path}/v_data/weights_monitor.npy")

# Clean up
sim_incident.reset_meep()

# Step 2: Set Up Fourier Fields for LCP and RCP
monitor_position = mp.Vector3(0, 0, 0.03)
monitor_size = mp.Vector3(Sx, Sy, 0)

# LCP Fourier fields
trans_Ex_LCP = mpa.FourierFields(sim_LCP, mp.Volume(center=monitor_position, size=monitor_size), mp.Ex, yee_grid=True)
trans_Ey_LCP = mpa.FourierFields(sim_LCP, mp.Volume(center=monitor_position, size=monitor_size), mp.Ey, yee_grid=True)
trans_Hx_LCP = mpa.FourierFields(sim_LCP, mp.Volume(center=monitor_position, size=monitor_size), mp.Hx, yee_grid=True)
trans_Hy_LCP = mpa.FourierFields(sim_LCP, mp.Volume(center=monitor_position, size=monitor_size), mp.Hy, yee_grid=True)

# RCP Fourier fields
trans_Ex_RCP = mpa.FourierFields(sim_RCP, mp.Volume(center=monitor_position, size=monitor_size), mp.Ex, yee_grid=True)
trans_Ey_RCP = mpa.FourierFields(sim_RCP, mp.Volume(center=monitor_position, size=monitor_size), mp.Ey, yee_grid=True)
trans_Hx_RCP = mpa.FourierFields(sim_RCP, mp.Volume(center=monitor_position, size=monitor_size), mp.Hx, yee_grid=True)
trans_Hy_RCP = mpa.FourierFields(sim_RCP, mp.Volume(center=monitor_position, size=monitor_size), mp.Hy, yee_grid=True)

ob_list_LCP = [trans_Ex_LCP, trans_Ey_LCP, trans_Hx_LCP, trans_Hy_LCP]
ob_list_RCP = [trans_Ex_RCP, trans_Ey_RCP, trans_Hx_RCP, trans_Hy_RCP]

# Step 3: Define J Function
def J(Ex, Ey, Hx, Hy):
    # Extract field components for the single frequency
    Ex0 = Ex[0]
    Ey0 = Ey[0]
    Hx0 = Hx[0]
    Hy0 = Hy[0]
    
    # Trim arrays to the smallest common shape
    min_shape = tuple(npa.min([Ex0.shape, Ey0.shape, Hx0.shape, Hy0.shape], axis=0))
    Ex0 = Ex0[:min_shape[0], :min_shape[1]]
    Ey0 = Ey0[:min_shape[0], :min_shape[1]]
    Hx0 = Hx0[:min_shape[0], :min_shape[1]]
    Hy0 = Hy0[:min_shape[0], :min_shape[1]]
    
    # Compute Poynting flux density
    flux_density = 0.5 * npa.real(npa.conj(Ex0) * Hy0 - npa.conj(Ey0) * Hx0)
    
    # Load and trim weights
    if not os.path.exists(f'{dir_path}/v_data/weights_monitor.npy'):
        raise FileNotFoundError("Weights file not found. Ensure incident flux calculation is run first.")
    w = np.load(f'{dir_path}/v_data/weights_monitor.npy')
    w = w[:min_shape[0], :min_shape[1]]
    
    # Integrate flux
    total_flux = npa.sum(w * flux_density)
    
    # Normalize by incident flux
    if not os.path.exists(f'{dir_path}/v_data/incident_flux_poynting.npy'):
        raise FileNotFoundError("Incident flux file not found. Ensure incident flux calculation is run first.")
    incident_flux = np.load(f'{dir_path}/v_data/incident_flux_poynting.npy')
    transmittance = total_flux / (incident_flux + 1e-15)
    
    return transmittance

# LCP Optimization: Maximize T_LCP
opt_LCP = mpa.OptimizationProblem(
    simulation=sim_LCP,
    objective_functions=[J],  # Use the created J_LCP
    objective_arguments=[trans_Ex_LCP, trans_Ey_LCP, trans_Hx_LCP, trans_Hy_LCP],
    design_regions=[design_region],
    frequencies=frequencies,
    decimation_factor=1,
    maximum_run_time=30,
)

# RCP Optimization: Minimize T_RCP
opt_RCP = mpa.OptimizationProblem(
    simulation=sim_RCP,
    objective_functions=[lambda Ex, Ey, Hx, Hy: -J(Ex, Ey, Hx, Hy)],  # Use the created J_RCP with negation
    objective_arguments=[trans_Ex_RCP, trans_Ey_RCP, trans_Hx_RCP, trans_Hy_RCP],
    design_regions=[design_region],
    frequencies=frequencies,
    decimation_factor=1,
    maximum_run_time=30,
)

# In[4]:part 4 Check the cross-sectional plane to confirm whether the design region, source, and structure are correct


#-----------------[4]---------------------------------#

def Plot_pre_fun(figsize, title, output_plane, path, Plot_save):
    plt.figure(figsize=figsize)
    plt.title(title)
    opt_RCP.plot2D(True, output_plane=output_plane)
    if Plot_save:
        plt.savefig(path)
        plt.close()
    else:
        plt.show()

Plot_pre_fun(figsize      = (5, 10), 
             title        = 'Detector_x_y', 
             output_plane = mp.Volume(center=monitor_position, size=mp.Vector3(Sx,Sy,0)), 
             path         = f'{dir_path}/Pre_Detector_x_y.png', 
             Plot_save    = 1 )

Plot_pre_fun(figsize      = (5, 10), 
             title        = 'Source', 
             output_plane = mp.Volume(center=source_center, size=mp.Vector3(Sx,Sy,0)), 
             path         = f'{dir_path}/Pre_Source_x_y.png', 
             Plot_save    = 1 )

Plot_pre_fun(figsize      = (5, 10), 
             title        = 'Structure', 
             output_plane = mp.Volume(center=design_region.center, size=mp.Vector3(Sx,Sy,0)), 
             path         = f'{dir_path}/Pre_Structure_x_y.png', 
             Plot_save    = 1 )

Plot_pre_fun(figsize      = (5, 10), 
             title        = 'Sx_Sz', 
             output_plane = mp.Volume(center=mp.Vector3(0,0,0), size=mp.Vector3(Sx,0,Sz)), 
             path         = f'{dir_path}/Pre_Sx_Sz.png', 
             Plot_save    = 1 )

Plot_pre_fun(figsize      = (5, 10), 
             title        = 'Sy_Sz', 
             output_plane = mp.Volume(center=mp.Vector3(0,0,0), size=mp.Vector3(0,Sy,Sz)), 
             path         = f'{dir_path}/Pre_Sy_Sz_d.png', 
             Plot_save    = 1 )

#-----------In[5]:part 5 Resolve gradient issues-------------------------------------------------------------------#




evaluation_history = []
cur_iter = [0]

def f(v, gradient, beta):
    print(f"Current iteration: {cur_iter[0] + 1}")
    
    # Map design variables to structure
    x_avg = mapping(v, eta_i, beta)
    
    # Simulate LCP and RCP
    f0_L, dJ_du_L = opt_LCP([x_avg])  # T_LCP
    f0_R, dJ_du_R = opt_RCP([x_avg])  # -T_RCP
    
    # Compute FOM (T_LCP - T_RCP)
    T_LCP = f0_L
    T_RCP = -f0_R  # Convert back to positive transmittance
    f0 = T_LCP - T_RCP  # Circular dichroism
    
    # Apply PCGrad to resolve gradient conflicts
    dJ_du_L = np.array(dJ_du_L)
    dJ_du_R = np.array(dJ_du_R)
    
    # Combine adjusted gradients for the FOM
    dJ_du = (dJ_du_L + dJ_du_R) * 1500  # Average the adjusted gradients
    
    print(f"LCP Transmittance: {T_LCP:.6f}")
    print(f"RCP Transmittance: {T_RCP:.6f}")
    print(f"FOM (CD): {f0:.6f}")
    
    
    # Backpropagate the adjusted gradient
    if gradient.size > 0:
        gradient[:] = tensor_jacobian_product(mapping, 0)(v, eta_i, beta, dJ_du)
    
    evaluation_history.append(np.real(f0))
    
    # Plot current structure
    if mp.am_master():
        plt.figure(figsize=(5, 5))
        opt_LCP.plot2D(
            plot_sources_flag=False,
            plot_monitors_flag=False,
            plot_boundaries_flag=False,
            output_plane=mp.Volume(
                center=mp.Vector3(0, 0, 0),
                size=mp.Vector3(design_region_x_width, design_region_y_height, 0)
            )
        )
        plt.title(f"Structure (Iteration {cur_iter[0] + 1})")
        plt.savefig(f'{dir_path}/s_change/s_change{cur_iter[0]:03d}.png')
        plt.close()
    
    # Save data
    np.save(f'{dir_path}/v_data/Post_v_array{cur_iter[0]:03d}.npy', v)
    np.save(f'{dir_path}/v_data/Post_x_array{cur_iter[0]:03d}.npy', x_avg)
    np.save(f'{dir_path}/v_data/Post_eta_i_array{cur_iter[0]:03d}.npy', eta_i)
    np.save(f'{dir_path}/v_data/Post_cur_beta_array{cur_iter[0]:03d}.npy', cur_beta)
    np.save(f'{dir_path}/v_data/Post_beta_scale_array{cur_iter[0]:03d}.npy', beta_scale)
    np.save(f'{dir_path}/v_data/T_LCP_array{cur_iter[0]:03d}.npy', T_LCP)
    np.save(f'{dir_path}/v_data/T_RCP_array{cur_iter[0]:03d}.npy', T_RCP)
    np.save(f'{dir_path}/v_data/Post_f0_L_array{cur_iter[0]:03d}.npy', f0_L)
    np.save(f'{dir_path}/v_data/Post_f0_R_array{cur_iter[0]:03d}.npy', f0_R)
    np.save(f'{dir_path}/v_data/Post_FOM_array{cur_iter[0]:03d}.npy', f0)
    np.save(f'{dir_path}/v_data/Post_grad_total_array{cur_iter[0]:03d}.npy', dJ_du)
    np.save(f'{dir_path}/v_data/Post_grad_LCP_array{cur_iter[0]:03d}.npy', dJ_du_L)
    np.save(f'{dir_path}/v_data/Post_grad_RCP_array{cur_iter[0]:03d}.npy', dJ_du_R)
    cur_iter[0] += 1
    return np.real(f0)



#-------In[6]  part6 initial setting and algorithm setting---------------------------------------#
algorithm = nlopt.LD_MMA
n = Nx * Ny * Nz  # number of parameters

# Initial guess
x = np.zeros((n,))
for i in range(Nx):
    for j in range(Ny):
        idx = i * Ny + j
        r = np.sqrt((i / Nx - 0.5)**2 + (j / Ny - 0.5)**2)
        theta = np.arctan2(j / Ny - 0.5, i / Nx - 0.5)
        x[idx] = 0.5 + 0.4 * np.sin(6 * theta) * np.exp(-r**2 / 0.2)
x = np.clip(x, 0, 1)


# lower and upper bounds
lb = np.zeros((n,))
ub = np.ones((n,))



cur_beta = 2
beta_scale = 2
num_betas = 16
update_factor = 16
ftol = 1e-7

for iters in range(num_betas):
    solver = nlopt.opt(algorithm, n)
    solver.set_lower_bounds(lb)
    solver.set_upper_bounds(ub)
    solver.set_max_objective(lambda a, g: f(a, g, cur_beta))
    solver.set_maxeval(update_factor)
    solver.set_ftol_rel(ftol)
    x[:] = solver.optimize(x)
    cur_beta *= beta_scale

#---------------------------------------------------------------------------------------------------#



# In[7]:part7 save iteration part


np.save(f'{dir_path}/final_data/Post_evaluation_history.npy', evaluation_history)
np.save(f'{dir_path}/final_data/Post_x_array.npy', x)
np.save(f'{dir_path}/final_data/Post_eta_i_array.npy', eta_i)
np.save(f'{dir_path}/final_data/Post_cur_beta_array.npy', cur_beta)
np.save(f'{dir_path}/final_data/Post_beta_scale_array.npy', beta_scale)
plt.figure()
plt.plot(evaluation_history, "o-")
Plot_save = 1
print(evaluation_history)
for i in range(1, 16):
    plt.axvline(x=16 * i, color='purple', linestyle='--')

plt.grid(True)
plt.xlim(0)
plt.ylim(0)
plt.xlabel("Iteration")
plt.ylabel("FOM")
if Plot_save:
    plt.savefig(f'{dir_path}/Post_Fom_change.png')
    plt.close()
else:
    plt.show()

#---------------------------------------------------------------
# In[8]:part 8 loading final stucture and test transmission


opt_LCP.update_design([mapping(x, eta_i, cur_beta / beta_scale)])
plt.figure()
ax = plt.gca()
opt_LCP.plot2D(
    False,
    ax=ax,
    plot_sources_flag=False,
    plot_monitors_flag=False,
    plot_boundaries_flag=False,
    output_plane=mp.Volume(center=mp.Vector3(0, 0, 0.005), size=mp.Vector3(design_region_x_width, design_region_y_height, 0))
)
circ = Circle((2, 2), minimum_length / 2)
ax.add_patch(circ)
ax.axis("off")
plt.savefig(f'{dir_path}/Post_Structure_x_y.png')
plt.close()


# In[]:

opt_LCP.sim = mp.Simulation(
    cell_size=cell_size,  
    boundary_layers=pml_layers,
    geometry=geometry,
    sources=lcp_sources,
    default_material=Air,
    resolution=resolution,
    k_point=mp.Vector3(0, 0, 0),
    extra_materials=[Ag],
)



#-------------------------------after here is old test version-------------------------------------this is continuos source part

# Step 1: Calculate incident flux using a ContinuousSource
fcen = 1 / 0.6  # Center frequency (from wavelength 0.7 µm)
fwidth = 0.2 * fcen  # Frequency width
src_z = -0.05
source_center = mp.Vector3(0, 0, src_z)
source_size = mp.Vector3(design_region_x_width, design_region_y_height, 0)

# Define the continuous source for incident flux calculation
src = mp.ContinuousSource(frequency=fcen, fwidth=fwidth, is_integrated=True)

# LCP sources for incident flux
lcp_sources_continuous = [
    mp.Source(
        src=src,
        component=mp.Ex,
        center=source_center,
        size=source_size,
        amplitude=1.0
    ),
    mp.Source(
        src=src,
        component=mp.Ey,
        center=source_center,
        size=source_size,
        amplitude=1.0j  # +90° phase for LCP
    )
]

# Set up the incident flux simulation
sim_incident = mp.Simulation(
    cell_size=cell_size,
    boundary_layers=pml_layers,
    geometry=[],  # No geometry for incident flux
    sources=lcp_sources_continuous,
    default_material=Air,
    k_point=mp.Vector3(0, 0, 0),
    resolution=resolution,
    extra_materials=[Ag],
)

 # Place the flux monitor just above the source to capture incident flux
incident_monitor_position = mp.Vector3(0, 0, 0.03)  # At z = -0.04
incident_monitor_size = mp.Vector3(Sx, Sy, 0)
incident_flux_monitor = sim_incident.add_flux(fcen, df, 1, mp.FluxRegion(center=incident_monitor_position, size=incident_monitor_size))

# Run the simulation for incident flux
sim_incident.run(until=30)
incident_flux = mp.get_fluxes(incident_flux_monitor)[0]
print(f"[✓] Incident flux: {incident_flux:.4f}")

# Save the incident flux
np.save(f'{dir_path}/v_data/incident_flux_continuous.npy', incident_flux)

# Clean up
sim_incident.reset_meep()

# Step 2: Define RCP continuous sources
rcp_sources_continuous = [
    mp.Source(
        src=src,
        component=mp.Ex,
        center=source_center,
        size=source_size,
        amplitude=1.0
    ),
    mp.Source(
        src=src,
        component=mp.Ey,
        center=source_center,
        size=source_size,
        amplitude=-1.0j  # -90° phase for RCP
    )
]

# Step 3: Set up the simulation with LCP sources
opt_LCP.sim = mp.Simulation(
    cell_size=cell_size,
    boundary_layers=pml_layers,
    geometry=geometry,
    sources=lcp_sources_continuous,
    default_material=Air,
    resolution=resolution,
    k_point=mp.Vector3(0, 0, 0),
    extra_materials=[Ag],
)

# Add a flux monitor to measure LCP transmission
monitor_position = mp.Vector3(0, 0, 0.03)
monitor_size = mp.Vector3(Sx, Sy, 0)
lcp_flux_monitor = opt_LCP.sim.add_flux(fcen, df, 1, mp.FluxRegion(center=monitor_position, size=monitor_size))

# Run the simulation to calculate LCP flux
opt_LCP.sim.run(until=30)
lcp_flux = mp.get_fluxes(lcp_flux_monitor)[0]

# Calculate LCP transmission
T_LCP = lcp_flux / incident_flux
print(f"LCP Transmission: {T_LCP:.4f}")

# Step 4: Switch to RCP sources
opt_LCP.sim.reset_meep()
opt_LCP.sim.change_sources(rcp_sources_continuous)

# Add a flux monitor for RCP transmission
rcp_flux_monitor = opt_LCP.sim.add_flux(fcen, df, 1, mp.FluxRegion(center=monitor_position, size=monitor_size))

# Run the simulation to calculate RCP flux
opt_LCP.sim.run(until=30)
rcp_flux = mp.get_fluxes(rcp_flux_monitor)[0]

# Calculate RCP transmission
T_RCP = rcp_flux / incident_flux
print(f"RCP Transmission: {T_RCP:.4f}")

# Step 5: Calculate the transmission difference
transmission_diff = T_LCP - T_RCP
print(f"Transmission Difference (T_LCP - T_RCP): {transmission_diff:.4f}")

# Step 6: Save the results
np.save(f'{dir_path}/final_data/Post_T_LCP_final_continuous1.npy', T_LCP)
np.save(f'{dir_path}/final_data/Post_T_RCP_final_continuous1.npy', T_RCP)
np.save(f'{dir_path}/final_data/Post_transmission_diff_continuous1.npy', transmission_diff)

# Step 7: Generate bar chart
# Data for the bar chart
labels = ['T_LCP', 'T_RCP', 'T_LCP - T_RCP']
values = [T_LCP, T_RCP, transmission_diff]
colors = ['blue', 'orange', 'green']

# Create the bar chart
plt.figure(figsize=(8, 6))
bars = plt.bar(labels, values, color=colors, edgecolor='black')

# Add value labels on top of each bar
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.02, f'{yval:.4f}', ha='center', va='bottom')

# Customize the plot
plt.title('Transmission Comparison: LCP vs RCP', fontsize=14, pad=15)
plt.ylabel('Transmission', fontsize=12)
plt.xlabel('Parameter', fontsize=12)
plt.ylim(0, max(values) + 0.1)  # Adjust y-axis limit
plt.grid(True, axis='y', linestyle='--', alpha=0.7)

# Save the plot
plt.savefig(f'{dir_path}/transmission_bar_chart.png', dpi=300, bbox_inches='tight')
plt.close()

# Clean up
opt_LCP.sim.reset_meep()