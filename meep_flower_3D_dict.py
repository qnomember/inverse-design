import meep as mp
from meep.materials import Ag
import numpy as np
import matplotlib.pyplot as plt
import sys
import json
from mpi4py import MPI # Make sure mpi4py is imported

#### test
ft = 1.6616161616161618
eps1 = np.real(Ag.epsilon(ft)[0][0])
eps2 = np.imag(Ag.epsilon(ft)[0][0])
A =  ft**2 * (eps2**2 + (1-eps1)**2)/(1-eps1)
B = ft * eps2/(1-eps1)
susceptibilities = [mp.DrudeSusceptibility(frequency = 1, gamma= B, sigma =  A)]

Ag_self_define = mp.Medium(epsilon= 1 , E_susceptibilities=susceptibilities)
Ag = Ag_self_define
####



def trans_flower(params):
    # unpack params with default values if keys don't exist
    rmin = params.get('rmin', 0.047)      
    rmax = params.get('rmax', 0.16)        
    alpha = params.get('alpha', 15.01)      
    beta = params.get('beta', 0)        
    s = params.get('s', 12.345)          
    a = params.get('a', 0.5)             
    Nf = params.get('Nf', 3)
    fac_decay = params.get('fac_decay', 1e-3)   
    ############################
    rm = (rmax + rmin) / 2
    c = rmax - rm + 1e-16
    thickness = 0.02
    a = 0.5
    Nf = 3  # Number of petals (6-fold symmetry)
    sz = 2
    cell_size = mp.Vector3(a, a, sz)  # 2D simulation (z=0)
    resolution = 100  # pixels per unit length
    boundary_layers = [mp.Absorber(thickness=0.5,direction=mp.Z)]  # Perfectly matched layers
    nfreq = 100  
    ############################
    def is_inside_flower(x, y, z):
        rm = (rmax+rmin)/2
        c = rmax - rm + 1e-16
    
        if z > thickness/2 or z < - thickness/2:
            return True
   
        # Convert to polar coordinates
        r = np.sqrt(x**2 + y**2)
        theta = np.arctan2(y, x)
        if theta < 0:
            theta += 2 * np.pi  # Ensure theta is in [0, 2π]

        # Check if r is within the radial bounds
        if r < rmin or r > rmax:
            return False
         # Compute the angular boundaries for the petal
        rv = r
        theta1 = s * np.sqrt(c**2 - (rv - rm)**2) + alpha * rv + beta * rv**2
        theta2 = -s * np.sqrt(c**2 - (rv - rm)**2) + alpha * rv + beta * rv**2

        # Normalize theta to the first petal and check all petals
        for i in range(Nf):
            dtheta = i * 2 * np.pi / Nf
            # Shift theta to the current petal
            theta_shifted = (theta - dtheta) % (2 * np.pi)
            if theta_shifted < 0:
                theta_shifted += 2 * np.pi

            # Check if theta_shifted is between theta2 and theta1
            # theta2 is the lower bound, theta1 is the upper bound
            if theta2 <= theta_shifted <= theta1:
                return True

        return False

    # Meep material function
    def flower_material(p):
        x, y, z = p.x, p.y, p.z
        if is_inside_flower(x, y, z):
            return mp.Medium(epsilon=1)  # Dielectric constant inside flower
        else:
            return Ag  # Vacuum outside

    ############################
    # Geometry with custom material
    geometry = [
        mp.Block(
            size=cell_size,
            center=mp.Vector3(),
            material=flower_material
        )
    ]
    #############################################
    # Source
    fcen = 1/0.6
    df = 1
    sources_RCP = [
        mp.Source(
            src=mp.GaussianSource(frequency=fcen,fwidth=df),
            component=mp.Ex,
            center=mp.Vector3(0, 0, -0.2),
            size=mp.Vector3(a, a, 0),  # Spans the xy-plane of the unit cell
            amplitude=1
        ),
            mp.Source(
            src=mp.GaussianSource(frequency=fcen,fwidth=df),
            component=mp.Ey,
            center=mp.Vector3(0, 0, -0.2),
            size=mp.Vector3(a, a, 0),  # Spans the xy-plane of the unit cell
            amplitude=1j
        )

    ]
#######################
    sources_LCP = [
        mp.Source(
            src=mp.GaussianSource(frequency=fcen,fwidth=df),
            component=mp.Ex,
            center=mp.Vector3(0, 0, -0.2),
            size=mp.Vector3(a, a, 0),  # Spans the xy-plane of the unit cell
            amplitude=1
        ),
            mp.Source(
            src=mp.GaussianSource(frequency=fcen,fwidth=df),
            component=mp.Ey,
            center=mp.Vector3(0, 0, -0.2),
            size=mp.Vector3(a, a, 0),  # Spans the xy-plane of the unit cell
            amplitude=-1j
        )

    ]


    
    # sim without structure
    sim = mp.Simulation(
        cell_size=cell_size,
        resolution=resolution,
        boundary_layers=boundary_layers,
        sources=sources_RCP,
        k_point=mp.Vector3(0, 0, 0),
        geometry=[]
    )


    # Add flux monitor for  incident
    flux_inci = sim.add_flux(
        fcen,
        df,
        nfreq,
        mp.FluxRegion(
            center=mp.Vector3(0, 0, 0.2),
            size=mp.Vector3(a, a, 0)
        )
    )
    # Run the simulation
    sim.run(until_after_sources=mp.stop_when_fields_decayed(5, mp.Ex, mp.Vector3(), fac_decay))
    # Get the incident flux data
    incident_flux = mp.get_fluxes(flux_inci)
    #############################################
    # simu of metal and RCP
    ############################################
    sim_film = mp.Simulation(
        cell_size=cell_size,
        resolution=resolution,
        boundary_layers=boundary_layers,
        sources=sources_RCP,
        k_point=mp.Vector3(0, 0, 0),
        geometry=geometry,  # Include the film and hole,
        extra_materials = [Ag] # !!!! required when using self-defined material
        )
    
    # Add flux monitor at the same position as in the empty simulation
    flux_trans = sim_film.add_flux(
        fcen,
        df,
        nfreq,
        mp.FluxRegion(
            center=mp.Vector3(0, 0, 0.2),
            size=mp.Vector3(a, a, 0)
        )
    )
    # Run the simulation
    import time
    start_time = time.perf_counter()
    sim_film.run(until_after_sources=mp.stop_when_fields_decayed(5, mp.Ex, mp.Vector3(0,0,-0.2), fac_decay))
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time:.4f} seconds")
    # Get the transmitted flux data
    transmitted_flux = mp.get_fluxes(flux_trans)
    # Compute the transmission spectrum
    transmission = [t / i for t, i in zip(transmitted_flux, incident_flux)]

    # Extract transmission at the center frequency
    flux_freqs = mp.get_flux_freqs(flux_trans)
    index = np.argmin(np.abs(np.array(flux_freqs) - fcen))
    transmission_at_fcen = transmission[index]
    
    ################################################  
    ## LCP run
    sim_film.restart_fields()
    sim_film.clear_dft_monitors()
    sim_film.change_sources([])
    sim_film.change_sources(sources_LCP)
    # Add flux monitor at the same position as in the empty simulation
    flux_trans = sim_film.add_flux(
        fcen,
        df,
        nfreq,
        mp.FluxRegion(
            center=mp.Vector3(0, 0, 0.2),
            size=mp.Vector3(a, a, 0)
        )
    )

    sim_film._evaluate_dft_objects() ## Test: sim.run also do this after each step. This line does eva dft at aeroth step.

    # Run the simulation
    sim_film.run(until_after_sources=mp.stop_when_fields_decayed(5, mp.Ex, mp.Vector3(0,0,-0.2), fac_decay ))
    # Get the transmitted flux data
    transmitted_flux_LCP = mp.get_fluxes(flux_trans)
    # Compute the transmission spectrum
    transmission_LCP = [t / i for t, i in zip(transmitted_flux_LCP, incident_flux)]
    # Extract transmission at the center frequency
    flux_freqs = mp.get_flux_freqs(flux_trans)
    index = np.argmin(np.abs(np.array(flux_freqs) - fcen))
    transmission_at_fcen_LCP = transmission_LCP[index]

    
    # Only rank 0 returns the result to the main process
    if mp.my_rank() == 0:
        return (transmission_at_fcen, transmission_at_fcen_LCP) # return a tuple   
    else:
        return None # Other ranks don't need to return anything to the orchestrator

    
if __name__ == '__main__':
    comm = MPI.COMM_WORLD # Initialize MPI communicator

    params = None # Initialize for all ranks
    if comm.rank == 0: # Only rank 0 reads the file
        if '--params_file' in sys.argv:
            param_file_index = sys.argv.index('--params_file') + 1
            params_filepath = sys.argv[param_file_index]
            with open(params_filepath, 'r') as f:
                params = json.load(f)
        else:
            print(f"Rank {comm.rank}: Error: --params_file argument missing.")
            sys.exit(1)

    # Broadcast parameters to all ranks
    params = comm.bcast(params, root=0)

    # Call the FDTD function on all ranks
    fdtd_results = trans_flower(params) # This will be (t_rcp, t_lcp) on rank 0, None on others

    if comm.rank == 0: # Only rank 0 writes the results to the file
        t_rcp, t_lcp = fdtd_results # Unpack only on rank 0, where it's a tuple
        result_filepath = params['result_filepath']
        with open(result_filepath, 'w') as f:
            json.dump({'t_rcp': t_rcp, 't_lcp': t_lcp}, f)
        print(t_lcp,t_rcp)



# if __name__ == '__main__':
#     # This block is executed when meep_FDTD.py is run directly (or by subprocess)
#     if mp.my_rank() == 0: # Only rank 0 handles file I/O for parameters/results
#         if '--params_file' in sys.argv:
#             param_file_index = sys.argv.index('--params_file') + 1
#             params_filepath = sys.argv[param_file_index]
#             with open(params_filepath, 'r') as f:
#                 params = json.load(f)
#             # You might need to broadcast these parameters to other ranks if not already done by Meep internal setup
#         else:
#             print("Error: --params_file argument missing when running meep_FDTD.py directly.")
#             sys.exit(1)
#     else:
#         params = None # Other ranks wait for broadcast

#     # *** THIS IS THE CRITICAL PART: Broadcast parameters to all ranks ***
#     # Meep might handle some object distribution, but for a simple Python dictionary,
#     # you'll need mpi4py to explicitly broadcast it if all ranks need it.
#     # If your Meep simulation parameters are solely within Meep objects after creation,
#     # Meep will handle their distribution. However, if your Python code
#     # (like trans_flower) needs access to the raw 'params' dictionary
#     # on all ranks before Meep objects are fully defined, you need to broadcast.

#     # Option A: If you are using mpi4py explicitly (recommended for explicit control)
#     try:
#         from mpi4py import MPI
#         comm = MPI.COMM_WORLD
#         params = comm.bcast(params, root=0)
#     except ImportError:
#         # Fallback if mpi4py isn't available or not explicitly used for this
#         if mp.my_rank() != 0:
#             print(f"Rank {mp.my_rank()}: Warning: mpi4py not found. Assuming parameters are not needed on non-root ranks for this part or Meep handles it.")
#         pass # If you're confident Meep handles everything internally

#     # Now, all ranks have the 'params' dictionary
#     t_rcp, t_lcp  = trans_flower(params)

#     if mp.my_rank() == 0:
#         # Save the result to a file for the calling process to read
#         result_filepath = params['result_filepath'] # Get result path from parameters
#         with open(result_filepath, 'w') as f:
#             json.dump({'rcp': t_rcp,'lcp':t_lcp}, f)
#         print(t_lcp,t_rcp)



