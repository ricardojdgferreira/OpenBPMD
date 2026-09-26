#!/usr/bin/env python
"""
OpenUNFOLD - an open source implementation of Thermal protein Unfolding
Molecular Dynamics (TUMD) with OpenMM.

Version 1.0.2

"""

# general
import os
import warnings
import argparse
from collections import defaultdict

# OpenMM
import openmm as mm
from openmm import app
from openmm import unit

# Analysis
import numpy as np
import mdtraj as md
import pandas as pd

# plotting
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit, OptimizeWarning

__author1__ = "Ricardo J. Ferreira"
__version__ = "1.0.2"
__email1__ = "ricardo.ferreira@rgdiscovery.com"

# supress warning
warnings.filterwarnings("ignore")

# def sigmoidal fitting
def sigmoid(T, Qf, Qu, Tm, k):
    '''define sigmoidal fitting'''
    return Qu + (Qf - Qu)/(1 + np.exp((T - Tm)/k))

# Monitoring function
def compute_observables(xyz, top_md, native, ca,pairs, native_dist):
    '''compute all necessary values for several paremeters'''
    traj = md.Trajectory(xyz[np.newaxis, :, :], top_md)
    rmsd = md.rmsd(traj, native, atom_indices=ca)[0]
    rg = md.compute_rg(traj)[0]
    current_dist = md.compute_distances(traj, pairs)[0]
    q = np.mean(current_dist < 1.2 * native_dist)

    return rmsd, rg, q

# check convergence
def converged(series, tol, window):
    ''' to check convergence in all parameters'''
    if len(series) < 2 * window:
        return False

    old_mean = np.mean(series[-2*window:-window])
    new_mean = np.mean(series[-window:])

    return abs(new_mean - old_mean) < tol

# sigmoidal fitting
def fit_current_tm(mean_q):
    ''' fit the q_mean values to a sigmoidal curve'''
    T_fit = np.array(sorted(mean_q.keys()))

    Q_fit = np.array([
        mean_q[T]
        for T in T_fit
    ])

    try:
        popt, _ = curve_fit(
            sigmoid,
            T_fit,
            Q_fit,
            p0=[
                max(Q_fit),
                min(Q_fit),
                np.median(T_fit),
                10.0
            ],
            maxfev=10000
        )

        return popt[2]  # Tm

    except (RuntimeError, ValueError, OptimizeWarning):
        return None

def main(args):
    '''define the main function'''
    print("\n###########################################################")
    print("                                                           ")
    print(" OpenUNFold  - Thermal Unfolding with OpenMM               ")
    print("                                                           ")
    print("###########################################################")

    if args.structure.endswith('.gro'):
        coords = app.GromacsGroFile(args.structure)
        box_vectors = coords.getPeriodicBoxVectors()
        parm = app.GromacsTopFile(args.parameters,
                                  periodicBoxVectors=box_vectors)
    else:
        coords = app.AmberInpcrdFile(args.structure)
        parm = app.AmberPrmtopFile(args.parameters)

    if not os.path.isdir(f'{args.output}'):
        os.mkdir(f'{args.output}')

    # Minimize
    min_file_name = 'minimized_system.pdb'
    if not os.path.isfile(os.path.join(args.output,min_file_name)):
        print("\nMinimizing...")
        minimize(parm, coords.positions, args.output,
                 min_file_name, args.cuda)
    min_pdb = os.path.join(args.output,min_file_name)

    # Equilibrate
    eq_file_name = 'equil_system.pdb'
    if not os.path.isfile(os.path.join(args.output,eq_file_name)):
        print("Equilibrating...")
        equilibrate(min_pdb, parm, args.output, eq_file_name, args.cuda)

    # mdtraj can't use GMX TOP, so we have to specify the GRO file instead
    if args.structure.endswith('.gro'):
        mdtraj_top = args.structure
    else:
        mdtraj_top = args.parameters

    eq_pdb = os.path.join(args.output,eq_file_name)
    cent_eq_pdb = os.path.join(args.output,'centred_'+eq_file_name)
    if os.path.isfile(eq_pdb) and not os.path.isfile(cent_eq_pdb):

        mdu = md.load(eq_pdb, top=mdtraj_top)
        mdu.image_molecules()
        mdu.save_pdb(cent_eq_pdb)

    # Run N number of production simulations between temperature intervals
    produce(eq_pdb, mdtraj_top, parm, args.output, args.cuda)

    return None

def minimize(parm, input_positions, out_dir, min_file_name, cuda):
    '''initial energy minimization'''
    system = parm.createSystem(nonbondedMethod=app.PME,
                               nonbondedCutoff=1.2*unit.nanometer,
                               constraints=app.HBonds,)

    # Define platform properties
    platform = mm.Platform.getPlatformByName('CUDA')
    properties = {'DeviceIndex': f'{cuda}', 'CudaPrecision': 'mixed'}

    # Set up the simulation parameters
    # Langevin integrator at 300 K w/ 1 ps^-1 friction coefficient
    # and a 2-fs timestep
    # NOTE - no dynamics performed, but required for setting up
    # the OpenMM system.
    integrator = mm.LangevinMiddleIntegrator(300*unit.kelvin,
                                             1/unit.picosecond,
                                             0.002*unit.picoseconds)
    simulation = app.Simulation(parm.topology, system, integrator,
                                platform, properties)
    simulation.context.setPositions(input_positions)

    # Minimize the system - no predefined number of steps
    simulation.minimizeEnergy()

    # Write out the minimized system to use w/ MDAnalysis
    positions = simulation.context.getState(getPositions=True).getPositions()
    out_file = os.path.join(out_dir,min_file_name)
    app.PDBFile.writeFile(simulation.topology, positions, 
                          open(out_file, 'w', encoding="utf-8"))

    return None


def equilibrate(min_pdb, parm, out_dir, eq_file_name, cuda):
    '''system equilibration after minimization'''
    # create system
    system = parm.createSystem(nonbondedMethod=app.PME,
                               nonbondedCutoff=1.2*unit.nanometers,
                               constraints=app.HBonds,)

    # Add the restraints on the positions of specified atoms
    restraint = mm.CustomExternalForce('k*periodicdistance(x, y, z, x0, y0, z0)^2')
    system.addForce(restraint)
    restraint.addGlobalParameter('k', 100.0*unit.kilojoules_per_mole/unit.nanometer)
    restraint.addPerParticleParameter('x0')
    restraint.addPerParticleParameter('y0')
    restraint.addPerParticleParameter('z0')

    input_positions = app.PDBFile(min_pdb).getPositions()
    positions = input_positions

    # Go through the indices of all heavy atoms and apply restraints
    protein_resnames = {"ALA","ARG","ASN","ASP","CYS","GLN","GLU","GLY",
                        "HIS","ILE","LEU","LYS","MET","PHE","PRO","SER",
                        "THR","TRP","TYR","VAL"}
    pdb = app.PDBFile(min_pdb)
    for atom in pdb.topology.atoms():
        if atom.residue.name in protein_resnames and atom.element.symbol != "H":
            restraint.addParticle(atom.index, pdb.positions[atom.index])

    integrator = mm.LangevinMiddleIntegrator(300*unit.kelvin,
                                        1/unit.picosecond,
                                        0.002*unit.picoseconds)
    platform = mm.Platform.getPlatformByName('CUDA')
    properties = {'DeviceIndex': f'{cuda}', 'CudaPrecision': 'mixed'}

    sim = app.Simulation(parm.topology, system, integrator,
                         platform, properties)
    sim.context.setPositions(input_positions)
    sim.step(25000)  # run 50 ps of equilibration

    # Write out the equilibrated system to use w/ MDAnalysis
    positions = sim.context.getState(getPositions=True,
                                     enforcePeriodicBox=True).getPositions()
    out_file = os.path.join(out_dir, eq_file_name)
    app.PDBFile.writeFile(sim.topology, positions,
                          open(out_file, 'w', encoding="utf-8"))

    return None

def produce(eq_pdb, mdtraj_top, parm, out_dir, cuda):
    '''production runs for several temperatures'''
    # Get calpha indices
    native = md.load(eq_pdb, top=mdtraj_top)
    top_md = native.topology
    ca = top_md.select("protein and name CA")

    # Native contacts definition
    pairs = []
    for i in range(len(ca)):
        for j in range(i + 4, len(ca)):
            d = np.linalg.norm(native.xyz[0, ca[i]] - native.xyz[0, ca[j]])
            if d < 0.8:      # nm
                pairs.append((ca[i], ca[j]))

    pairs = np.array(pairs)
    native_dist = md.compute_distances(native, pairs)[0]

    # Set up the system
    system = parm.createSystem(nonbondedMethod=app.PME,
                               nonbondedCutoff=1.2*unit.nanometers,
                               constraints=app.HBonds,
                               hydrogenMass=4*unit.amu)

    # get the atom positions for the system from the equilibrated system
    input_positions = app.PDBFile(eq_pdb).getPositions()

    # Set up and run MD
    platform = mm.Platform.getPlatformByName('CUDA')
    properties = {'DeviceIndex': f'{cuda}', 'CudaPrecision': 'mixed'}

    # Temperature ladder
    temperatures = np.arange(293.15, 343.15 + 2.5, 2.5)	# K
    steps_per_cycle = 25000    						    # 100 ps @ 4 fs
    max_cycles = 1000
    window = 50

    tol_q = 0.02
    tol_rg = 0.05      # nm
    tol_rmsd = 0.05    # nm
    tol_tm = 1.0       # K

    Tm_history = []

    # Create simulations
    simulations = {}

    for T in temperatures:
        integrator = mm.LangevinMiddleIntegrator(T*unit.kelvin,
                                                  1.0/unit.picosecond,
                                                  0.004*unit.picoseconds)
        sim = app.Simulation(parm.topology, system, integrator,
                             platform, properties)
        sim.context.setPositions(input_positions)

        print(f"Created {T} K")
        simulations[T] = sim

    # Data containers
    rmsd_dict = defaultdict(list)
    rgyr_dict = defaultdict(list)
    q_dict = defaultdict(list)

    # Production
    print(f"Running parallel MD simulations on GPU {args.cuda}")
    for cycle in range(max_cycles):
        for T, sim in simulations.items():

            sim.step(steps_per_cycle)
            state = sim.context.getState(getPositions=True)
            xyz = state.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
            rmsd, rg, q = compute_observables(
				xyz,
				top_md,
				native,
				ca,
				pairs,
				native_dist
			)

            # append to dictionaries
            rmsd_dict[T].append(rmsd)
            rgyr_dict[T].append(rg)
            q_dict[T].append(q)

        if cycle % 10 == 0:

            print(f"\nCycle {cycle}")
            mean_q = {T: np.mean(q_dict[T])for T in temperatures}

            current_tm = fit_current_tm(mean_q)
            if current_tm is not None:
                Tm_history.append(current_tm)
                print(f"Current estimated Tm {current_tm:.2f} K")

            # convergence check--
            all_conv = True

            for T in temperatures:
                q_ok = converged(q_dict[T],tol_q,window)
                rg_ok = converged(rgyr_dict[T],tol_rg,window)
                rmsd_ok = converged(rmsd_dict[T],tol_rmsd,window)
                if not (q_ok and rg_ok and rmsd_ok):
                    all_conv = False
                    break

            tm_conv = False

            if len(Tm_history) >= 10:
                old_tm = np.mean(Tm_history[-10:-5])
                new_tm = np.mean(Tm_history[-5:])
                tm_conv = (abs(new_tm - old_tm) < tol_tm)
            if all_conv and tm_conv:
                print("\nConvergence reached")
                print(f"Final Tm = {Tm_history[-1]:.2f} K")
                break

    print("\nPrinting results...")
    # get mean values for all parameters
    mean_q = {T: np.mean(q_dict[T]) for T in temperatures}

    # curve fitting
    T_fit = np.array(sorted(mean_q.keys()))
    Q_fit = np.array([mean_q[T] for T in T_fit])

    popt, _ = curve_fit(
        sigmoid,
        T_fit,
        Q_fit,
        p0=[
            max(Q_fit),
            min(Q_fit),
            np.median(T_fit),
            10.0
        ],
        maxfev=10000
    )

    Qf, Qu, Tm, k = popt

    print("\nFinal fit")

    print(f"Qf = {Qf:.3f}")
    print(f"Qu = {Qu:.3f}")
    print(f"Tm = {Tm:.2f} K")
    print(f"k  = {k:.2f}")

    # produce denaturation plot
    Q_pred = sigmoid(T_fit, *popt)

    print("\nT(K)   Mean_Q   Sigmoid_Q")

    for t, q, fit in zip(T_fit,Q_fit,Q_pred):
        print(
            f"{t:5.1f} "
            f"{q:8.4f} "
            f"{fit:8.4f}"
        )

    # print melting temperature plot
    # Generate a smooth fitted curve
    T_smooth = np.linspace(T_fit.min(), T_fit.max(), 500)
    Q_smooth = sigmoid(T_smooth, *popt)

    # Plot
    plt.figure(figsize=(8, 6))

    # Experimental/simulated points
    plt.scatter(
        T_fit,
        Q_fit,
        color="navy",
        s=60,
        label="Mean Q"
    )

    # Sigmoidal fit
    plt.plot(
        T_smooth,
        Q_smooth,
        color="red",
        lw=2,
        label=f"Fit (Tm = {Tm:.2f} K)"
    )

    # Optional vertical Tm marker
    plt.axvline(
        Tm,
        color="gray",
        linestyle="--",
        alpha=0.7,
        label=f"Tm = {Tm:.2f} K"
    )

    plt.xlabel("Temperature (K)")
    plt.ylabel("Fraction of native contacts (Q)")
    plt.title("Thermal Unfolding Curve")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    outfile = os.path.join(out_dir, "denaturation_curve.png")
    plt.savefig(outfile, dpi=300)
    plt.close()

    print(f"Plot written to: {outfile}")

    # save raw data
    pd.DataFrame({
        "Temperature_K": T_fit,
        "Mean_Q": Q_fit,
        "Fit_Q": Q_pred
    }).to_csv(
        os.path.join(out_dir, "denaturation_curve.csv"),
        index=False
    )

    print("Raw data written to denaturation_curve.csv")

if __name__ == "__main__":
    # Parse the CLI arguments
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument("-s", "--structure", type=str, default='solvated.rst7',
                        help='input structure file name (default: %(default)s)')
    parser.add_argument("-p", "--parameters", type=str, default='solvated.prm7',
                        help='input topology file name (default: %(default)s)')
    parser.add_argument("-o", "--output", type=str, default='.',
                        help='output location (default: %(default)s)')
    parser.add_argument("-cuda", type=str, default="0",
                        help="GPU to be used (default: %(default)s)")

    args = parser.parse_args()
    main(args)
