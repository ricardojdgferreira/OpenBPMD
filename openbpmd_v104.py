#!/usr/bin/env python
descriptn = \
    """
    OpenBPMD - an open source implementation of Binding Pose Metadynamics
    (BPMD) with OpenMM. Replicates the protocol as described by
    Clark et al. 2016 (DOI: 10.1021/acs.jctc.6b00201).

    Runs ten 10 ns metadynamics simulations that biases the RMSD of the ligand.

    The stability of the ligand is calculated using the ligand RMSD (PoseScore)
    and the persistence of the original noncovalent interactions between the
    protein and the ligand (ContactScore). Stable poses have a low RMSD and
    a high fraction of the native contacts preserved until the end of the
    simulation.

    A composite score is calculated using the following formula:
    CompScore = PoseScore - 5 * ContactScore
    
    Version 1.0.4
      >> instead of N reps, it sequentially performs MD runs within
         the specified temperature ramp (300k-450k, dT=10K)
      >> uses fingerprint scoring (IFPcs) instead of ContactScore
      >> outputs "titration_profile.png" and "titration_timeline.png"
         instead of "results.csv"
      >> now writes checkpoint files (MD runs restart from the previous step)
      >> nonbondedCutoff increased to 1.2*nanometers
    """

# general
import os, warnings, re
import glob
import argparse, json
import pandas as pd

# OpenMM
from openmm import *
from openmm.app import *
from openmm.unit import *

# Analysis
import numpy as np
import MDAnalysis as mda
from MDAnalysis.analysis import rms
import mdtraj as md
import parmed as pmd

# fingerprinting
import oddt
from oddt import fingerprints
import sklearn.metrics

# plotting
import matplotlib.pyplot as plt
from scipy.stats import linregress

__author1__ = "Dominykas Lukauskis"
__author2__ = "Ricardo J. Ferreira"
__version__ = "1.0.4"
__email1__ = "dominykas.lukauskis.19@ucl.ac.uk"
__email2__ = "ricardo.ferreira@rgdiscovery.com"


# supress warning
warnings.filterwarnings("ignore")


def main(args):
    """Main entry point of the app. Takes in argparse.Namespace object as
    a function argument. Carries out a sequence of steps required to obtain a
    stability score for a given ligand pose in the provided structure file.

    1. Load the structure and parameter files.
    2. If absent, create an output folder.
    3. Minimization up to ener. tolerance of 10 kJ/mol.
    4. 500 ps equilibration in NVT ensemble with position
       restraints on solute heavy atoms with the force 
       constant of 5 kcal/mol/A^2
    5. Run ...
    6. Collect results from the OpenBPMD simulations and
       write a final score for a given protein-ligand
       structure.

    Parameters
    ----------
    args.structure : str, default='solvated.rst7'
        Name of the structure file, either Amber or Gromacs format.
    args.parameters : str, default='solvated.prm7'
        Name of the parameter or topology file, either Amber or Gromacs
        format.
    args.output : str, default='.'
        Path to and the name of the output directory.
    args.ligand : str, default='LIG' or 'A'
        Residue name of the ligand in the structure/parameter file.
    args.temp_ramp : float, default=[300, 450]
        Temperature ramp to be used.
    """

    print("\n###########################################################")
    print(" 													      ")
    print("	OpenTTMD  - Thermal Titration with OpenMM                 ")
    print(" 													      ")
    print("###########################################################")

    if args.structure.endswith('.gro'):
        coords = GromacsGroFile(args.structure)
        box_vectors = coords.getPeriodicBoxVectors()
        parm = GromacsTopFile(args.parameters, periodicBoxVectors=box_vectors)
    else:
        coords = AmberInpcrdFile(args.structure)
        parm = AmberPrmtopFile(args.parameters)

    if not os.path.isdir(f'{args.output}'):
        os.mkdir(f'{args.output}')

    # Minimize
    min_file_name = 'minimized_system.pdb'
    if not os.path.isfile(os.path.join(args.output,min_file_name)):
        print("\nMinimizing...")
        minimize(parm, coords.positions, args.output, min_file_name)
    min_pdb = os.path.join(args.output,min_file_name)

    # Equilibrate
    eq_file_name = 'equil_system.pdb'
    if not os.path.isfile(os.path.join(args.output,eq_file_name)):
        print("\nEquilibrating...\n")
        equilibrate(min_pdb, parm, args.output, eq_file_name)
    else:
        print("\nResuming from last state...\n")
    eq_pdb = os.path.join(args.output,eq_file_name)
    cent_eq_pdb = os.path.join(args.output,'centred_'+eq_file_name)
    if os.path.isfile(eq_pdb) and not os.path.isfile(cent_eq_pdb):
	# mdtraj can't use GMX TOP, so we have to specify the GRO file instead
        if args.structure.endswith('.gro'):
            mdtraj_top = args.structure
        else:
            mdtraj_top = args.parameters
        mdu = md.load(eq_pdb, top=mdtraj_top)
        mdu.image_molecules()
        mdu.save_pdb(cent_eq_pdb)

    # Run N number of production simulations between temperature intervals
    for idx in range(args.temp_ramp[0], args.temp_ramp[1] + 10, 10):
        rep_dir = os.path.join(args.output,f'rep_{idx}')
        if not os.path.isdir(rep_dir):
            os.mkdir(rep_dir)

        if os.path.isfile(os.path.join(rep_dir,'ttmd_results.csv')):
            continue

        print("Running at temperature:", str(idx), "K")        
        produce(args.output, idx, eq_pdb, parm, args.parameters, args.structure, args.temp_ramp[0])
                
        trj_name = os.path.join(rep_dir,'trj.dcd')               
        IFPcs = get_fp_score(cent_eq_pdb, trj_name, args.lig_resname)

        # Save scores to CSV
        df = pd.DataFrame(IFPcs, columns=['IFPcs'])
        df.to_csv(os.path.join(rep_dir,'ttmd_results.csv'), index=False)
        
        last_2_ns = len(IFPcs)//5
        if np.mean(IFPcs[-last_2_ns:]) > -0.05:
            break

    print("\nProcessing results...")        
    collect_results(cent_eq_pdb, args.output)
    print("\nCalculations finished!")

    return None
    

def get_fp_score(structure_file, trajectory_file, lig_resname):
    """A function the gets the ContactScore from an OpenBPMD trajectory.

    Parameters
    ----------
    structure_file : str
        The name of the centred equilibrated system PDB file that 
        was used to start the OpenBPMD simulation.
    trajectory_file : str
        The name of the OpenBPMD trajectory file.
    lig_resname : str
        Residue name of the ligand that was biased.

    Returns
    -------
    fp_scores : np.array 
        Fingerprint scoring for every frame of the trajectory.
    """
    
    # this creates finerprint reference
    ref = mda.Universe(structure_file)
    protein = ref.select_atoms('protein')
    protein_file = os.path.join(args.output,'protein_ref.pdb')
    with mda.Writer(protein_file, protein.n_atoms) as w:
        w.write(protein)
    ligand = ref.select_atoms(f'resname {lig_resname}')
    ligand_file = os.path.join(args.output,'ligand_ref.pdb')
    with mda.Writer(ligand_file, ligand.n_atoms) as w:
        w.write(ligand)
    protein = next(oddt.toolkit.readfile('pdb', protein_file))
    protein.protein = True
    ligand = next(oddt.toolkit.readfile('pdb', ligand_file))
    fp_ref = fingerprints.InteractionFingerprint(ligand, protein, strict='no')
    os.system(f'rm -r {protein_file} {ligand_file}')
    
    # this calculates fingerprints for the trajectory
    u = mda.Universe(structure_file, trajectory_file)
    fps = []
    fp_scores = []
    
    for i in range(0, len(u.trajectory)):
        f = calc_ifp(u, i, lig_resname)
        fps.append(f)
        
    for fp in fps:
        l_plif_temp = []
        l_plif_temp.append(fp_ref)
        l_plif_temp.append(fp)
        matrix = np.stack(l_plif_temp, axis=0)
        idx = np.argwhere(np.all(matrix[..., :] == 0, axis=0))
        matrix_dense = np.delete(matrix, idx, axis=1)
        x = matrix_dense[0].reshape(1,-1)
        y = matrix_dense[1].reshape(1,-1)
        sim_giovanni = float(sklearn.metrics.pairwise.cosine_similarity(x, y).item())
        sim = round(sim_giovanni * -1,3)
        fp_scores.append(sim)
         
    return fp_scores


def calc_ifp(u, i, lig_resname):
    # calculate reference
    u.trajectory[i]        
    u_protein = u.select_atoms('protein')
    protein_file = os.path.join(args.output,f'protein_{i}.pdb')
    with mda.Writer(protein_file, u_protein.n_atoms) as w:
        w.write(u_protein)
    u_ligand = u.select_atoms(f'resname {lig_resname}')
    ligand_file = os.path.join(args.output,f'ligand_{i}.pdb')
    with mda.Writer(ligand_file, u_ligand.n_atoms) as w:
            w.write(u_ligand)
    p = next(oddt.toolkit.readfile('pdb', protein_file))
    p.protein = True
    l = next(oddt.toolkit.readfile('pdb', ligand_file))
    fp = fingerprints.InteractionFingerprint(l, p, strict='no')
    os.system(f'rm -r {protein_file} {ligand_file}')

    return fp


def minimize(parm, input_positions, out_dir, min_file_name):
    """An energy minimization function down with an energy tolerance
    of 10 kJ/mol.

    Parameters
    ----------
    parm : Parmed or OpenMM parameter file object
        Used to create the OpenMM System object.
    input_positions : OpenMM Quantity
        3D coordinates of the equilibrated system.
    out_dir : str
        Directory to write the outputs.
    min_file_name : str
        Name of the minimized PDB file to write.
    """
    system = parm.createSystem(nonbondedMethod=PME, nonbondedCutoff=1.2*nanometers, constraints=HBonds,)

    # Define platform properties
    platform = Platform.getPlatformByName('CUDA')
    properties = {'DeviceIndex': '0', 'CudaPrecision': 'mixed'}

    # Set up the simulation parameters
    # Langevin integrator at 300 K w/ 1 ps^-1 friction coefficient
    # and a 2-fs timestep
    # NOTE - no dynamics performed, but required for setting up
    # the OpenMM system.
    integrator = LangevinIntegrator(300*kelvin, 1/picosecond, 0.002*picoseconds)
    simulation = Simulation(parm.topology, system, integrator, platform, properties)
    simulation.context.setPositions(input_positions)

    # Minimize the system - no predefined number of steps
    simulation.minimizeEnergy()

    # Write out the minimized system to use w/ MDAnalysis
    positions = simulation.context.getState(getPositions=True).getPositions()
    out_file = os.path.join(out_dir,min_file_name)
    PDBFile.writeFile(simulation.topology, positions, open(out_file, 'w'))

    return None


def equilibrate(min_pdb, parm, out_dir, eq_file_name):
    """A function that does a 500 ps NVT equilibration with position
    restraints, with a 5 kcal/mol/A**2 harmonic constant on solute heavy
    atoms, using a 2 fs timestep.

    Parameters
    ----------
    min_pdb : str
        Name of the minimized PDB file.
    parm : Parmed or OpenMM parameter file object
        Used to create the OpenMM System object.
    out_dir : str
        Directory to write the outputs to.
    eq_file_name : str
        Name of the equilibrated PDB file to write.
    """
    
    # Get the solute heavy atom indices to use
    # for defining position restraints during equilibration
    universe = mda.Universe(min_pdb, format='XPDB', in_memory=True)

    system = parm.createSystem(nonbondedMethod=PME, nonbondedCutoff=1.2*nanometers, constraints=HBonds,)

    # Add the restraints on the positions of specified atoms
    restraint = CustomExternalForce('k*periodicdistance(x, y, z, x0, y0, z0)^2')
    system.addForce(restraint)
    restraint.addGlobalParameter('k', 100.0*kilojoules_per_mole/nanometer)
    restraint.addPerParticleParameter('x0')
    restraint.addPerParticleParameter('y0')
    restraint.addPerParticleParameter('z0')

    input_positions = PDBFile(min_pdb).getPositions()
    positions = input_positions

    # Go through the indices of all heavy atoms and apply restraints
    protein_resnames = {"ALA","ARG","ASN","ASP","CYS","GLN","GLU","GLY","HIS","ILE","LEU","LYS","MET","PHE","PRO","SER","THR","TRP","TYR","VAL"}
    pdb = PDBFile(min_pdb)
    for atom in pdb.topology.atoms():
        if atom.residue.name in protein_resnames and atom.element.symbol != "H":
            restraint.addParticle(atom.index, pdb.positions[atom.index])

    integrator = LangevinIntegrator(300*kelvin, 1/picosecond, 0.002*picoseconds)
    platform = Platform.getPlatformByName('CUDA')
    properties = {'DeviceIndex': '0', 'CudaPrecision': 'mixed'}

    sim = Simulation(parm.topology, system, integrator, platform, properties)
    sim.context.setPositions(input_positions)
    integrator.step(25000)  # run 50 ps of equilibration

    # Write out the equilibrated system to use w/ MDAnalysis
    positions = sim.context.getState(getPositions=True, enforcePeriodicBox=True).getPositions()
    out_file = os.path.join(out_dir, eq_file_name)
    PDBFile.writeFile(sim.topology, positions, open(out_file, 'w'))

    return None


def produce(out_dir, idx, eq_pdb, parm, parm_file, coords_file, ref_temp):
    """An OpenBPMD production simulation function. Ligand RMSD is biased with
    metadynamics. The integrator uses a 4 fs time step and
    runs for 10 ns, writing a frame every 100 ps.

    Writes a 'trj.dcd', 'COLVAR.npy', and 'bias_*.npy' files
    during the metadynamics simulation in the '{out_dir}/rep_{idx}' directory.
    After the simulation is done, it analyses the trajectories and writes a
    'bpm_results.csv' file with time-resolved PoseScore and ContactScore.

    Parameters
    ----------
    out_dir : str
        Directory where your equilibration PDBs and 'rep_*' dirs are at.
    idx : int
        Current replica index.
    ligand : str
        Residue name (or chain) of the ligand.
    eq_pdb : str
        Name of the PDB for equilibrated system.
    parm : Parmed or OpenMM parameter file object
        Used to create the OpenMM System object.
    parm_file : str
        The name of the parameter or topology file of the system.
    coords_file : str
        The name of the coordinate file of the system.
    temp : int
        Temperature of the simulation, in K

    """
    # First, assign the replica directory to which we'll write the files
    write_dir = os.path.join(out_dir,f'rep_{idx}')
    # Get the anchor atoms by ...
    universe = mda.Universe(eq_pdb, format='XPDB', in_memory=True)

    # Set up the system
    system = parm.createSystem(nonbondedMethod=PME, nonbondedCutoff=1.2*nanometers, constraints=HBonds, hydrogenMass=4*amu)
    for force in system.getForces():
        if isinstance(force, openmm.NonbondedForce):
            force.setUseSwitchingFunction(True)
            force.setSwitchingDistance(1.0*nanometers)

    # get the atom positions for the system from the equilibrated system
    input_positions = PDBFile(eq_pdb).getPositions()

    # Set up and run MD
    platform = Platform.getPlatformByName('CUDA')
    properties = {'DeviceIndex': '0', 'CudaPrecision': 'mixed'}

    # create integrator and simulation
    integrator = LangevinIntegrator(idx*kelvin, 1.0/picosecond, 0.004*picoseconds)
    simulation = Simulation(parm.topology, system, integrator, platform, properties)
    
    # allow continuation of previous run by using checkpoints
    if idx == ref_temp:
        simulation.context.setPositions(input_positions)
    else:
        chk_file = os.path.join(out_dir,f'rep_{idx - 10}','state.chk')
        simulation.loadCheckpoint(chk_file)
        integrator.setTemperature(idx*kelvin)

    trj_name = os.path.join(write_dir,'trj.dcd')
    chk_name = os.path.join(write_dir,'state.chk')
    pdb_name = os.path.join(write_dir,f'final_{idx}.pdb')

    sim_time = 10  # ns
    steps = 250000 * sim_time

    simulation.reporters.append(DCDReporter(trj_name, 25000))
    simulation.reporters.append(CheckpointReporter(chk_name, 5000))
    simulation.reporters.append(StateDataReporter(os.path.join(write_dir,'sim_log.csv'), 250000, step=True, temperature=True, progress=True, remainingTime=True, speed=True, totalSteps=steps, separator='\t'))
    simulation.step(steps)

    positions = simulation.context.getState(getPositions=True, enforcePeriodicBox=True).getPositions()  

    # center everything using MDTraj, to fix any PBC imaging issues
    # mdtraj can't use GMX TOP, so we have to specify the GRO file instead
    if coords_file.endswith('.gro'):
        mdtraj_top = coords_file
    else:
        mdtraj_top = parm_file
    mdu = md.load(trj_name, top=mdtraj_top)
    mdu.image_molecules()
    mdu.save(trj_name)
    mdu2pdb = mdu[-1]
    mdu2pdb.save_pdb(pdb_name)

    return None


def collect_results(structure_file, write_dir):
    """A function that calculates i) average IFPcs regarding
    simulation temperature, and ii) both IFPcs and RMSD evolution
    over time. It concatenatesall trajectories into a single one
    and retrieves RMSD. Reads all "ttmd_results.csv" files and
    produces a plot (colored by temperature) of its evolution.

    Writes a 'titration_profile.png' and 'titration_timeline.png'
    file in 'out_dir' directory. MS value is inside the former.
    
    Parameters
    ----------
    in_dir : str
        Directory with 'rep_*' directories.
    out_dir : str
        Directory where the 'results.csv' file will be written
    """
    u = mda.Universe(structure_file)

    # find how many repeats have been run
    glob_str = os.path.join(write_dir, 'rep_*')
    reps = glob.glob(glob_str)
    reps.sort(key=lambda f: int(re.sub('\D', '', f)))
    
    # get all tested temperatures
    temperatures = []
    for t in reps:
        temperatures.append(int(t[-3:]))

    # getting IFPcs from result files
    fps = [os.path.join(write_dir, f'rep_{idx}','ttmd_results.csv') for idx in temperatures]
    IFPtp = []
    IFPtt = []
    for fp in fps:
        df = pd.read_csv(fp)
        IFPtt.append(df['IFPcs'].tolist())				# for titration timeline
        IFPlist = df['IFPcs'].tolist()
        last_2_ns = len(IFPlist)//5
        IFPtp.append(np.mean(IFPlist[-last_2_ns:]))		# for titration profile
    
    # getting rmsd from trajectories
    trajectories = [os.path.join(write_dir, f'rep_{idx}','trj.dcd') for idx in temperatures]
    full_traj = os.path.join(write_dir, "full_trajectory.dcd")
    if os.path.isfile(full_traj) == False:
        with mda.Writer(full_traj, u.atoms.n_atoms) as W:
            for traj in trajectories:
                u.load_new(traj)
                for ts in u.trajectory:
                    W.write(u.atoms)
    u_new = mda.Universe(structure_file, full_traj)

    # get time steps to use as x on axs[0] and axs[1]
    time_steps = [ts.time for ts in u_new.trajectory]

    # get RMSDs to use as y on axs[1]
    rmsd_prot = rms.RMSD(u_new, select='backbone',groupselections=['protein'], ref_frame=0).run()
    rmsd_lig = rms.RMSD(u_new, select='backbone',groupselections=['resname UNL and not name H*'], ref_frame=0).run()

    # plotting titration timeline
    fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(12,9))
    x = time_steps											# time steps as x
    r_y1 = rmsd_prot.results.rmsd[:, -1]					# backbone RMSD as y1
    r_y2 = rmsd_lig.results.rmsd[:, -1]						# ligand RMSD as y2
    fp_y = [item for sublist in IFPtt for item in sublist]
    
    #plot IFPcs
    N = len(x)
    edges = np.linspace(0, N, len(reps) + 1).astype(int)
    cmap = plt.get_cmap('tab10', len(reps))				    # different colors according to temperature
    labels = [f'{i} K' for i in temperatures]				# different labels according to temperature

    for i in range(len(reps)):
        s, e = edges[i], edges[i+1]
        axs[0].plot(x[s:e], fp_y[s:e], color=cmap(i), label=labels[i])
    axs[0].set_ylabel('IFPcs')
    axs[0].set_xlabel('MD steps')
    axs[0].set_title('IFPcs')
    axs[0].set_xlim(0,time_steps[-1])
    axs[0].set_ylim(-1.1, 0.1)
    axs[0].set_xticks(np.arange(0, len(time_steps)+1, 100))
    axs[0].legend()
    axs[0].legend(loc='center left', bbox_to_anchor=(1, 0.5))

    # plot RMSDs
    axs[1].plot(x, r_y1, label='Backbone')
    axs[1].plot(x, r_y2, label='Ligand')
    axs[1].set_ylabel('RMSD (Å)')
    axs[1].set_xlabel('MD steps')
    axs[1].set_title('RMSD')
    axs[1].set_xlim(0,time_steps[-1])
    axs[1].set_ylim(0)
    axs[1].set_xticks(np.arange(0, len(time_steps)+1, 100))
    axs[1].legend()
    axs[1].legend(loc='center left', bbox_to_anchor=(1, 0.5))
    fig.tight_layout()
    fig.savefig(os.path.join(write_dir,"titration_timeline.png"), dpi=300)
    
    # print titration profile and MS result
    temp = [int(x) for x in temperatures]					# temperatures as x
    IFP = IFPtp												# average of IFPcs (last 2 ns) as y
    
    fig, axs = plt.subplots(nrows=1, ncols=1)
    first_last_t = [temp[0], temp[-1]]
    axs.scatter(temp, IFP)
    first_last_score = [-1, IFP[-1]]
    try:
        f = np.poly1d(np.polyfit(first_last_t, first_last_score, 1))
        slope, intercept, r_value, p_value, std_err = linregress(first_last_t, first_last_score)
        axs.plot(temp, f(temp), ls='--', label="MS = {:.5f}".format(slope))
    except Exception:
        slope = 'Impossible to calculate MS: required at least 2 TTMD steps'

    axs.set_title('Titration Profile')
    axs.set_xlabel('Temperature (K)')
    axs.set_ylabel('Average IFP$_{CS}$')

    axs.set_xlim(first_last_t)
    axs.set_ylim(first_last_score)
    axs.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(write_dir, "titration_profile.png"), dpi=300) 
    

if __name__ == "__main__":
    """ This is executed when run from the command line """
    # Parse the CLI arguments
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=descriptn)

    parser.add_argument("-s", "--structure", type=str, default='solvated.rst7',
                        help='input structure file name (default: %(default)s)')
    parser.add_argument("-p", "--parameters", type=str, default='solvated.prm7',
                        help='input topology file name (default: %(default)s)')
    parser.add_argument("-o", "--output", type=str, default='.',
                        help='output location (default: %(default)s)')
    parser.add_argument("-lig_resname", type=str, default='MOL',
                        help='the name of the ligand (default: %(default)s)')
    parser.add_argument("-temp_ramp", type=json.loads, default="[300,450]",
                        help="the min and max for the temperature ramp (default: %(default)s)")

    args = parser.parse_args()
    main(args)
