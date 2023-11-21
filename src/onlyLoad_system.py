import mdtraj as md
import os
import openmm
import bgmol
from bgmol.systems.peptide import peptide

complex_name = 'OppA'
peptide_name = '1b5j'
n_atoms = 64
n_res = 3

file_dir = os.path.dirname(os.path.realpath(__file__))
data_path = os.path.realpath(file_dir + "/../data/") + f'/Molecules/{complex_name}/Peptides/{peptide_name}'

system = peptide(short = False, n_atoms=n_atoms, n_res=n_res, filepath=data_path, complex_name=complex_name, peptide_name=peptide_name)
topology = system._mdtraj_topology

nb_fname = complex_name+'_'+peptide_name

traj=md.load_hdf5(f'traj.h5')

traj.n_atoms
data = traj.xyz.reshape(len(traj.xyz), -1)
print(data)
print(traj)
