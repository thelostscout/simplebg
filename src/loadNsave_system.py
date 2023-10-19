import mdtraj as md
import os
import openmm
import bgmol
from bgmol.systems.peptide import peptide

complex_name = 'OppA'
peptide_name = '1qka'
n_atoms = 71
n_res = 3

file_dir = os.path.dirname(os.path.realpath(__file__))
data_path = os.path.realpath(file_dir + "/../data/") + f'/Molecules/{complex_name}/Peptides/{peptide_name}'

system = peptide(short = False, n_atoms=n_atoms, n_res=n_res, filepath=data_path, complex_name=complex_name, peptide_name=peptide_name)
topology = system._mdtraj_topology

traj = md.load_pdb(os.path.join(data_path,'traj.pdb'), top=topology)

traj.save(data_path + '/traj.h5')
