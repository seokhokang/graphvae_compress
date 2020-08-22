import numpy as np
import pickle as pkl
import sys, sparse
from util import atomFeatures, bondFeatures#, _vec_to_mol
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors

def read_smiles(smiles_file):
    with open(smiles_file, 'r') as f:
        smiles_list = [line.strip() for line in f.readlines()]
    return smiles_list

# GuacaMol benchmark dataset can be downloaded from
# https://github.com/BenevolentAI/guacamol
# https://ndownloader.figshare.com/files/13612745
smisuppl = np.array(read_smiles('./guacamol_v1_all.smiles'))
print('count:', len(smisuppl))

atom_list = ['B','C','N','O','F','Si','P','S','Cl','Se','Br','I']
bond_list = ['SINGLE', 'DOUBLE', 'TRIPLE']
bpatt_dim = [3, 3, 2, 1, 2, 1]

bpatt1 = Chem.MolFromSmarts('*-[#6;D2]-[#6;D2]-*')#3
bpatt2 = Chem.MolFromSmarts('*-[#6;D2]-*')#3
bpatt3 = Chem.MolFromSmarts('*-[#6;D2]=[#6;D2]-*')#2
bpatt4 = Chem.MolFromSmarts('*=[#6;D2]-[#6;D2]=*')#1
bpatt5 = Chem.MolFromSmarts('*-[#7;D2]-*')#2
bpatt6 = Chem.MolFromSmarts('*-[#8;D2]-*')#1
bpatt_list = [bpatt1, bpatt2, bpatt3, bpatt4, bpatt5, bpatt6]

n_max = 52
dim_node = 4 + 3 + len(atom_list)
dim_edge = len(bond_list) + sum(bpatt_dim)

current_max = 0

DV = sparse.COO.from_numpy(np.empty(shape=(0, n_max, dim_node), dtype=bool))
DE = sparse.COO.from_numpy(np.empty(shape=(0, n_max, n_max, dim_edge), dtype=bool)) 
DY = []
Dsmi = []

dv_list = []
de_list = []

for i, smi in enumerate(smisuppl):

    if i % 10000 == 0: print(i, len(Dsmi), current_max, flush=True)

    smi = Chem.MolToSmiles(Chem.MolFromSmiles(smi), isomericSmiles=False)
    mol = Chem.MolFromSmiles(smi)  
    Chem.Kekulize(mol)

    n_atom = mol.GetNumHeavyAtoms()

    # node DV
    node = np.zeros((n_atom, dim_node), dtype=bool)
    for j in range(n_atom):
        atom = mol.GetAtomWithIdx(j)
        node[j, :] = atomFeatures(atom, atom_list)

    # edge pattern search
    del_ids = []
    case_list = []
    for m in range(len(bpatt_list)):
        for aids in mol.GetSubstructMatches(bpatt_list[m]):
            if np.sum([(aid in del_ids) for aid in aids]) == 0 and np.sum(np.abs([mol.GetAtomWithIdx(aid).GetFormalCharge() for aid in aids])) == 0:
                del_ids = del_ids + list(aids[1:-1])
                case_list.append([m, np.min([aids[0], aids[-1]]), np.max([aids[0], aids[-1]])])

    # edge DE
    edge = np.zeros((n_atom, n_atom, dim_edge), dtype=bool)
    for j in range(n_atom - 1):
        for k in range(j + 1, n_atom):
            bond = mol.GetBondBetweenAtoms(j, k)
            if bond is not None:
                edge[j, k, :] = np.concatenate([bondFeatures(bond, bond_list), np.zeros(np.sum(bpatt_dim))], 0)
                
            for m in range(len(bpatt_list)):
                if [m, j, k] in case_list:
                    #assert case_list.count([m, j, k]) <= bpatt_dim[m]                        
                    edge[j, k, int(len(bond_list) + np.sum(bpatt_dim[:m]) + case_list.count([m, j, k]) - 1)] = 1

            edge[k, j, :] = edge[j, k, :]

    # compression
    node = np.delete(node, del_ids, 0)
    edge = np.delete(edge, del_ids, 0)
    edge = np.delete(edge, del_ids, 1)
    
    #assert node.shape[0] <= n_max        
    if current_max < node.shape[0]: current_max = node.shape[0]
        
    node = np.pad(node, ((0, n_max - node.shape[0]), (0, 0)))
    edge = np.pad(edge, ((0, n_max - edge.shape[0]), (0, n_max - edge.shape[1]), (0, 0))) 

    # property DY
    property = [Descriptors.ExactMolWt(mol), Descriptors.MolLogP(mol)]

    # validity check: smi vs. _vec_to_mol(node, edge, atom_list, bpatt_dim)

    # append
    dv_list.append(node)
    de_list.append(edge)
    DY.append(property)
    Dsmi.append(smi)
    
    if i % 10000 == 0 or i == len(smisuppl)-1:
        DV = np.concatenate([DV, sparse.COO.from_numpy(dv_list)], 0)
        DE = np.concatenate([DE, sparse.COO.from_numpy(de_list)], 0)
        dv_list = []
        de_list = []


# np array    
DY = np.asarray(DY)
Dsmi = np.asarray(Dsmi)

DV = DV[:,:current_max,:]
DE = DE[:,:current_max,:current_max,:]

print(DV.shape, DE.shape, DY.shape)

# save
with open('guacamol_graph.pkl','wb') as fw:
    pkl.dump([DV, DE, DY, Dsmi], fw)