import argparse
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Recap, BRICS
from collections import defaultdict
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
import re
from pathlib import Path
from rdkit.Chem.rdchem import RWMol

from preprocess_utils import remove_atom_mapping
from preprocess_utils import convert_smiles_to_mol_objects
from preprocess_utils import set_atom_map_numbers

# JT
import rdkit
import rdkit.Chem as Chem
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
from collections import defaultdict
from rdkit.Chem import rdmolfiles

MST_MAX_WEIGHT = 100 
MAX_NCAND = 2000

def tree_decomp(mol):
    n_atoms = mol.GetNumAtoms()
    if n_atoms == 1:    
        return [[0]], []

    cliques = []
    for bond in mol.GetBonds():
        a1 = bond.GetBeginAtom().GetIdx()
        a2 = bond.GetEndAtom().GetIdx()
        if not bond.IsInRing():  # non-ring 
            cliques.append([a1,a2])

    ssr = [list(x) for x in Chem.GetSymmSSSR(mol)]  # ring
    cliques.extend(ssr)

    nei_list = [[] for i in range(n_atoms)]
    for i in range(len(cliques)):
        for atom in cliques[i]:
            nei_list[atom].append(i)
    
    #Merge Rings with intersection > 2 atoms
    for i in range(len(cliques)):
        if len(cliques[i]) <= 2: continue
        for atom in cliques[i]:
            for j in nei_list[atom]:
                if i >= j or len(cliques[j]) <= 2: continue
                inter = set(cliques[i]) & set(cliques[j])
                if len(inter) > 2:
                    cliques[i].extend(cliques[j])
                    cliques[i] = list(set(cliques[i]))
                    cliques[j] = []

    cliques = [c for c in cliques if len(c) > 0]
    nei_list = [[] for i in range(n_atoms)]
    for i in range(len(cliques)):
        for atom in cliques[i]:
            nei_list[atom].append(i)
    
    #Build edges and add singleton cliques
    edges = defaultdict(int)
    for atom in range(n_atoms):
        if len(nei_list[atom]) <= 1: 
            continue
        cnei = nei_list[atom]
        bonds = [c for c in cnei if len(cliques[c]) == 2]
        rings = [c for c in cnei if len(cliques[c]) > 4]
        if len(bonds) > 2 or (len(bonds) == 2 and len(cnei) > 2): #In general, if len(cnei) >= 3, a singleton should be added, but 1 bond + 2 ring is currently not dealt with.
            cliques.append([atom])
            c2 = len(cliques) - 1
            for c1 in cnei:
                edges[(c1,c2)] = 1
        elif len(rings) > 2: #Multiple (n>2) complex rings
            cliques.append([atom])
            c2 = len(cliques) - 1
            for c1 in cnei:
                edges[(c1,c2)] = MST_MAX_WEIGHT - 1
        else:
            for i in range(len(cnei)):
                for j in range(i + 1, len(cnei)):
                    c1,c2 = cnei[i],cnei[j]
                    inter = set(cliques[c1]) & set(cliques[c2])
                    if edges[(c1,c2)] < len(inter):
                        edges[(c1,c2)] = len(inter) #cnei[i] < cnei[j] by construction

    edges = [u + (MST_MAX_WEIGHT-v,) for u,v in edges.items()]  # modification :  iteritems() -> items()
    if len(edges) == 0:
        return cliques, edges

    #Compute Maximum Spanning Tree
    row,col,data = zip(*edges)
    n_clique = len(cliques)
    clique_graph = csr_matrix( (data,(row,col)), shape=(n_clique,n_clique) )
    junc_tree = minimum_spanning_tree(clique_graph)
    row,col = junc_tree.nonzero()
    edges = [(row[i],col[i]) for i in range(len(row))]
    return (cliques, edges)

def cliques_to_smiles(mol, cliques):
    smiles_list = []
    for clique in cliques:
        # 서브모르 생성
        atom_indices = list(clique)
        atom_indices.sort()  # 인덱스 순서 정렬
        emol = Chem.EditableMol(Chem.Mol())
        
        # 서브모르 생성 및 원자 추가
        idx_map = {}
        for idx in atom_indices:
            atom = mol.GetAtomWithIdx(idx)
            new_idx = emol.AddAtom(atom)
            idx_map[idx] = new_idx

        # 서브모르에서 결합 추가
        for bond in mol.GetBonds():
            a1 = bond.GetBeginAtomIdx()
            a2 = bond.GetEndAtomIdx()
            if a1 in atom_indices and a2 in atom_indices:
                emol.AddBond(idx_map[a1], idx_map[a2], bond.GetBondType())
        
        submol = emol.GetMol()
        smiles = Chem.MolToSmiles(submol)
        smiles_list.append(smiles)
    return smiles_list

def process_junction_tree_decomposition(data):
    jt_error_indices = []
    jt_all_frag = set()
    jt_indices = []
    jt_mols = []
    bonds_to_break_jt = []

    # jt_decompose 함수 실행을 위한 루프
    for i in range(len(data)):
        try:
            mol = Chem.MolFromSmiles(data['smiles'][i])
            cliques, edges = tree_decomp(mol)
            jt_indices.append(cliques)
            bonds_to_break_jt.append(edges)
            # print(cliques)
            # print(edges)
            mols = cliques_to_smiles(mol, cliques)
            jt_mols.append(mols)
            convert_jt_mols = set(mols)
            jt_all_frag = jt_all_frag.union(convert_jt_mols)
        except Exception as e:
            jt_error_indices.append(i)  # 에러 발생 인덱스 저장

    return jt_error_indices, jt_all_frag, jt_indices, jt_mols, bonds_to_break_jt

    