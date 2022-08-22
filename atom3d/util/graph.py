import numpy as np
import scipy.spatial as ss
import torch
import torch.nn.functional as F
from torch_geometric.utils import to_undirected
from torch_sparse import coalesce

import atom3d.util.formats as fo

# PDB atom names -- these include co-crystallized metals
prot_atoms = ['C', 'H', 'O', 'N', 'S', 'P', 'ZN', 'NA', 'FE', 'CA', 'MN', 'NI', 'CO', 'MG', 'CU', 'CL', 'SE', 'F']
md_prot_atoms = {1: '2C', 2: '3C', 3: 'C', 4: 'C*', 5: 'C8', 6: 'CA', 7: 'CB', 8: 'CC', 9: 'CN', 10: 'CO', 11: 'CR', 12: 'CT', 13: 'CW', 14: 'CX', 15: 'H', 16: 'H1', 17: 'H4', 18: 'H5', 19: 'HA', 20: 'HC', 21: 'HO', 22: 'HP', 23: 'HS', 24: 'N', 25: 'N2', 26: 'N3', 27: 'NA', 28: 'NB', 29: 'O', 30: 'O2', 31: 'OH', 32: 'S', 33: 'SH', 34: 'br', 35: 'c', 36: 'c1', 37: 'c2', 38: 'c3', 39: 'ca', 40: 'cc', 41: 'cd', 42: 'ce', 43: 'cf', 44: 'cg', 45: 'ch', 46: 'cl', 47: 'cp', 48: 'cq', 49: 'cs', 50: 'cu', 51: 'cx', 52: 'cy', 53: 'cz', 54: 'f', 55: 'h1', 56: 'h2', 57: 'h3', 58: 'h4', 59: 'h5', 60: 'ha', 61: 'hc', 62: 'hn', 63: 'ho', 64: 'hp', 65: 'hs', 66: 'hx', 67: 'i', 68: 'n', 69: 'n1', 70: 'n2', 71: 'n3', 72: 'n4', 73: 'n7', 74: 'n8', 75: 'na', 76: 'nb', 77: 'nc', 78: 'nd', 79: 'ne', 80: 'nf', 81: 'nh', 82: 'ni', 83: 'nj', 84: 'nk', 85: 'nl', 86: 'nm', 87: 'nn', 88: 'no', 89: 'nq', 90: 'ns', 91: 'nt', 92: 'nu', 93: 'nv', 94: 'nx', 95: 'ny', 96: 'nz', 97: 'o', 98: 'oh', 99: 'op', 100: 'oq', 101: 'os', 102: 'p5', 103: 'py', 104: 's', 105: 's4', 106: 's6', 107: 'sh', 108: 'ss', 109: 'sx', 110: 'sy'}
md_residues = {0: 'MOL', 1: 'ACE', 2: 'ALA', 3: 'ARG', 4: 'ASN', 5: 'ASP', 6: 'CYS', 7: 'CYX', 8: 'GLN', 9: 'GLU', 10: 'GLY', 11: 'HIE', 12: 'ILE', 13: 'LEU', 14: 'LYS', 15: 'MET', 16: 'PHE', 17: 'PRO', 18: 'SER', 19: 'THR', 20: 'TRP', 21: 'TYR', 22: 'VAL'}

# RDKit molecule atom names
mol_atoms = ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na',
             'Ca', 'Fe', 'As', 'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb',
             'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se', 'Ti', 'Zn', 'H',  # H?
             'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr',
             'Cr', 'Pt', 'Hg', 'Pb']


# Residue names
residues = ['ALA', 'CYS', 'ASP', 'GLU', 'PHE', 'GLY', 'HIS', 'ILE', 'LYS', 'LEU', 'MET', 'ASN', 'PRO', 'GLN', 'ARG',
                'SER', 'THR', 'VAL', 'TRP', 'TYR']


def prot_df_to_graph(df, feat_col='element', allowable_feats=prot_atoms, edge_dist_cutoff=4.5):
    r"""
    Converts protein in dataframe representation to a graph compatible with Pytorch-Geometric, where each node is an atom.

    :param df: Protein structure in dataframe format.
    :type df: pandas.DataFrame
    :param node_col: Column of dataframe to find node feature values. For example, for atoms use ``feat_col="element"`` and for residues use ``feat_col="resname"``
    :type node_col: str, optional
    :param allowable_feats: List containing all possible values of node type, to be converted into 1-hot node features. 
        Any elements in ``feat_col`` that are not found in ``allowable_feats`` will be added to an appended "unknown" bin (see :func:`atom3d.util.graph.one_of_k_encoding_unk`).
    :type allowable_feats: list, optional
    :param edge_dist_cutoff: Maximum distance cutoff (in Angstroms) to define an edge between two atoms, defaults to 4.5.
    :type edge_dist_cutoff: float, optional

    :return: tuple containing

        - node_feats (torch.FloatTensor): Features for each node, one-hot encoded by values in ``allowable_feats``.

        - edges (torch.LongTensor): Edges in COO format

        - edge_weights (torch.LongTensor): Edge weights, defined as a function of distance between atoms given by :math:`w_{i,j} = \frac{1}{d(i,j)}`, where :math:`d(i, j)` is the Euclidean distance between node :math:`i` and node :math:`j`.

        - node_pos (torch.FloatTensor): x-y-z coordinates of each node
    :rtype: Tuple
    """ 
    node_pos = torch.FloatTensor(df[['x', 'y', 'z']].to_numpy())

    kd_tree = ss.KDTree(node_pos)
    edge_tuples = list(kd_tree.query_pairs(edge_dist_cutoff))
    edges = torch.LongTensor(edge_tuples).t().contiguous()
    edges = to_undirected(edges)

    node_feats = torch.FloatTensor([one_of_k_encoding_unk(e, allowable_feats) for e in df[feat_col]])
    edge_weights = torch.FloatTensor(
        [1.0 / (np.linalg.norm(node_pos[i] - node_pos[j]) + 1e-5) for i, j in edges.t()]).view(-1)
    # feats = F.one_hot(elems, num_classes=len(atom_int_dict))
    
    return node_feats, edges, edge_weights, node_pos


def mol_df_to_graph(df, bonds=None, allowable_atoms=None, edge_dist_cutoff=4.5, onehot_edges=True):
    """
    Converts molecule in dataframe to a graph compatible with Pytorch-Geometric

    :param df: Molecule structure in dataframe format
    :type mol: pandas.DataFrame
    :param bonds: Molecule structure in dataframe format
    :type bonds: pandas.DataFrame
    :param allowable_atoms: List containing allowable atom types
    :type allowable_atoms: list[str], optional

    :return: Tuple containing \n
        - node_feats (torch.FloatTensor): Features for each node, one-hot encoded by atom type in ``allowable_atoms``.
        - edge_index (torch.LongTensor): Edges from chemical bond graph in COO format.
        - edge_feats (torch.FloatTensor): Edge features given by bond type. Single = 1.0, Double = 2.0, Triple = 3.0, Aromatic = 1.5.
        - node_pos (torch.FloatTensor): x-y-z coordinates of each node.
    """
    if allowable_atoms is None:
        allowable_atoms = mol_atoms
    node_pos = torch.FloatTensor(df[['x', 'y', 'z']].to_numpy())
    
    
    N = df.shape[0]
    bond_mapping = {1.0: 0, 2.0: 1, 3.0: 2, 1.5: 3}
    
    if bonds is not None:
        bond_data = torch.FloatTensor(bonds)
        edge_tuples = torch.cat((bond_data[:, :2], torch.flip(bond_data[:, :2], dims=(1,))), dim=0)
        edge_index = edge_tuples.t().long().contiguous()
        
        if onehot_edges:
            bond_idx = list(map(lambda x: bond_mapping[x], bond_data[:,-1].tolist())) + list(map(lambda x: bond_mapping[x], bond_data[:,-1].tolist()))
            edge_attr = F.one_hot(torch.tensor(bond_idx), num_classes=4).to(torch.float)
            edge_index, edge_attr = coalesce(edge_index, edge_attr, N, N)
        else:
            edge_attr = torch.cat((torch.FloatTensor(bond_data[:,-1]).view(-1), torch.FloatTensor(bond_data[:,-1]).view(-1)), dim=0)
    else:
        kd_tree = ss.KDTree(node_pos)
        edge_tuples = list(kd_tree.query_pairs(edge_dist_cutoff))
        edge_index = torch.LongTensor(edge_tuples).t().contiguous()
        edge_index = to_undirected(edge_index)
        edge_attr = torch.FloatTensor([1.0 / (np.linalg.norm(node_pos[i] - node_pos[j]) + 1e-5) for i, j in edge_index.t()]).view(-1)
    node_feats = torch.FloatTensor([one_of_k_encoding_unk(e, allowable_atoms) for e in df['element']])

    return node_feats, edge_index, edge_attr, node_pos


def combine_graphs(graph1, graph2, edges_between=True, edges_between_dist=4.5):
    """Combine two graphs into one, optionally adding edges between the two graphs using :func:`atom3d.util.graph.edges_between_graphs`. Node features are concatenated in the feature dimension, to distinguish which nodes came from which graph.

    :param graph1: One of the graphs to be combined, in the format returned by :func:`atom3d.util.graph.prot_df_to_graph` or :func:`atom3d.util.graph.mol_df_to_graph`.
    :type graph1: Tuple
    :param graph2: The other graph to be combined, in the format returned by :func:`atom3d.util.graph.prot_df_to_graph` or :func:`atom3d.util.graph.mol_df_to_graph`.
    :type graph2: Tuple
    :param edges_between: Indicates whether to add new edges between graphs, defaults to True.
    :type edges_between: bool, optional
    :param edges_between_dist: Distance cutoff in Angstroms for adding edges between graphs, defaults to 4.5.
    :type edges_between_dist: float, optional
    :return: Tuple containing \n
        - node_feats (torch.FloatTensor): Features for each node in the combined graph, concatenated along the feature dimension.\n
        - edges (torch.LongTensor): Edges of combined graph in COO format, including edges from two input graphs and edges between them, if specified.\n
        - edge_weights (torch.FloatTensor): Concatenated edge features from two input graphs and edges between them, if specified.\n
        - node_pos (torch.FloatTensor): x-y-z coordinates of each node in combined graph.
    :rtype: Tuple
    """    
    node_feats1, edges1, edge_feats1, pos1 = graph1.x, graph1.edge_index, graph1.edge_attr, graph1.pos
    node_feats2, edges2, edge_feats2, pos2 = graph2.x, graph2.edge_index, graph2.edge_attr, graph2.pos

    dummy_node_feats1 = torch.zeros(pos1.shape[0], node_feats2.shape[1])
    dummy_node_feats2 = torch.zeros(pos2.shape[0], node_feats1.shape[1])
    node_feats1 = torch.cat((node_feats1, dummy_node_feats1), dim=1)
    node_feats2 = torch.cat((dummy_node_feats2, node_feats2), dim=1)

    edges2 += pos1.shape[0]

    node_pos = torch.cat((pos1, pos2), dim=0)
    node_feats = torch.cat((node_feats1, node_feats2), dim=0)

    if edges_between:
        edges_between, edge_feats_between = edges_between_graphs(pos1, pos2)
        edge_feats = torch.cat((edge_feats1, edge_feats2, edge_feats_between), dim=0)
        edges = torch.cat((edges1, edges2, edges_between), dim=1)
    else:
        edge_feats = torch.cat((edge_feats1, edge_feats2), dim=0)
        edges = torch.cat((edges1, edges2), dim=1)

    return node_feats, edges, edge_feats, node_pos


def edges_between_graphs(pos1, pos2, dist=4.5):
    """calculates edges between nodes in two separate graphs using a specified cutoff distance.

    :param pos1: x-y-z node coordinates from Graph 1
    :type pos1: torch.FloatTensor or numpy.ndarray
    :param pos2: x-y-z node coordinates from Graph 2
    :type pos2: torch.FloatTensor or numpy.ndarray
    :return: Tuple containing\n
        - edges (torch.LongTensor): Edges between two graphs, in COO format.\n
        - edge_weights (torch.FloatTensor): Edge weights between two graphs.\n
    :rtype: Tuple
    """    
    tree1 = ss.KDTree(pos1)
    tree2 = ss.KDTree(pos2)
    res = tree1.query_ball_tree(tree2, r=dist)
    edges = []
    edge_weights = []
    for i, contacts in enumerate(res):
        if len(contacts) == 0:
            continue
        for j in contacts:
            edges.append((i, j + pos1.shape[0]))
            edges.append((j + pos1.shape[0], i))
            d = 1.0 / (np.linalg.norm(pos1[i] - pos2[j]) + 1e-5)
            edge_weights.append(d)
            edge_weights.append(d)

    edges = torch.LongTensor(edges).t().contiguous()
    edge_weights = torch.FloatTensor(edge_weights).view(-1)
    return edges, edge_weights    


def adjust_graph_indices(graph):
    """Adjusts indices into graphs for concatenated multi-graph batches. Specifically, if each graph in the batch has a different selection index defined relative to that graph, the index is adjusted to be defined relative to the batch indexing.

    :param graph: Pytorch-geometric graph object representing a batch of graphs. Assumed to have a ``select_idx`` attribute set, specifying a node index for each graph
    :type graph: torch_geometric.data.Data
    :return: Same graph with selection indices adjusted
    :rtype: torch_geometric.data.Data
    """    
    batch_size = len(graph.n_nodes)
    total_n = 0
    for i in range(batch_size-1):
        n_nodes = graph.n_nodes[i].item()
        total_n += n_nodes
        graph.select_idx[i+1] += total_n
    return graph


# below functions are adapted from DeepChem repository:
def one_of_k_encoding(x, allowable_set):
    """Converts input to 1-hot encoding given a set of allowable values."""
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(x, allowable_set))
    return list(map(lambda s: x == s, allowable_set))


def one_of_k_encoding_unk(x, allowable_set):
    """Converts input to 1-hot encoding given a set of allowable values. Additionally maps inputs not in the allowable set to the last element."""
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))
