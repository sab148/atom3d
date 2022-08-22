import os

import dotenv as de
import numpy as np
import pandas as pd

from atom3d.datasets import LMDBDataset
from atom3d.util.voxelize import dotdict, get_center, gen_rot_matrix, get_grid
from torch.utils.data import DataLoader


de.load_dotenv(de.find_dotenv(usecwd=True))


class CNN3D_TransformLBA(object):
    def __init__(self, element_map, random_seed=None, **kwargs):
        self.random_seed = random_seed
        
        element_map = {1: '2C', 2: '3C', 3: 'C', 4: 'C*', 5: 'C8', 6: 'CA', 7: 'CB', 8: 'CC', 9: 'CN', 10: 'CO', 11: 'CR', 12: 'CT', 13: 'CW', 14: 'CX', 15: 'H', 16: 'H1', 17: 'H4', 18: 'H5', 19: 'HA', 20: 'HC', 21: 'HO', 22: 'HP', 23: 'HS', 24: 'N', 25: 'N2', 26: 'N3', 27: 'NA', 28: 'NB', 29: 'O', 30: 'O2', 31: 'OH', 32: 'S', 33: 'SH', 34: 'br', 35: 'c', 36: 'c1', 37: 'c2', 38: 'c3', 39: 'ca', 40: 'cc', 41: 'cd', 42: 'ce', 43: 'cf', 44: 'cg', 45: 'ch', 46: 'cl', 47: 'cp', 48: 'cq', 49: 'cs', 50: 'cu', 51: 'cx', 52: 'cy', 53: 'cz', 54: 'f', 55: 'h1', 56: 'h2', 57: 'h3', 58: 'h4', 59: 'h5', 60: 'ha', 61: 'hc', 62: 'hn', 63: 'ho', 64: 'hp', 65: 'hs', 66: 'hx', 67: 'i', 68: 'n', 69: 'n1', 70: 'n2', 71: 'n3', 72: 'n4', 73: 'n7', 74: 'n8', 75: 'na', 76: 'nb', 77: 'nc', 78: 'nd', 79: 'ne', 80: 'nf', 81: 'nh', 82: 'ni', 83: 'nj', 84: 'nk', 85: 'nl', 86: 'nm', 87: 'nn', 88: 'no', 89: 'nq', 90: 'ns', 91: 'nt', 92: 'nu', 93: 'nv', 94: 'nx', 95: 'ny', 96: 'nz', 97: 'o', 98: 'oh', 99: 'op', 100: 'oq', 101: 'os', 102: 'p5', 103: 'py', 104: 's', 105: 's4', 106: 's6', 107: 'sh', 108: 'ss', 109: 'sx', 110: 'sy'}
        self.grid_config =  dotdict({
            # Mapping from elements to position in channel dimension.
            #'element_mapping': {
            #    'H': 0,
            #    'C': 1,
            #    'O': 2,
            #    'N': 3,
            #    'F': 4,
            #},
            'element_mapping': element_map,
            # Radius of the grids to generate, in angstroms.
            'radius': 20.0,
            # Resolution of each voxel, in angstroms.
            'resolution': 1.0,
            # Number of directions to apply for data augmentation.
            'num_directions': 20,
            # Number of rolls to apply for data augmentation.
            'num_rolls': 20,
        })
        # Update grid configs as necessary
        self.grid_config.update(kwargs)

    def _voxelize(self, atoms_pocket, atoms_ligand):
        # Use center of ligand as subgrid center
        ligand_pos = atoms_ligand[['x', 'y', 'z']].astype(np.float32)
        ligand_center = get_center(ligand_pos)
        # Generate random rotation matrix
        rot_mat = gen_rot_matrix(self.grid_config, random_seed=self.random_seed)
        # Transform protein/ligand into voxel grids and rotate
        grid = get_grid(pd.concat([atoms_pocket, atoms_ligand]),
                        ligand_center, config=self.grid_config, rot_mat=rot_mat)
        # Last dimension is atom channel, so we need to move it to the front
        # per pytroch style
        grid = np.moveaxis(grid, -1, 0)
        return grid

    def __call__(self, item):
        # Transform protein/ligand into voxel grids.
        # Apply random rotation matrix.
        transformed = {
            'feature': self._voxelize(item['atoms_protein'], item['atoms_ligand']),
            'label': item['scores'],
            'id': item['id']
        }
        return transformed


if __name__=="__main__":
    dataset_path = os.path.join(os.environ['LBA_DATA'], 'val')
    dataset = LMDBDataset(dataset_path, transform=CNN3D_TransformLBA(radius=10.0))
    dataloader = DataLoader(dataset, batch_size=8, shuffle=False)

    for item in dataloader:
        print('feature shape:', item['feature'].shape)
        print('label:', item['label'])
        break
