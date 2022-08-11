# pytorch imports
import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

# atom3d imports
import atom3d.datasets.datasets as da
import atom3d.util.graph as gr
import atom3d.util.transforms as tr
from atom3d.models.gnn import GCN
from atom3d.models.mlp import MLP

# define training hyperparameters
learning_rate=1e-4
epochs = 5
feat_dim = 128
out_dim = 1

# Load dataset (with transform to convert dataframes to graphs) and initialize dataloader
#tr.GraphTransform(atom_key='atoms_protein', label_key='scores')
PATH_TO_DATA = "/p/project/hai_drug_qm/Dataset/paris/DB/qm.hdf5"
dataset = da.load_dataset(PATH_TO_DATA, 'molh5', transform=None)
dataloader = DataLoader(dataset, batch_size=8, shuffle=False)

# get number of input features from first graph
for batch in dataloader:
    in_dim = batch["atoms"].num_features
    print(batch)
    break

# GCN feature extraction module
feat_model = GCN(in_dim, feat_dim)
# Feed-forward output module
out_model = MLP(feat_dim, [64], out_dim)

# define optimizer and criterion
params = [x for x in feat_model.parameters()] + [x for x in out_model.parameters()]
optimizer = torch.optim.Adam(params, lr=1e-4)
criterion = nn.BCEWithLogitsLoss()

# Training loop
for epoch in range(epochs):
    print("epoch = ", epoch)
    load = 0
    for batch in dataloader:
        load += 1
        print("iter = ",load,'/',len(dataloader))

        # labels need to be float for BCE loss
        labels = batch.y['neglog_aff'].float()
        # calculate 128-dim features
        feats = feat_model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        # calculate predictions
        out = out_model(feats)
        
        # compute loss and backprop
        
        loss = criterion(out.view(-1), labels)
        
        loss.backward()
        
        optimizer.step()
        
    print('Epoch {}: train loss {}'.format(epoch, loss))


