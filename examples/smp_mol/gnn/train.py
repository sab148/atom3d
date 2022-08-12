import argparse
import logging
import os
import time
import datetime
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
# import seaborn as sns
import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from model import GNN_SMP
# from data import GNNTransformSMP
from atom3d.datasets import LMDBDataset, MolH5Dataset

def train_loop(model, loader, optimizer, device):
    model.train()

    loss_all = 0
    total = 0
    try :
        for data in tqdm(loader):
            data = data.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = F.mse_loss(output, data.y["Electron_Affinity"])
            loss.backward()
            loss_all += loss.item() * data.num_graphs
            total += data.num_graphs
            optimizer.step()
    except Exception as e:
        print(e)
        print(data)
    return loss_all / total

@torch.no_grad()
def test(model, loader, device):
    model.eval()
    loss_all = 0
    total = 0
    y_true = []
    y_pred = []
    for data in loader:
        data = data.to(device)
        output=model(data)
        loss = F.l1_loss(output, data.y["Electron_Affinity"])  # MAE
        loss_all += loss.item() * data.num_graphs
        total += data.num_graphs
        y_true.extend([x.item() for x in data.y["Electron_Affinity"]])
        y_pred.extend(output.tolist())
    return loss_all / total, y_true, y_pred


def save_weights(model, weight_dir):
    torch.save(model.state_dict(), weight_dir)

def train(args, device, log_dir, rep=None, test_mode=False):
    # logger = logging.getLogger('lba')
    # logger.basicConfig(filename=os.path.join(log_dir, f'train_{split}_cv{fold}.log'),level=logging.INFO)

    h5_file = "/p/home/jusers/benassou1/juwels/hai_drug_qm/atom3d/examples/smp_mol/data/qm/aeneas/h5/qm.hdf5"

    train_idx = "/p/home/jusers/benassou1/juwels/hai_drug_qm/atom3d/examples/smp_mol/data/qm/aeneas/mol_split/train.txt"
    val_idx = "/p/home/jusers/benassou1/juwels/hai_drug_qm/atom3d/examples/smp_mol/data/qm/aeneas/mol_split/val.txt"
    test_idx = "/p/home/jusers/benassou1/juwels/hai_drug_qm/atom3d/examples/smp_mol/data/qm/aeneas/mol_split/test.txt"

    train_dataset = MolH5Dataset(h5_file, train_idx, transform=None)
    val_dataset = MolH5Dataset(h5_file, val_idx, transform=None)
    test_dataset = MolH5Dataset(h5_file, test_idx, transform=None)
    

    train_loader = DataLoader(train_dataset, args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, args.batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, args.batch_size, shuffle=False, num_workers=4)

    print("data loaded")

    for data in train_loader:     
        # print(data)   
        num_features = data.num_features
        break

    model = GNN_SMP(num_features, dim=args.hidden_dim).to(device)
    model.to(device)

    best_val_loss = 999


    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                           factor=0.7, patience=3,
                                                           min_lr=0.00001)

    # for epoch in tqdm(range(1, args.num_epochs+1)):
    #     start = time.time()
    #     train_loss = train_loop(model, train_loader, optimizer, device)
    #     print('validating...')
    #     val_loss,  _,_ = test(model, val_loader, device)
    #     scheduler.step(val_loss)
    #     if val_loss < best_val_loss:
    #         print("saving ...")
    #         torch.save({
    #             'epoch': epoch,
    #             'model_state_dict': model.state_dict(),
    #             'optimizer_state_dict': optimizer.state_dict(),
    #             'loss': train_loss,
    #             }, os.path.join(log_dir, f'best_weights_rep{rep}.pt'))
    #         best_val_loss = val_loss
    #     elapsed = (time.time() - start)
    #     print('Epoch: {:03d}, Time: {:.3f} s'.format(epoch, elapsed))
    #     print('\tTrain Loss: {:.7f}, Val MAE: {:.7f}'.format(train_loss, val_loss))

    if test_mode:
        train_file = os.path.join(log_dir, f'smp-rep{rep}.best.train.pt')
        val_file = os.path.join(log_dir, f'smp-rep{rep}.best.val.pt')
        test_file = os.path.join(log_dir, f'smp-rep{rep}.best.test.pt')
        cpt = torch.load(os.path.join(log_dir, f'best_weights_rep{rep}.pt'))
        model.load_state_dict(cpt['model_state_dict'])
        _, y_true_train, y_pred_train = test(model, train_loader, device)
        torch.save({'targets':y_true_train, 'predictions':y_pred_train}, train_file)
        _, y_true_val, y_pred_val = test(model, val_loader, device)
        torch.save({'targets':y_true_val, 'predictions':y_pred_val}, val_file)
        mae, y_true_test, y_pred_test = test(model, test_loader, device)
        print(f'\tTest MAE {mae}')
        torch.save({'targets':y_true_test, 'predictions':y_pred_test}, test_file)



if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str)
    parser.add_argument('--target_name', type=str)
    parser.add_argument('--mode', type=str, default='test')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--hidden_dim', type=int, default=64)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--log_dir', type=str, default=None)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log_dir = args.log_dir


    if args.mode == 'train':
        if log_dir is None:
            now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
            log_dir = os.path.join('logs', now)
        else:
            log_dir = os.path.join('logs', log_dir)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        train(args, device, log_dir)
        
    elif args.mode == 'test':
        for rep, seed in enumerate(np.random.randint(0, 1000, size=3)):
            print('seed:', seed)
            log_dir = os.path.join('logs', f'smp_test_{args.target_name}')
            if not os.path.exists(log_dir):
                os.makedirs(log_dir)
            np.random.seed(seed)
            torch.manual_seed(seed)
            train(args, device, log_dir, rep, test_mode=True)
