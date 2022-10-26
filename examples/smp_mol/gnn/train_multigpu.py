import argparse
import logging
import os
import time
import datetime
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import math
# import seaborn as sns
import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from model import GNN_SMP
from data import GNNTransformSMP
from atom3d.datasets import LMDBDataset, MolH5Dataset
from module import SMPLitModule
from pytorch_lightning import Trainer
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar
from sklearn.metrics import mean_absolute_error
from pytorch_lightning.loggers import TensorBoardLogger

def train(args, device, log_dir, rep=None, test_mode=False):
    # logger = logging.getLogger('lba')
    # logger.basicConfig(filename=os.path.join(log_dir, f'train_{split}_cv{fold}.log'),level=logging.INFO)

    print(args)
    pl.seed_everything(12345, workers=True)
    
    h5_file = "/dev/shm/qm.hdf5"

    train_idx = "/dev/shm/train.txt"
    val_idx = "/dev/shm/val.txt"
    test_idx = "/dev/shm/test.txt"

    
    train_dataset = MolH5Dataset(h5_file, train_idx, target=args.target_name, transform=GNNTransformSMP(args.target_name))
    val_dataset = MolH5Dataset(h5_file, val_idx, target=args.target_name, transform=GNNTransformSMP(args.target_name))
    test_dataset = MolH5Dataset(h5_file, test_idx, target=args.target_name, transform=GNNTransformSMP(args.target_name))
    
    train_dataloader = DataLoader(train_dataset, args.batch_size, shuffle=True, num_workers=8)
    val_dataloader = DataLoader(val_dataset, args.batch_size, shuffle=False, num_workers=8)
    test_dataloader = DataLoader(test_dataset, args.batch_size, shuffle=False, num_workers=8)

    print("data loaded")
    for data in train_dataloader:     
        # print(data)   
        num_features = data.num_features
        break
        
    print("number gpus : ", torch.cuda.device_count())
    model = GNN_SMP(num_features, dim=args.hidden_dim)
    module = SMPLitModule(model)
    print("model",next(module.parameters()).is_cuda) 
    bar = TQDMProgressBar(refresh_rate = 10)
    logger = TensorBoardLogger(args.tensor_board, name="00")
    checkpoint_callback = ModelCheckpoint(monitor="val/loss", dirpath=args.model_ckpt, filename='best', verbose=True )
    # trainer = Trainer(devices=2, accelerator="gpu", strategy="ddp")
    trainer = Trainer(devices=1, accelerator="gpu", callbacks=[checkpoint_callback, bar], max_epochs=args.num_epochs, profiler="simple",logger=logger, gradient_clip_val=1.5)
    
    print("world size : ", trainer.world_size)
    trainer.fit(module, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
    
    train_file = os.path.join(log_dir, f'smp-rep{rep}.best.train.pt')
    val_file = os.path.join(log_dir, f'smp-rep{rep}.best.val.pt')
    test_file = os.path.join(log_dir, f'smp-rep{rep}.best.test.pt')

    trainer2 = Trainer(devices=1, accelerator="auto", resume_from_checkpoint=args.model_ckpt+"best.ckpt")
    
    train_dataloader = DataLoader(train_dataset, args.batch_size, shuffle=False, num_workers=0)
    pred_train = trainer2.predict(module, dataloaders=train_dataloader)
    print(f'\tTrain MAE {math.sqrt(mean_absolute_error(pred_train[0]["preds"], pred_train[1]["targets"]))}')
    torch.save({'targets':pred_train[0], 'predictions':pred_train[1]}, train_file)
    print("val prediction")
    pred_val = trainer2.predict(module, dataloaders=val_dataloader)

    print(f'\tVal MAE {math.sqrt(mean_absolute_error(pred_val[0]["preds"], pred_val[1]["targets"]))}')
    torch.save({'targets':pred_val[0], 'predictions':pred_val[1]}, val_file)
    print("test prediction")
    pred_test = trainer2.predict(module, dataloaders=test_dataloader)
    print(f'\tTest MAE {math.sqrt(mean_absolute_error(pred_test[0]["preds"], pred_test[1]["targets"]))}')
    torch.save({'targets':pred_test[0], 'predictions':pred_test[1]}, test_file)
    

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str)
    parser.add_argument('--target_name', type=str, default="Ionization_Potential")
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--hidden_dim', type=int, default=64)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--log_dir', type=str, default='/p/project/hai_drug_qm/atom3d/examples/smp_mol/gnn/logs')
    parser.add_argument('--tensor_board', type=str, default="/p/project/hai_drug_qm/atom3d/examples/smp_mol/gnn/tensor_board/")
    parser.add_argument('--model_ckpt', type=str, default="/p/project/hai_drug_qm/atom3d/examples/smp_mol/gnn/model_ckpts/")
    args = parser.parse_args()
    log_dir = args.log_dir

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if log_dir is None:
        now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        log_dir = os.path.join('logs', now)
    else:
        log_dir = os.path.join('logs', log_dir)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    train(args, device, log_dir)
