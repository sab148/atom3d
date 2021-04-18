import argparse
import datetime
import json
import os
import time
import tqdm

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from atom3d.datasets import LMDBDataset
from scipy.stats import spearmanr

from model import CNN3D_LBA
from data import CNN3D_TransformLBA


# Construct model
def conv_model(in_channels, spatial_size, args):
    num_conv = args.num_conv
    conv_filters = [32 * (2**n) for n in range(num_conv)]
    conv_kernel_size = 3
    max_pool_positions = [0, 1]*int((num_conv+1)/2)
    max_pool_sizes = [2]*num_conv
    max_pool_strides = [2]*num_conv
    fc_units = [512]

    model = CNN3D_LBA(
        in_channels, spatial_size,
        args.conv_drop_rate,
        args.fc_drop_rate,
        conv_filters, conv_kernel_size,
        max_pool_positions,
        max_pool_sizes, max_pool_strides,
        fc_units,
        batch_norm=args.batch_norm,
        dropout=not args.no_dropout)
    return model


def train_loop(model, loader, optimizer, device):
    model.train()

    losses = []
    epoch_loss = 0
    progress_format = 'train loss: {:6.6f}'
    with tqdm.tqdm(total=len(loader), desc=progress_format.format(0)) as t:
        for i, data in enumerate(loader):
            feature = data['feature'].to(device).to(torch.float32)
            label = data['label'].to(device).to(torch.float32)
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            output = model(feature)
            batch_losses = F.mse_loss(output, label, reduction='none')
            batch_losses_mean = batch_losses.mean()
            batch_losses_mean.backward()
            optimizer.step()
            # stats
            epoch_loss += (batch_losses_mean.item() - epoch_loss) / float(i + 1)
            losses.extend(batch_losses.tolist())
            t.set_description(progress_format.format(np.sqrt(epoch_loss)))
            t.update(1)

    return np.sqrt(np.mean(losses))


@torch.no_grad()
def test(model, loader, device):
    model.eval()

    losses = []

    ids = []
    y_true = []
    y_pred = []

    for data in loader:
        feature = data['feature'].to(device).to(torch.float32)
        label = data['label'].to(device).to(torch.float32)
        output = model(feature)
        batch_losses = F.mse_loss(output, label, reduction='none')
        losses.extend(batch_losses.tolist())
        ids.extend(data['id'])
        y_true.extend(label.tolist())
        y_pred.extend(output.tolist())

    results_df = pd.DataFrame(
        np.array([ids, y_true, y_pred]).T,
        columns=['structure', 'true', 'pred'],
        )
    r_p = np.corrcoef(y_true, y_pred)[0,1]
    r_s = spearmanr(y_true, y_pred)[0]

    return np.sqrt(np.mean(losses)), r_p, r_s, results_df


def plot_corr(y_true, y_pred, plot_dir):
    plt.clf()
    sns.scatterplot(y_true, y_pred)
    plt.xlabel('Actual -log(K)')
    plt.ylabel('Predicted -log(K)')
    plt.savefig(plot_dir)


def save_weights(model, weight_dir):
    torch.save(model.state_dict(), weight_dir)


def train(args, device, test_mode=False):
    print("Training model with config:")
    print(str(json.dumps(args.__dict__, indent=4)) + "\n")

    # Save config
    with open(os.path.join(args.output_dir, 'config.json'), 'w') as f:
        json.dump(args.__dict__, f, indent=4)

    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)

    train_dataset = LMDBDataset(os.path.join(args.data_dir, 'train'), transform=CNN3D_TransformLBA())
    val_dataset = LMDBDataset(os.path.join(args.data_dir, 'val'), transform=CNN3D_TransformLBA())
    test_dataset = LMDBDataset(os.path.join(args.data_dir, 'test'), transform=CNN3D_TransformLBA())

    train_loader = DataLoader(train_dataset, args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, args.batch_size, shuffle=False)

    for data in train_loader:
        in_channels, spatial_size = data['feature'].size()[1:3]
        print('num channels: {:}, spatial size: {:}'.format(in_channels, spatial_size))
        break

    model = conv_model(in_channels, spatial_size, args)
    print(model)
    model.to(device)

    best_val_loss = np.Inf
    best_rp = 0
    best_rs = 0

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    for epoch in range(1, args.num_epochs+1):
        start = time.time()
        train_loss = train_loop(model, train_loader, optimizer, device)
        val_loss, r_p, r_s, val_df = test(model, val_loader, device)
        if val_loss < best_val_loss:
            print(f"Save model at epoch {epoch:03d}, val_loss: {val_loss:.4f}, "
                  f"best_val_loss: {best_val_loss:.4f}")
            save_weights(model, os.path.join(args.output_dir, f'best_weights.pt'))
            #plot_corr(val_df['true'], val_df['pred'],
            #          os.path.join(args.output_dir, f'corr_val-epoch_{epoch:}.png'))
            best_val_loss = val_loss
            best_rp = r_p
            best_rs = r_s
        elapsed = (time.time() - start)
        print('Epoch {:03d} finished in : {:.3f} s'.format(epoch, elapsed))
        print('\tTrain RMSE: {:.7f}, Val RMSE: {:.7f}, Pearson R: {:.7f}, Spearman R: {:.7f}'.format(
            train_loss, val_loss, r_p, r_s))

    if test_mode:
        model.load_state_dict(torch.load(os.path.join(args.output_dir, f'best_weights.pt')))
        rmse, pearson, spearman, test_df = test(model, test_loader, device)
        test_df.to_pickle(os.path.join(args.output_dir, 'test_results.pkl'))
        #plot_corr(test_df['true'], test_df['pred'],
        #          os.path.join(args.output_dir, f'corr_test.png'))
        print('Test RMSE: {:.7f}, Pearson R: {:.7f}, Spearman R: {:.7f}'.format(
            rmse, pearson, spearman))
        test_file = os.path.join(args.output_dir, f'test_results.txt')
        with open(test_file, 'a+') as out:
            out.write('{}\t{:.7f}\t{:.7f}\t{:.7f}\n'.format(
                args.random_seed, rmse, pearson, spearman))

    return best_val_loss, best_rp, best_rs


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default=os.environ['LBA_DATA'])
    parser.add_argument('--mode', type=str, default='test',
                        choices=['train', 'test'])
    parser.add_argument('--output_dir', type=str, default=os.environ['LOG_DIR'])
    parser.add_argument('--unobserved', action='store_true', default=False)

    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--conv_drop_rate', type=float, default=0.1)
    parser.add_argument('--fc_drop_rate', type=float, default=0.25)
    parser.add_argument('--num_epochs', type=int, default=50)

    parser.add_argument('--num_conv', type=int, default=4)
    parser.add_argument('--batch_norm', action='store_true', default=False)
    parser.add_argument('--no_dropout', action='store_true', default=False)

    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--random_seed', type=int, default=int(np.random.randint(1, 10e6)))

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Set up output dir
    args.output_dir = os.path.join(args.output_dir, 'lba')
    assert args.output_dir != None
    if args.unobserved:
        args.output_dir = os.path.join(args.output_dir, 'None')
        os.makedirs(args.output_dir, exist_ok=True)
    else:
        num = 0
        while True:
            dirpath = os.path.join(args.output_dir, str(num))
            if os.path.exists(dirpath):
                num += 1
            else:
                args.output_dir = dirpath
                print('Creating output directory {:}'.format(args.output_dir))
                os.makedirs(args.output_dir)
                break

    print(f"Running mode {args.mode:} with seed {args.random_seed:} "
          f"and output dir {args.output_dir}")
    train(args, device, args.mode=='test')
