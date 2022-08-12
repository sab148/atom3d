#!/usr/bin/env bash
# `bash -x` for detailed Shell debugging

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --output=/p/home/jusers/benassou1/juwels/hai_drug_qm/atom3d/examples/smp_mol/outputs/electron_affinity/12-08-2022-rslt-gnn.out
#SBATCH --error=/p/home/jusers/benassou1/juwels/hai_drug_qm/atom3d/examples/smp_mol/outputs/electron_affinity/12-08-2022-rslt-gnn.err
#SBATCH --time=02:00:00
#SBATCH --partition=develbooster
#SBATCH --gres=gpu:4
#SBATCH --account=hai_drug_qm

source ../../sc_venv_template/activate.sh
CUDA_VISIBLE_DEVICES = 0

srun python gnn/train.py
