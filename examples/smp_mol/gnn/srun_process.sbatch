#!/bin/bash -x

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --output=/p/project/hai_drug_qm/atom3d/examples/smp_mol/gnn/outputs/Ionization_Potential/%j-gnn.out
#SBATCH --error=/p/project/hai_drug_qm/atom3d/examples/smp_mol/gnn/outputs/Ionization_Potential/%j-gnn.err
#SBATCH --time=00:30:00
#SBATCH --partition=develbooster
#SBATCH --gres=gpu:1
#SBATCH --account=hai_md_proli

source $HOME/.bashrc

cp -r /p/project/hai_drug_qm/atom3d/examples/smp_mol/data/qm/aeneas/h5/qm.hdf5 /dev/shm/

cp -r /p/project/hai_drug_qm/atom3d/examples/smp_mol/data/qm/aeneas/mol_split2/train.txt /dev/shm/
cp -r /p/project/hai_drug_qm/atom3d/examples/smp_mol/data/qm/aeneas/mol_split2/val.txt /dev/shm/
cp -r /p/project/hai_drug_qm/atom3d/examples/smp_mol/data/qm/aeneas/mol_split2/test.txt /dev/shm/

source ../../../sc_venv_template/activate.sh
#export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 

srun python -u train_multigpu.py --target_name "Ionization_Potential" --tensor_board "/p/project/hai_drug_qm/atom3d/examples/smp_mol/gnn/tensor_board/Ionization_Potential/26-10-2022/" --model_ckpt "/p/project/hai_drug_qm/atom3d/examples/smp_mol/gnn/model_ckpts/Ionization_Potential/26-10-2022/"
