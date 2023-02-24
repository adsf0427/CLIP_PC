#!/bin/bash
#SBATCH --account=def-rjliao
#SBATCH --gres=gpu:a100:2        # Number of GPUs per node (specifying v100l gpu)
#SBATCH --nodes=1                 # Number of nodes
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=4        # CPU cores per MPI process
#SBATCH --mem=64G                 # memory per node
#SBATCH --time=0-12:00            # time (DD-HH:MM)
#SBATCH --output=./slurm_log/%x-%j.out
#SBATCH --mail-user=adsf_zsx@qq.com
#SBATCH --mail-type=ALL

module load python/3.8 cuda/11.4
source ~/zsx/bin/activate

cd $SLURM_TMPDIR
unzip -q /home/zyliang/scratch/ShapeNetCore.v2.PC15k.zip

cd /lustre07/scratch/zyliang/train-CLIP
srun python train_pc.py --model_name PC-B --root_dir ''$SLURM_TMPDIR/ShapeNetCore.v2.PC15k/'' --batch_size 4