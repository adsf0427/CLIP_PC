salloc --nodes=1 --time=2:0:0 --ntasks=1 --account=def-rjliao --cpus-per-task=32 --mem=32G --mail-user=adsf_zsx@qq.com --mail-type=ALL
salloc --nodes=1 --time=2:0:0 --ntasks=1 --account=def-rjliao --gres=gpu:a100:1 --cpus-per-task=24 --mem=64G --mail-user=adsf_zsx@qq.com --mail-type=ALL
module load python/3.8 cuda/11.4
source ~/zsx/bin/activate 


cd $SLURM_TMPDIR
unzip -q /home/zyliang/scratch/ShapeNetCore.v2.PC15k.zip

cd /lustre07/scratch/zyliang/train-CLIP
python train_pc.py --model_name PC-B --root_dir '/lustre07/scratch/zyliang/3dGeneration/ShapeNetCore.v2.PC15k' --batch_size 4

export NCCL_BLOCKING_WAIT=1

