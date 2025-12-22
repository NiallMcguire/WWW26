#!/bin/bash
#SBATCH --export=ALL
#SBATCH --partition=gpu --gpus=1 --mem-per-cpu=48000
#SBATCH --account=moshfeghi-pmwc
#SBATCH --time=20:00:00
#SBATCH --mail-user=niall.mcguire@strath.ac.uk
#SBATCH --mail-type=END
#SBATCH --job-name=stbpr_3datasets
#SBATCH --output=slurm-stbpr-3datasets-%j.out

module purge
module load nvidia/sdk/23.3
module load anaconda/python-3.9.7/2021.11

# Run with all 3 datasets: Nieuwland (visual) + Alice (auditory) + DERCo (visual)
python controller.py \
  --data_paths \
    /users/gxb18167/ECIR2026/SpatialTemporalDecompositionMultiDataset/Dataset/nieuwland_ict_pairs_RUNTIME_MASKING.npy \
    /users/gxb18167/ECIR2026/SpatialTemporalDecompositionMultiDataset/Dataset/alice_ict_pairs_RUNTIME_MASKING.npy \
    /users/gxb18167/WWW26/dataset/derco_ict_pairs_RUNTIME_MASKING.npy \
  --dataset_types nieuwland alice derco \
  --decomp_level sequence \
  --pooling_strategy mean \
  --query_type eeg \
  --no_lora \
  --eeg_arch transformer \
  --colbert_model_name colbert-ir/colbertv2.0 \
  --lr 1e-4 --patience 3 --epochs 100 --batch_size 32 --max_text_len 256 \
  --enable_multi_masking_validation --validation_masking_levels 90 \
  --multi_masking_frequency 50 --primary_masking_level 90 --training_masking_level 90 \
  --enable_test_evaluation --test_masking_levels 0 25 50 75 90 100

/opt/software/scripts/job_epilogue.sh