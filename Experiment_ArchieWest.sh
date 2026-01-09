#!/bin/bash
#SBATCH --export=ALL
#SBATCH --partition=gpu --gpus=1 --mem-per-cpu=48000
#SBATCH --account=moshfeghi-pmwc
#SBATCH --time=48:00:00
#SBATCH --mail-user=niall.mcguire@strath.ac.uk
#SBATCH --mail-type=END
#SBATCH --job-name=labram_true
#SBATCH --output=slurm-labram-%j.out

module purge
module load nvidia/sdk/23.3
module load anaconda/python-3.9.7/2021.11

# TRUE LaBraM: Channel-Wise Patching + Single-Channel CNN + Spatial Embeddings
# Reference: Jiang et al., "Large Brain Model", ICLR 2024
# WARNING: Computationally expensive (64 channels × 4 time patches = 256 patches/word)
python controller.py \
  --data_paths \
    /users/gxb18167/ECIR2026/SpatialTemporalDecompositionMultiDataset/Dataset/nieuwland_ict_pairs_RUNTIME_MASKING.npy \
    /users/gxb18167/ECIR2026/SpatialTemporalDecompositionMultiDataset/Dataset/alice_ict_pairs_RUNTIME_MASKING.npy \
    /users/gxb18167/WWW26/dataset/narrative_ict_pairs_RUNTIME_MASKING.npy/ \
    /users/gxb18167/WWW26/dataset/derco_ict_pairs_RUNTIME_MASKING.npy/ \
  --dataset_types nieuwland alice narrative derco \
  --pooling_strategy cls \
  --query_type eeg \
  --use_temporal_spatial_decomp \
  --decomp_level word \
  --no_lora \
  --eeg_arch transformer \
  --colbert_model_name colbert-ir/colbertv2.0 \
  --lr 1e-4 --patience 10 --epochs 100 --batch_size 32 --max_text_len 256 \
  --enable_multi_masking_validation --validation_masking_levels 90 \
  --multi_masking_frequency 50 --primary_masking_level 90 --training_masking_level 90 \
  --enable_test_evaluation --test_masking_levels 0 25 50 75 90 100
/opt/software/scripts/job_epilogue.sh