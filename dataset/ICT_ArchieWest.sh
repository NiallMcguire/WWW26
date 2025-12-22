#!/bin/bash
#=================================================================
#
# Job script for running DERCo ICT pair generation
#
#=================================================================

#======================================================
# Propogate environment variables to the compute node
#SBATCH --export=ALL
#
# Run in the standard partition with high memory
#SBATCH --partition=standard
#SBATCH --mem-per-cpu=180000
#
# Specify project account
#SBATCH --account=moshfeghi-pmwc
#
# Specify (hard) runtime (HH:MM:SS)
#SBATCH --time=02:00:00
#
# Email notifications
#SBATCH --mail-user=niall.mcguire@strath.ac.uk
#SBATCH --mail-type=END
#
# Job name
#SBATCH --job-name=derco_ict_gen
#
# Output file
#SBATCH --output=slurm_derco-%j.out
#======================================================

module purge
module load nvidia/sdk/23.3
module load anaconda/python-3.9.7/2021.11

#=========================================================
# Prologue script to record job details
# Do not change the line below
#=====================
/opt/software/scripts/job_prologue.sh
#------------------------------------------

# Activate virtual environment
source ~/venv_clean/bin/activate

# Navigate to dataset directory
cd /users/gxb18167/WWW26/dataset

# Print environment info
echo "=========================================="
echo "Starting DERCo ICT Pair Generation"
echo "Date: $(date)"
echo "Working directory: $(pwd)"
echo "Python: $(which python)"
echo "Python version: $(python --version)"
echo "=========================================="

# Check if data directory exists
if [ ! -d "DERCo_preprocessed_rsvp" ]; then
    echo "ERROR: DERCo_preprocessed_rsvp directory not found!"
    exit 1
fi

# Count downloaded files
echo "Verifying downloaded data..."
num_participants=$(ls -d DERCo_preprocessed_rsvp/*/ 2>/dev/null | wc -l)
num_fif_files=$(find DERCo_preprocessed_rsvp -name "*.fif" 2>/dev/null | wc -l)
echo "Found $num_participants participants"
echo "Found $num_fif_files .fif files"
echo "=========================================="

# Run the DERCo ICT reader
echo "Running derco_ict_reader.py..."
python derco_ict_reader.py

# Check if output was created
if [ -f "derco_ict_pairs_RUNTIME_MASKING.npy" ]; then
    echo "=========================================="
    echo "SUCCESS: ICT pairs generated!"
    echo "Output file: derco_ict_pairs_RUNTIME_MASKING.npy"
    ls -lh derco_ict_pairs_RUNTIME_MASKING.npy
    ls -lh derco_ict_pairs_RUNTIME_MASKING.json
    echo "=========================================="

    # Quick statistics
    python -c "import numpy as np; d=np.load('derco_ict_pairs_RUNTIME_MASKING.npy', allow_pickle=True).item(); print(f'Total ICT pairs: {len(d[\"ict_pairs\"])}'); print(f'Participants: {d[\"metadata\"][\"participants_processed\"]}')"
else
    echo "=========================================="
    echo "ERROR: Output file not created!"
    echo "Check the log above for errors"
    echo "=========================================="
    exit 1
fi

echo "=========================================="
echo "Job completed: $(date)"
echo "=========================================="

# Do not change the line below
#=========================================================
/opt/software/scripts/job_epilogue.sh
#----------------------------------------------------------