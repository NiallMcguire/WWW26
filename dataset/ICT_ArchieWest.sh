#!/bin/bash
#=================================================================
#
# Job script for running Narrative ICT pair generation v3
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
#SBATCH --time=03:00:00
#
# Email notifications
#SBATCH --mail-user=niall.mcguire@strath.ac.uk
#SBATCH --mail-type=END
#
# Job name
#SBATCH --job-name=narrative_v3_gen
#
# Output file
#SBATCH --output=slurm_narrative_v3-%j.out
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
echo "Starting Narrative ICT Pair Generation v3"
echo "Date: $(date)"
echo "Working directory: $(pwd)"
echo "Python: $(which python)"
echo "Python version: $(python --version)"
echo "=========================================="

# Check if data directories exist
echo "Verifying Narrative dataset directories..."

if [ ! -d "/users/gxb18167/Natural Speech/Stimuli/Text" ]; then
    echo "ERROR: Text directory not found!"
    echo "Expected: /users/gxb18167/Natural Speech/Stimuli/Text"
    exit 1
fi

if [ ! -d "/users/gxb18167/Natural Speech/EEG" ]; then
    echo "ERROR: EEG directory not found!"
    echo "Expected: /users/gxb18167/Natural Speech/EEG"
    exit 1
fi

# Count available data
echo "Checking available data..."
num_text_files=$(find "/users/gxb18167/Natural Speech/Stimuli/Text" -name "*.mat" | wc -l)
num_subjects=$(ls -d "/users/gxb18167/Natural Speech/EEG"/*/ 2>/dev/null | wc -l)
echo "Found $num_text_files text/stimuli files"
echo "Found $num_subjects subject directories"
echo "=========================================="

# Run the Narrative ICT reader v3
echo "Running narrative_ict_reader_v3.py..."
echo "⚙️  Expected processing time: ~30-60 minutes"
echo "⚙️  Creating 3D word-level EEG arrays"
echo "⚙️  Runtime masking support enabled"
echo "=========================================="

python narrative_ict_reader_v2.py

# Check if output was created
if [ -f "narrative_ict_pairs_RUNTIME_MASKING.npy" ]; then
    echo "=========================================="
    echo "SUCCESS: ICT pairs generated!"
    echo "Output file: narrative_ict_pairs_RUNTIME_MASKING.npy"
    ls -lh narrative_ict_pairs_RUNTIME_MASKING.npy
    ls -lh narrative_ict_pairs_RUNTIME_MASKING.json
    echo "=========================================="

    # Quick statistics
    echo "📊 Dataset Statistics:"
    python -c "
import numpy as np
data = np.load('narrative_ict_pairs_RUNTIME_MASKING.npy', allow_pickle=True).item()
pairs = data['ict_pairs']
print(f'Total ICT pairs: {len(pairs)}')
print(f'Participants: {data[\"metadata\"][\"subjects_processed\"]}')
print(f'Sentences: {data[\"metadata\"][\"sentences_loaded\"]}')

# Check first pair structure
pair = pairs[0]
print(f'Query EEG shape: {pair[\"query_eeg\"].shape}')
print(f'Doc EEG shape: {pair[\"doc_eeg\"].shape}')
print(f'Has full_sentence_text: {\"full_sentence_text\" in pair}')
print(f'Runtime masking: {data[\"metadata\"][\"ict_generation_params\"][\"supports_runtime_masking\"]}')
"

    echo "=========================================="
    echo "✅ Verification Checklist:"
    echo "=========================================="

    # Verify 3D arrays
    python -c "
import numpy as np
data = np.load('narrative_ict_pairs_RUNTIME_MASKING.npy', allow_pickle=True).item()
pair = data['ict_pairs'][0]

# Check 3D arrays
query_is_3d = len(pair['query_eeg'].shape) == 3
doc_is_3d = len(pair['doc_eeg'].shape) == 3
has_full_sentence = 'full_sentence_text' in pair and 'full_sentence_words' in pair
no_is_masked = 'is_masked' not in pair
runtime_masking = data['metadata']['ict_generation_params']['supports_runtime_masking']

print(f'✅ Query EEG is 3D: {query_is_3d}' if query_is_3d else f'❌ Query EEG is NOT 3D')
print(f'✅ Doc EEG is 3D: {doc_is_3d}' if doc_is_3d else f'❌ Doc EEG is NOT 3D')
print(f'✅ Has full_sentence fields: {has_full_sentence}' if has_full_sentence else f'❌ Missing full_sentence fields')
print(f'✅ No is_masked field: {no_is_masked}' if no_is_masked else f'❌ Still has is_masked field')
print(f'✅ Runtime masking enabled: {runtime_masking}' if runtime_masking else f'❌ Runtime masking NOT enabled')

all_checks = query_is_3d and doc_is_3d and has_full_sentence and no_is_masked and runtime_masking
print()
if all_checks:
    print('🎉 ALL CHECKS PASSED - Ready for training!')
else:
    print('⚠️  SOME CHECKS FAILED - Review output above')
"

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