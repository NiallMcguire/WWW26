#!/usr/bin/env python3
"""
UPDATED Narrative EEG Reader with RUNTIME MASKING SUPPORT (v3)
✅ Now uses 3D word-level EEG arrays [num_words, time_samples, channels]
✅ Matches Alice/DERCo format exactly
✅ Generates unmasked ICT pairs with metadata for runtime masking
✅ Full reproducibility with comprehensive metadata tracking
✅ Compatible with multi-masking validation (0%, 25%, 50%, 75%, 90%, 100%)
"""

import os
import re
import numpy as np
import scipy.io as sio
import torch
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
from tqdm import tqdm
from scipy import signal
import random
from dataclasses import dataclass
import json
from datetime import datetime
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

# Set global seeds for reproducibility
GLOBAL_SEED = 42
random.seed(GLOBAL_SEED)
np.random.seed(GLOBAL_SEED)
torch.manual_seed(GLOBAL_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(GLOBAL_SEED)


@dataclass
class ICTPair:
    """Data class for ICT query-document pairs with RUNTIME MASKING SUPPORT"""
    query_text: str
    query_eeg: np.ndarray  # 3D array: [num_words, time_samples, channels]
    query_words: List[str]
    doc_text: str  # ALWAYS UNMASKED - full sentence
    doc_eeg: np.ndarray  # ALWAYS UNMASKED - 3D array: [num_words, time_samples, channels]
    doc_words: List[str]  # ALWAYS UNMASKED - full sentence words
    participant_id: str
    sentence_id: int
    query_start_idx: int  # For runtime masking
    query_end_idx: int  # For runtime masking
    full_sentence_text: str  # Store original full sentence
    full_sentence_words: List[str]  # Store original full sentence words
    fs: float


class EnhancedNarrativeSentenceReader:
    """
    UPDATED Narrative EEG reader with 3D word-level EEG arrays and runtime masking

    Key changes from v2:
    1. ✅ Uses 3D arrays [num_words, time_samples, channels] like Alice/DERCo
    2. ✅ Segments sentence-level EEG into word-level EEG using word timings
    3. ✅ Always stores unmasked documents for runtime masking
    4. ✅ Adds full_sentence_text and full_sentence_words fields
    5. ✅ Removes is_masked field (no generation-time masking)
    6. ✅ Matches Alice/DERCo ICTPair structure exactly
    """

    def __init__(
            self,
            text_path: str,
            eeg_base_path: str,
            target_freq: int = 128,
            low_freq: float = 0.5,
            high_freq: float = 100.0,
            preprocess: bool = True,
            verbose: bool = True,
            random_seed: int = 42,
            limit_subjects: Optional[int] = None,
            limit_runs_per_subject: Optional[int] = None
    ):
        self.text_path = text_path
        self.eeg_base_path = eeg_base_path
        self.target_freq = target_freq
        self.low_freq = low_freq
        self.high_freq = high_freq
        self.preprocess = preprocess
        self.verbose = verbose
        self.random_seed = random_seed
        self.limit_subjects = limit_subjects
        self.limit_runs_per_subject = limit_runs_per_subject

        # Set seeds for reproducibility
        self._set_seeds(random_seed)

        # Data containers
        self.all_sentences = []  # Store all sentence data for ICT generation

        # Metadata for reproducibility
        self.metadata = {
            'creation_date': datetime.now().isoformat(),
            'random_seed': random_seed,
            'text_path': str(text_path),
            'eeg_base_path': str(eeg_base_path),
            'target_freq': target_freq,
            'low_freq': low_freq,
            'high_freq': high_freq,
            'preprocess': preprocess,
            'limit_subjects': limit_subjects,
            'limit_runs_per_subject': limit_runs_per_subject,
            'version': 'v3_RUNTIME_MASKING_3D_ARRAYS'
        }

        print(f"🔧 UPDATED Narrative EEG Reader v3 (Runtime Masking + 3D Arrays)")
        print(f"🎲 Random seed: {random_seed}")
        print(f"✅ EEG format: 3D arrays [num_words, time_samples, channels]")
        print(f"✅ Masking: Runtime only (no generation-time masking)")
        if limit_subjects is None:
            print("🚀 Processing ALL subjects in dataset")
        else:
            print(f"🚀 Processing first {limit_subjects} subjects")

    def _set_seeds(self, seed: int):
        """Set all random seeds for reproducibility"""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def preprocess_eeg(self, eeg_data: np.ndarray, original_fs: float) -> Tuple[np.ndarray, float]:
        """Apply preprocessing to EEG data"""
        if not self.preprocess:
            return eeg_data, original_fs

        # Downsample if needed
        if original_fs != self.target_freq:
            num_samples = int(eeg_data.shape[0] * self.target_freq / original_fs)
            eeg_data = signal.resample(eeg_data, num_samples)

        # Bandpass filter
        nyq = 0.5 * self.target_freq
        low = max(0.001, min(self.low_freq / nyq, 0.99))
        high = max(low + 0.001, min(self.high_freq / nyq, 0.99))

        b, a = signal.butter(4, [low, high], btype='band')
        eeg_data = signal.filtfilt(b, a, eeg_data, axis=0)

        # Average reference
        eeg_data = eeg_data - np.mean(eeg_data, axis=1, keepdims=True)

        return eeg_data, self.target_freq

    def _find_words_in_sentence(self, word_vec, onset_times, offset_times, start_time, end_time):
        """Find words in a sentence given time boundaries"""
        return [
            (word[0].decode('utf-8') if isinstance(word[0], bytes) else word[0], onset, offset)
            for word, onset, offset in zip(word_vec, onset_times, offset_times)
            if start_time <= onset <= end_time
        ]

    def _from_mat_get_sentences(self, data):
        """Extract sentences from .mat file"""
        sentence_boundaries = data['sentence_boundaries'][0]
        word_vec = data['wordVec'].flatten()
        onset_times = data['onset_time'].flatten()
        offset_times = data['offset_time'].flatten()

        sentences = []
        for i in range(len(sentence_boundaries)):
            start_time = 0 if i == 0 else sentence_boundaries[i - 1]
            end_time = sentence_boundaries[i]

            words_in_sentence = self._find_words_in_sentence(
                word_vec, onset_times, offset_times, start_time, end_time
            )

            if words_in_sentence:  # Only add non-empty sentences
                sentences.append({
                    'words': [word for word, _, _ in words_in_sentence],
                    'word_onsets': [onset for _, onset, _ in words_in_sentence],
                    'word_offsets': [offset for _, _, offset in words_in_sentence],
                    'start_time': start_time,
                    'end_time': end_time,
                    'text': ' '.join([word for word, _, _ in words_in_sentence])
                })

        return sentences

    def _create_stimuli_sentences(self):
        """Create mapping of stimuli sentences - deterministic order for reproducibility"""
        stimuli_sentences = {}

        # Use sorted order for reproducibility
        for file in sorted(os.listdir(self.text_path)):
            if file.endswith('.mat'):
                path = os.path.join(self.text_path, file)
                data = sio.loadmat(path)
                stimuli_sentences[file] = self._from_mat_get_sentences(data)

        return stimuli_sentences

    def extract_word_level_eeg(self, sentence_eeg: np.ndarray, fs: float,
                               word_onsets: List[float], word_offsets: List[float]) -> List[np.ndarray]:
        """
        Extract word-level EEG segments from continuous sentence EEG

        NEW METHOD for v3: Segments sentence-level EEG into individual word EEGs

        Args:
            sentence_eeg: Continuous EEG [time_samples, channels]
            fs: Sampling frequency
            word_onsets: List of word onset times in seconds
            word_offsets: List of word offset times in seconds

        Returns:
            List of word-level EEG arrays, each shape [word_time_samples, channels]
        """
        word_eegs = []
        sentence_start_time = 0.0  # Sentence EEG starts at time 0

        for onset, offset in zip(word_onsets, word_offsets):
            # Convert times to sample indices relative to sentence start
            word_start_sample = int((onset - sentence_start_time) * fs)
            word_end_sample = int((offset - sentence_start_time) * fs)

            # Clip to valid range
            word_start_sample = max(0, min(word_start_sample, sentence_eeg.shape[0]))
            word_end_sample = max(word_start_sample + 1, min(word_end_sample, sentence_eeg.shape[0]))

            # Extract word EEG
            word_eeg = sentence_eeg[word_start_sample:word_end_sample, :]

            # Ensure we have at least some samples
            if word_eeg.shape[0] == 0:
                # Create minimal dummy word if extraction failed
                word_eeg = np.zeros((1, sentence_eeg.shape[1]))

            word_eegs.append(word_eeg)

        return word_eegs

    def pad_and_stack_word_eegs(self, word_eegs: List[np.ndarray]) -> np.ndarray:
        """
        Pad word EEGs to same length and stack into 3D array

        NEW METHOD for v3: Converts list of variable-length word EEGs into uniform 3D array

        Args:
            word_eegs: List of [time_samples, channels] arrays with variable time_samples

        Returns:
            3D array: [num_words, max_time_samples, channels]
        """
        if not word_eegs:
            return np.array([])

        # Find maximum time samples across all words
        max_samples = max(eeg.shape[0] for eeg in word_eegs)
        channels = word_eegs[0].shape[1]

        # Pad each word to max_samples
        padded_words = []
        for word_eeg in word_eegs:
            if word_eeg.shape[0] < max_samples:
                # Pad with zeros at the end
                padding = np.zeros((max_samples - word_eeg.shape[0], channels))
                padded_word = np.vstack([word_eeg, padding])
            else:
                padded_word = word_eeg[:max_samples]  # Trim if longer
            padded_words.append(padded_word)

        # Stack into 3D array: [num_words, time_samples, channels]
        return np.array(padded_words)

    def read_sentences_reproducibly(self) -> List[Dict]:
        """
        Read and process all sentences from the dataset with word-level EEG extraction

        UPDATED for v3: Now extracts word-level EEGs from sentence-level EEG
        """
        stimuli_sentences = self._create_stimuli_sentences()

        # Get all subject paths in deterministic order
        subject_paths = sorted([
            os.path.join(self.eeg_base_path, subj)
            for subj in os.listdir(self.eeg_base_path)
            if os.path.isdir(os.path.join(self.eeg_base_path, subj))
        ])

        # Limit subjects if requested
        if self.limit_subjects is not None:
            subject_paths = subject_paths[:self.limit_subjects]
            print(f"🔬 Limited to {len(subject_paths)} subjects")

        subjects_processed = 0
        sentence_idx = 0

        for subj_path in tqdm(subject_paths, desc="Processing subjects"):
            subject_id = os.path.basename(subj_path)

            # Get all run paths for this subject in deterministic order
            run_paths = sorted([
                os.path.join(subj_path, run)
                for run in os.listdir(subj_path)
                if run.startswith('Run') and os.path.isdir(os.path.join(subj_path, run))
            ])

            # Limit runs if requested
            if self.limit_runs_per_subject is not None:
                run_paths = run_paths[:self.limit_runs_per_subject]

            for run_path in run_paths:
                run_id = os.path.basename(run_path)

                # Find EEG and stimuli files
                eeg_file = None
                stimuli_file = None
                for file in os.listdir(run_path):
                    if file.endswith('_EEG.mat'):
                        eeg_file = os.path.join(run_path, file)
                    elif file.endswith('_Stimuli.mat'):
                        stimuli_file = os.path.join(run_path, file)

                if not eeg_file or not stimuli_file:
                    continue

                try:
                    # Load EEG data
                    eeg_data_raw = sio.loadmat(eeg_file)['eegData']
                    original_fs = 128.0  # Narrative dataset sampling rate

                    # Preprocess EEG
                    eeg_data, fs = self.preprocess_eeg(eeg_data_raw, original_fs)

                    # Get stimuli info
                    # Extract run number from stimuli filename
                    # Format: SubjectXX_RunYY_Stimuli.mat -> need to match with RunYY.mat from text folder
                    stimuli_basename = os.path.basename(stimuli_file)

                    # Try to extract run number
                    run_match = re.search(r'Run(\d+)', stimuli_basename, re.IGNORECASE)
                    if not run_match:
                        if self.verbose:
                            print(f"⚠ Could not extract run number from {stimuli_basename}")
                        continue

                    # Construct the expected text file name
                    run_num = run_match.group(1)
                    text_key = f"Run{run_num}.mat"

                    if text_key not in stimuli_sentences:
                        if self.verbose:
                            print(f"⚠ Text file {text_key} not found for {stimuli_basename}")
                        continue

                    sentences = stimuli_sentences[text_key]

                    # Process each sentence
                    for sent_info in sentences:
                        # Extract sentence-level EEG (as before)
                        start_sample = int(sent_info['start_time'] * fs)
                        end_sample = int(sent_info['end_time'] * fs)
                        end_sample = min(end_sample, eeg_data.shape[0])

                        if end_sample <= start_sample:
                            continue

                        sentence_eeg = eeg_data[start_sample:end_sample, :]

                        # ✅ NEW: Extract word-level EEGs from sentence EEG
                        word_eegs = self.extract_word_level_eeg(
                            sentence_eeg, fs,
                            sent_info['word_onsets'],
                            sent_info['word_offsets']
                        )

                        # ✅ NEW: Pad and stack into 3D array
                        word_eegs_3d = self.pad_and_stack_word_eegs(word_eegs)

                        if word_eegs_3d.shape[0] == 0 or word_eegs_3d.shape[0] != len(sent_info['words']):
                            # Skip if word segmentation failed
                            continue

                        # Store sentence data with word-level EEGs
                        sentence_data = {
                            'subject_id': subject_id,
                            'run_id': run_id,
                            'sentence_idx': sentence_idx,
                            'unique_sentence_id': f"{subject_id}_{run_id}_{sentence_idx}",
                            'words': sent_info['words'],
                            'word_eegs': word_eegs_3d,  # ✅ Now 3D: [num_words, time_samples, channels]
                            'word_onsets': sent_info['word_onsets'],
                            'word_offsets': sent_info['word_offsets'],
                            'text': sent_info['text'],
                            'fs': fs
                        }

                        self.all_sentences.append(sentence_data)
                        sentence_idx += 1

                except Exception as e:
                    if self.verbose:
                        print(f"⚠ Error processing {subject_id}/{run_id}: {e}")
                    continue

            subjects_processed += 1

        print(f"✅ Loaded {len(self.all_sentences)} sentences from {subjects_processed} subjects")
        print(f"✅ EEG format: 3D arrays [num_words, time_samples, channels]")

        # Update metadata
        self.metadata['sentences_loaded'] = len(self.all_sentences)
        self.metadata['subjects_processed'] = subjects_processed

        return self.all_sentences

    def generate_ict_pairs(
            self,
            min_query_length: int = 2,
            max_query_length: int = 50,
            query_length_ratio: float = 0.30,
            min_sentence_length: int = 6,
            use_ratio_based_queries: bool = True,
            max_pairs_per_sentence: int = 2,
            max_total_pairs: Optional[int] = None,
            random_seed: Optional[int] = None
    ) -> List[ICTPair]:
        """
        Generate ICT pairs with RUNTIME MASKING SUPPORT and 3D word-level EEG arrays

        UPDATED for v3:
        - Always stores UNMASKED documents (full sentence)
        - Uses 3D word-level EEG arrays [num_words, time_samples, channels]
        - No generation-time masking (is_masked field removed)
        """
        if random_seed is not None:
            self._set_seeds(random_seed)

        ict_pairs = []

        # Store generation parameters for reproducibility
        generation_params = {
            'min_query_length': min_query_length,
            'max_query_length': max_query_length,
            'query_length_ratio': query_length_ratio,
            'min_sentence_length': min_sentence_length,
            'use_ratio_based_queries': use_ratio_based_queries,
            'max_pairs_per_sentence': max_pairs_per_sentence,
            'max_total_pairs': max_total_pairs,
            'random_seed': random_seed,
            'masking_applied_at_generation': False,  # ✅ No generation-time masking
            'supports_runtime_masking': True  # ✅ Runtime masking support
        }
        self.metadata['ict_generation_params'] = generation_params

        # Process sentences in deterministic order
        for sentence_data in tqdm(sorted(self.all_sentences, key=lambda x: x['unique_sentence_id']),
                                  desc="Creating ICT pairs"):
            try:
                words = sentence_data['words']

                # Skip short sentences
                if len(words) < min_sentence_length:
                    continue

                # Generate multiple pairs per sentence
                sentence_pairs = 0
                attempts = 0
                max_attempts = max_pairs_per_sentence * 5

                while sentence_pairs < max_pairs_per_sentence and attempts < max_attempts:
                    pair = self._create_ict_pair_from_sentence(
                        sentence_data,
                        min_query_length,
                        max_query_length,
                        query_length_ratio,
                        use_ratio_based_queries
                    )

                    if pair is not None:
                        ict_pairs.append(pair)
                        sentence_pairs += 1

                        # Check if we've reached the total limit
                        if max_total_pairs and len(ict_pairs) >= max_total_pairs:
                            break

                    attempts += 1

                # Early termination if max pairs reached
                if max_total_pairs and len(ict_pairs) >= max_total_pairs:
                    break

            except Exception as e:
                if self.verbose:
                    print(f"⚠ Error generating ICT pair: {e}")
                continue

        print(f"🎉 Generated {len(ict_pairs)} ICT pairs with RUNTIME MASKING SUPPORT")
        self._print_ict_statistics(ict_pairs)

        # Update metadata
        self.metadata['generated_ict_pairs'] = len(ict_pairs)

        return ict_pairs

    def _create_ict_pair_from_sentence(self, sentence_data: Dict, min_query_length: int,
                                       max_query_length: int, query_length_ratio: float,
                                       use_ratio_based_queries: bool) -> Optional[ICTPair]:
        """
        Create a single ICT pair with RUNTIME MASKING SUPPORT and 3D word-level EEG

        UPDATED for v3:
        - Uses 3D word-level EEG arrays [num_words, time_samples, channels]
        - Always stores UNMASKED document (full sentence)
        - Stores query span metadata for runtime masking
        - No generation-time masking
        """
        words = sentence_data['words']
        word_eegs_3d = sentence_data['word_eegs']  # ✅ Already 3D: [num_words, time_samples, channels]
        sentence_length = len(words)

        # Determine query length
        if use_ratio_based_queries:
            query_length = max(min_query_length, int(sentence_length * query_length_ratio))
            query_length = min(query_length, sentence_length - 1)
            query_length = min(query_length, max_query_length)
        else:
            query_length = min(
                random.randint(min_query_length, max_query_length),
                sentence_length - 1
            )

        # Select query span
        max_start_idx = sentence_length - query_length
        query_start_idx = random.randint(0, max_start_idx)
        query_end_idx = query_start_idx + query_length

        # Extract query text and words
        query_words = words[query_start_idx:query_end_idx]
        query_text = ' '.join(query_words)

        # ✅ Extract query EEG (3D slicing)
        query_eeg = word_eegs_3d[query_start_idx:query_end_idx]  # Shape: [query_length, time_samples, channels]

        if query_eeg.shape[0] == 0:
            return None

        # ✅ RUNTIME MASKING: ALWAYS store UNMASKED document (full sentence)
        doc_words = words[:]  # Full sentence words
        doc_text = ' '.join(doc_words)  # Full sentence text
        doc_eeg = word_eegs_3d  # Full sentence EEG (3D: [num_words, time_samples, channels])

        # ✅ Create ICT pair with RUNTIME MASKING SUPPORT
        return ICTPair(
            query_text=query_text,
            query_eeg=query_eeg,  # 3D: [num_query_words, time_samples, channels]
            query_words=query_words,
            doc_text=doc_text,  # ✅ ALWAYS UNMASKED - full sentence
            doc_eeg=doc_eeg,  # ✅ ALWAYS UNMASKED - 3D: [num_words, time_samples, channels]
            doc_words=doc_words,  # ✅ ALWAYS UNMASKED - full sentence words
            participant_id=sentence_data['subject_id'],
            sentence_id=int(sentence_data['sentence_idx']),
            query_start_idx=query_start_idx,  # ✅ For runtime masking
            query_end_idx=query_end_idx,  # ✅ For runtime masking
            full_sentence_text=doc_text,  # ✅ Store full sentence
            full_sentence_words=doc_words,  # ✅ Store full sentence words
            fs=sentence_data['fs']
        )

    def _print_ict_statistics(self, ict_pairs: List[ICTPair]):
        """Print statistics about generated ICT pairs"""
        if not ict_pairs:
            return

        participants = set(pair.participant_id for pair in ict_pairs)

        query_lengths = [len(pair.query_words) for pair in ict_pairs]
        doc_lengths = [len(pair.doc_words) for pair in ict_pairs]

        # EEG shapes for 3D arrays
        query_eeg_shapes = [pair.query_eeg.shape for pair in ict_pairs]
        doc_eeg_shapes = [pair.doc_eeg.shape for pair in ict_pairs]

        print(f"\n📊 ICT PAIR STATISTICS (Runtime Masking Ready):")
        print(f"   👥 Participants: {len(participants)}")
        print(f"   🎭 All pairs UNMASKED (masking applied at runtime)")
        print(f"   📝 Query length: {np.mean(query_lengths):.1f} ± {np.std(query_lengths):.1f} words")
        print(f"   📄 Document length: {np.mean(doc_lengths):.1f} ± {np.std(doc_lengths):.1f} words")
        print(f"   🧠 Query EEG shape: [num_words, {query_eeg_shapes[0][1]}, {query_eeg_shapes[0][2]}]")
        print(f"   🧠 Document EEG shape: [num_words, {doc_eeg_shapes[0][1]}, {doc_eeg_shapes[0][2]}]")
        print(f"   ✅ Supports masking levels: 0%, 25%, 50%, 75%, 90%, 100%")

    def save_ict_pairs_with_metadata(self, ict_pairs: List[ICTPair], save_path: str):
        """Save ICT pairs to disk with comprehensive metadata"""
        save_path = Path(save_path)

        print(f"\n💾 Saving {len(ict_pairs)} ICT pairs with RUNTIME MASKING SUPPORT to {save_path}")

        # Convert ICT pairs to serializable format
        pairs_data = []
        for pair in ict_pairs:
            pair_dict = {
                'query_text': pair.query_text,
                'query_eeg': pair.query_eeg,
                'query_words': pair.query_words,
                'doc_text': pair.doc_text,
                'doc_eeg': pair.doc_eeg,
                'doc_words': pair.doc_words,
                'participant_id': pair.participant_id,
                'sentence_id': pair.sentence_id,
                'query_start_idx': pair.query_start_idx,
                'query_end_idx': pair.query_end_idx,
                'full_sentence_text': pair.full_sentence_text,
                'full_sentence_words': pair.full_sentence_words,
                'fs': pair.fs
            }
            pairs_data.append(pair_dict)

        # Create comprehensive dataset
        dataset = {
            'ict_pairs': pairs_data,
            'metadata': self.metadata,
            'version': '3.0',
            'description': 'Narrative EEG-text ICT pairs with 3D word-level EEG arrays and runtime masking support'
        }

        # Save as .npy file
        if save_path.suffix != '.npy':
            save_path = save_path.with_suffix('.npy')

        np.save(save_path, dataset)

        # Also save metadata as JSON for easy inspection
        metadata_path = save_path.with_suffix('.json')
        with open(metadata_path, 'w') as f:
            json.dump(self.metadata, f, indent=2)

        print(f"✅ Saved ICT pairs to: {save_path}")
        print(f"✅ Saved metadata to: {metadata_path}")
        print(f"🎲 Reproducible with seed: {self.metadata['random_seed']}")


# Main execution with RUNTIME MASKING PARAMETERS
if __name__ == "__main__":
    print("🚀 NARRATIVE ICT PAIR GENERATION v3 - RUNTIME MASKING + 3D ARRAYS")
    print("=" * 80)

    # Configuration
    RANDOM_SEED = 42
    LIMIT_SUBJECTS = None  # None = ALL subjects
    LIMIT_RUNS_PER_SUBJECT = None  # None = ALL runs per subject
    MAX_ICT_PAIRS = None  # None = ALL possible ICT pairs

    # File paths
    text_path = '/users/gxb18167/Natural Speech/Stimuli/Text'
    eeg_base_path = '/users/gxb18167/Natural Speech/EEG'
    output_file = "narrative_ict_pairs_RUNTIME_MASKING.npy"

    print(f"🔧 KEY UPDATES IN v3:")
    print(f"   ✅ 3D word-level EEG arrays [num_words, time_samples, channels]")
    print(f"   ✅ Matches Alice/DERCo format exactly")
    print(f"   ✅ Word-level EEG segmentation from sentence EEG")
    print(f"   ✅ Always stores unmasked documents")
    print(f"   ✅ Runtime masking support (0%-100%)")
    print(f"   ✅ Full reproducibility with seed: {RANDOM_SEED}")

    # Initialize the reader
    reader = EnhancedNarrativeSentenceReader(
        text_path=text_path,
        eeg_base_path=eeg_base_path,
        target_freq=128,
        low_freq=0.5,
        high_freq=100.0,
        preprocess=True,
        verbose=True,
        random_seed=RANDOM_SEED,
        limit_subjects=LIMIT_SUBJECTS,
        limit_runs_per_subject=LIMIT_RUNS_PER_SUBJECT
    )

    # Step 1: Read all sentences from dataset
    print(f"\n🔬 Step 1: Reading sentences with word-level EEG extraction...")
    sentences = reader.read_sentences_reproducibly()

    if not sentences:
        print("❌ No sentences found. Check data paths.")
        exit(1)

    # Step 2: Generate ICT pairs with runtime masking support
    print(f"\n🔧 Step 2: Generating ICT pairs with runtime masking...")
    ict_pairs = reader.generate_ict_pairs(
        min_query_length=2,
        max_query_length=50,
        query_length_ratio=0.30,
        min_sentence_length=6,
        use_ratio_based_queries=True,
        max_pairs_per_sentence=2,
        max_total_pairs=MAX_ICT_PAIRS,
        random_seed=RANDOM_SEED
    )

    # Step 3: Save with metadata
    if ict_pairs:
        print(f"\n💾 Step 3: Saving ICT pairs with metadata...")
        reader.save_ict_pairs_with_metadata(ict_pairs, output_file)

        # Final statistics
        print(f"\n📊 FINAL RESULTS:")
        print("=" * 40)
        participants = set(pair.participant_id for pair in ict_pairs)
        sentences_used = set(pair.sentence_id for pair in ict_pairs)

        print(f"👥 Participants: {len(participants)}")
        print(f"📝 Unique sentences: {len(sentences_used)}")
        print(f"🎭 All pairs UNMASKED (masking at runtime)")
        print(f"🧠 Total ICT pairs: {len(ict_pairs)}")
        print(f"📏 EEG format: 3D [num_words, time_samples, channels]")
        print(f"✅ Saved to: {output_file}")

        # Sample ICT pair info
        sample_pair = ict_pairs[0]
        print(f"\n📋 Sample ICT Pair:")
        print(f"   Query shape: {sample_pair.query_eeg.shape}")
        print(f"   Doc shape: {sample_pair.doc_eeg.shape}")
        print(f"   Query text: {sample_pair.query_text}")

        print(f"\n🎉 SUCCESS - v3 GENERATION COMPLETE!")
        print(f"✅ 3D word-level EEG arrays created")
        print(f"✅ Runtime masking support enabled")
        print(f"✅ Compatible with Alice/DERCo format")
        print(f"✅ Full reproducibility with seed: {RANDOM_SEED}")
        print(f"🚀 Ready for multi-dataset training!")

    else:
        print("❌ No ICT pairs generated. Check data paths and parameters.")