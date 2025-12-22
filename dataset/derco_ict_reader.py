#!/usr/bin/env python3
"""
DERCo EEG-Reading Dataset Reader with RUNTIME MASKING SUPPORT
Creates ICT pairs from DERCo preprocessed EEG data for Brain Passage Retrieval

Dataset: DERCo (https://osf.io/rkqbu/)
- 22 participants
- 3 fairy tales (RSVP @ 200ms/word)
- 32 channels @ 1000 Hz
- Preprocessed epochs: -50ms to +550ms around word onset

✅ Compatible with existing Alice/Nieuwland pipeline
✅ Supports runtime masking (0%-100%)
✅ Supports both word-level and sequence concatenation processing
"""

import os
import numpy as np
import mne
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import warnings
from collections import defaultdict
from tqdm import tqdm
import random
from dataclasses import dataclass
import json
from datetime import datetime
import torch

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
    doc_eeg: np.ndarray  # ALWAYS UNMASKED - full sentence EEG
    doc_words: List[str]  # ALWAYS UNMASKED - full sentence words
    participant_id: str
    sentence_id: int
    query_start_idx: int  # For runtime masking
    query_end_idx: int  # For runtime masking
    full_sentence_text: str  # Store original full sentence
    full_sentence_words: List[str]  # Store original full sentence words
    fs: float


# Story texts from DERCo dataset (Grimm's fairy tales)
DERCO_STORIES = {
    0: {
        'title': "The Mouse, the Bird, and the Sausage",
        'text': """Once upon a time a mouse a bird and a sausage formed a partnership They kept house together and for a long time they lived in peace and prosperity steadily adding to their property The bird's task was to fly to the forest every day to fetch wood The mouse carried water kindled the fire and set the table The sausage did the cooking One day the bird met another bird on the way who boasted to him of his own situation praising his own master This bird reproached him for working so hard while the other two enjoyed themselves at home The mouse after making the fire and carrying the water could retire to her room to rest until it was time to set the table The sausage had only to watch the pot to see that the food was properly cooked and when it was time to eat she could saunter through the vegetables or the porridge and grease them She had it done in a jiffy The bird heard all this and it worried him He flew home and put down his load He sat down at the table and after they had eaten their meal he announced that he would no longer fetch wood for he had been a servant long enough and had been made a fool of by both of them They should change jobs for a change The mouse begged and the sausage entreated him not to do this but all was of no use The bird insisted on his plan and so it had to be tried They cast lots and it fell to the sausage to fetch wood the mouse became the cook and the bird was to carry water The sausage went out toward the forest the bird kindled the fire and the mouse put on the pot and waited until the sausage came home bringing wood for the next day But the sausage stayed out so long that the other two became worried and the bird flew out a little way to meet her He had not gone far when he came upon a dog who had seized the sausage as free booty and was making off with her The bird complained bitterly to the dog about this brazen abduction but it did him no good for the dog claimed that he had discovered forged letters on her person and that she would therefore have to forfeit her life to him In his grief the bird carried the wood home himself and told the mouse what he had seen and heard They were both very sad but they decided to stay together and make the best of it The bird set the table and the mouse prepared the food She wanted to dress it and to grease the vegetables as the sausage used to do by sliding and winding her way through them but before she reached the middle she was stopped and trapped by the steam and there she perished When the bird wanted to eat and no cook appeared he scattered the wood here and there called and searched everywhere but could not find his cook Due to his carelessness the wood caught fire and a conflagration ensued The bird rushed to fetch water but the bucket fell into the well and he fell in after it and could not get out again and there he drowned"""
    },
    1: {
        'title': "Straw, Coal, and Bean",
        'text': """In a village there lived a poor old woman who had collected a bunch of beans and wanted to cook them She prepared a fire on her hearth and to make it burn faster she kindled it with a handful of straw When she was pouring the beans into the pot one of them fell unnoticed to the floor and came to rest next to a piece of straw Shortly afterward a glowing coal jumped out of the fire and landed next to them The straw began to speak saying Why did you jump out of that fire dear friends I managed to escape by a hair Had I not succeeded my death would have been certain I would have been burned to ashes The coal responded I too was fortunate enough to escape Otherwise I would be dead by now I would have burned to death The bean said I too just barely escaped with a whole skin If the old woman had gotten me into the pot I would have been cooked to mush like my comrades What should we do now asked the straw I think said the coal that because we have so fortunately escaped death we should join together as comrades and in order to prevent some new misfortune from befalling us here we should together journey forth to a foreign land This proposal pleased the other two and they set forth all together Soon they came to a small brook There was no bridge nor walkway there so they did not know how they would get across The straw had a good idea and said I will lay myself across then you can walk over me like over a bridge The straw stretched himself from one bank to the other The coal who was a hot headed fellow stepped brashly onto the newly constructed bridge But when he got to the middle and heard the water rushing beneath him he took fright stopped and did not dare to go any further The straw began to burn broke in two and fell into the brook The coal slid after him hissed as he fell into the water and gave up the ghost The bean who had cautiously remained on the bank had to laugh at the event She could not stop and she laughed so hard that she burst Apart from this she too would have died but fortunately a wandering tailor who had stopped there to rest happened to be sitting near the brook Having a compassionate heart he got out a needle and thread and sewed her back together The bean thanked him most kindly But because he had used black thread all beans since then have had a black seam"""
    },
    2: {
        'title': "Poverty and Humility Lead to Heaven",
        'text': """There once was a king's son who went out into the world and he was full of thought and sad He looked at the sky and it was so beautifully pure and blue then he sighed and said How well must all be with one up there in heaven On the way he saw a poor gray man who asked him Where are you going He answered I would gladly go to heaven If you want to quickly get there said the man make yourself poor like me and go with me into the world for seven years learn to know misery and suffer it patiently I will be your companion So the king's son gave up his fine coat the man took it and he gave the poor man his dress in exchange The man led him out into the great world and he had to work in the fields had to keep the pigs and suffer much Formerly I wore gold and silver said the king's son at the end of the first year but that is all over I had to be contented with coarse food and sour bread At the end of the second year he sighed even more and said Oh when will the seven years be at an end At the beginning of the third year when he was sitting under a tree he heard a farmer plowing in the field singing a joyful song The farmer was so cheerful and happy and the king's son thought How gladsome that man is but I am miserable All he had wished for was to be in heaven When six years had gone by the king's son's desire grew ever stronger and finally when the seventh year was drawing to a close he could no longer conceal it and said To stay here any longer I cannot I must go up to heaven So the gray man said to him It is well then that you have fulfilled your time in patience look upon the great sea there in the distance There is an island and on it stands a church and in the church stands an altar and on the altar is a holy silver chalice Drink from it and you will get to heaven So they went to the island entered the church and there stood the altar and the silver chalice and on it was red wine And he drank from it and all his sorrow vanished at once and he was in heaven"""
    }
}


class DERCoEEGReader:
    """
    DERCo EEG-Reading Dataset Reader with RUNTIME MASKING SUPPORT
    
    Loads preprocessed EEG data from DERCo RSVP experiment and creates ICT pairs
    compatible with existing Alice/Nieuwland pipeline.
    """
    
    def __init__(self, data_dir: str, random_seed: int = 42, 
                 limit_participants: Optional[int] = None,
                 target_freq: int = 128, target_channels: int = 128,
                 resample: bool = True, pad_channels: bool = True,
                 verbose: bool = True):
        """
        Initialize DERCo reader
        
        Args:
            data_dir: Path to DERCo_preprocessed_rsvp directory
            random_seed: Random seed for reproducibility
            limit_participants: Limit to first N participants (None = all)
            target_freq: Target sampling frequency (default 128 Hz to match Alice/Nieuwland)
            target_channels: Target channel count (default 128 to match Alice/Nieuwland)
            resample: Whether to resample to target_freq
            pad_channels: Whether to pad channels to target_channels
            verbose: Print progress messages
        """
        self.data_dir = Path(data_dir)
        self.random_seed = random_seed
        self.limit_participants = limit_participants
        self.target_freq = target_freq
        self.target_channels = target_channels
        self.resample = resample
        self.pad_channels = pad_channels
        self.verbose = verbose
        
        # Set seeds
        self._set_seeds(random_seed)
        
        # Data containers
        self.participants = []
        self.all_sentence_data = []
        
        # Metadata for reproducibility
        self.metadata = {
            'creation_date': datetime.now().isoformat(),
            'random_seed': random_seed,
            'data_dir': str(data_dir),
            'target_freq': target_freq,
            'target_channels': target_channels,
            'resample': resample,
            'pad_channels': pad_channels,
            'limit_participants': limit_participants,
            'version': 'DERCO_RUNTIME_MASKING_v1.0',
            'supports_runtime_masking': True,
            'masking_method': 'runtime_query_span_removal',
            'dataset': 'DERCo',
            'original_freq': 1000,
            'original_channels': 32,
            'epoch_window': '[-50ms, +550ms]',
            'presentation_method': 'RSVP @ 200ms/word + 300ms blank'
        }
        
        print(f"📚 DERCo EEG-Reading Dataset Reader (Runtime Masking Support)")
        print(f"🎲 Random seed: {random_seed}")
        print(f"📁 Data directory: {data_dir}")
        
        # Scan for participants
        self._scan_participants()
    
    def _set_seeds(self, seed: int):
        """Set all random seeds for reproducibility"""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    
    def _scan_participants(self):
        """Scan for available participants"""
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {self.data_dir}")
        
        print("📝 Scanning for participants...")
        
        # Get all participant directories
        participant_dirs = sorted([d for d in self.data_dir.iterdir() if d.is_dir()])
        
        if self.limit_participants:
            participant_dirs = participant_dirs[:self.limit_participants]
        
        for participant_dir in participant_dirs:
            participant_id = participant_dir.name
            
            # Check for article files
            articles_found = []
            for article_num in [0, 1, 2]:
                article_path = participant_dir / f"article_{article_num}" / "preprocessed_epoch.fif"
                if article_path.exists():
                    articles_found.append(article_num)
            
            if articles_found:
                self.participants.append({
                    'id': participant_id,
                    'path': participant_dir,
                    'articles': articles_found
                })
                print(f"✅ Found participant: {participant_id} with articles {articles_found}")
        
        print(f"📊 Total participants loaded: {len(self.participants)}")
        self.metadata['participants_processed'] = len(self.participants)
    
    def _load_epochs_from_fif(self, fif_path: Path) -> Tuple[Optional[mne.Epochs], Optional[List[str]]]:
        """
        Load preprocessed epochs from .fif file
        
        Returns:
            Tuple of (epochs, word_list) or (None, None) if loading fails
        """
        try:
            epochs = mne.read_epochs(fif_path, preload=True, verbose=False)
            
            # Extract word information from epoch metadata
            # DERCo epochs should have word information in metadata
            words = []
            for epoch_idx in range(len(epochs)):
                # Try to get word from metadata (adjust based on actual DERCo structure)
                # This may need adjustment based on actual .fif metadata structure
                words.append(f"word_{epoch_idx}")  # Placeholder - will be replaced by actual text
            
            return epochs, words
            
        except Exception as e:
            if self.verbose:
                print(f"⚠ Error loading {fif_path}: {e}")
            return None, None
    
    def _resample_eeg(self, eeg_data: np.ndarray, original_freq: float) -> np.ndarray:
        """Resample EEG data to target frequency"""
        if not self.resample or original_freq == self.target_freq:
            return eeg_data
        
        from scipy import signal
        num_samples = int(eeg_data.shape[0] * self.target_freq / original_freq)
        return signal.resample(eeg_data, num_samples, axis=0)
    
    def _pad_channels_array(self, eeg_data: np.ndarray) -> np.ndarray:
        """Pad EEG channels to target number"""
        if not self.pad_channels:
            return eeg_data
        
        current_channels = eeg_data.shape[-1]
        if current_channels < self.target_channels:
            padding_shape = list(eeg_data.shape)
            padding_shape[-1] = self.target_channels - current_channels
            padding = np.zeros(padding_shape)
            return np.concatenate([eeg_data, padding], axis=-1)
        
        return eeg_data
    
    def load_all_sentences(self, max_sentences_per_article: Optional[int] = None):
        """
        Load all sentences from all participants
        
        Args:
            max_sentences_per_article: Limit sentences per article (None = all)
        """
        print("\n📖 Loading sentences from all participants...")
        
        total_sentences = 0
        total_words = 0
        
        for participant in tqdm(self.participants, desc="Processing participants"):
            participant_id = participant['id']
            
            for article_num in participant['articles']:
                # Load story text
                story = DERCO_STORIES[article_num]
                story_text = story['text']
                story_words = story_text.split()
                
                # Load EEG epochs
                fif_path = participant['path'] / f"article_{article_num}" / "preprocessed_epoch.fif"
                epochs, _ = self._load_epochs_from_fif(fif_path)
                
                if epochs is None:
                    continue
                
                original_freq = epochs.info['sfreq']
                
                # Get EEG data: shape (n_epochs, n_channels, n_times)
                eeg_data = epochs.get_data()  # shape: (n_epochs, 32, time_samples)
                
                # Verify we have correct number of words
                if len(eeg_data) != len(story_words):
                    if self.verbose:
                        print(f"⚠ Mismatch: {participant_id} article_{article_num}: "
                              f"{len(eeg_data)} epochs vs {len(story_words)} words")
                    # Trim to minimum
                    min_len = min(len(eeg_data), len(story_words))
                    eeg_data = eeg_data[:min_len]
                    story_words = story_words[:min_len]
                
                # Process each word's EEG
                word_eeg_list = []
                for word_idx in range(len(story_words)):
                    word_eeg = eeg_data[word_idx].T  # Transpose to (time_samples, channels)
                    
                    # Resample if needed
                    word_eeg = self._resample_eeg(word_eeg, original_freq)
                    
                    # Pad channels if needed
                    word_eeg = self._pad_channels_array(word_eeg)
                    
                    word_eeg_list.append(word_eeg)
                
                # Split into sentences (simple heuristic: every ~15-20 words)
                # For DERCo, we can create "sentences" as chunks of the story
                sentence_length = 15
                num_sentences = len(story_words) // sentence_length
                
                if max_sentences_per_article and num_sentences > max_sentences_per_article:
                    num_sentences = max_sentences_per_article
                
                for sent_idx in range(num_sentences):
                    start_idx = sent_idx * sentence_length
                    end_idx = min(start_idx + sentence_length, len(story_words))
                    
                    if end_idx - start_idx < 6:  # Skip very short sentences
                        continue
                    
                    sentence_words = story_words[start_idx:end_idx]
                    sentence_eegs = word_eeg_list[start_idx:end_idx]
                    sentence_text = ' '.join(sentence_words)
                    
                    # Create sentence data structure
                    sentence_data = {
                        'sentence_id': f"{participant_id}_art{article_num}_sent{sent_idx}",
                        'participant_id': participant_id,
                        'article_num': article_num,
                        'full_sentence': sentence_text,
                        'words': sentence_words,
                        'word_eegs': sentence_eegs,
                        'fs': self.target_freq if self.resample else original_freq
                    }
                    
                    self.all_sentence_data.append(sentence_data)
                    total_sentences += 1
                    total_words += len(sentence_words)
        
        print(f"\n✅ Loaded {total_sentences} sentences with {total_words} total words")
        print(f"   from {len(self.participants)} participants")
        
        self.metadata['total_sentences'] = total_sentences
        self.metadata['total_words'] = total_words
    
    def generate_ict_pairs(self, min_query_length: int = 2, max_query_length: int = 5,
                          query_length_ratio: float = 0.3, min_sentence_length: int = 8,
                          use_ratio_based_queries: bool = True, max_pairs_per_sentence: int = 2,
                          max_total_pairs: Optional[int] = None, 
                          random_seed: Optional[int] = None) -> List[ICTPair]:
        """
        Generate ICT pairs from sentence data with RUNTIME MASKING SUPPORT
        
        Args:
            min_query_length: Minimum query length in words
            max_query_length: Maximum query length in words
            query_length_ratio: Query length as ratio of sentence length
            min_sentence_length: Minimum sentence length to use
            use_ratio_based_queries: Use ratio-based query lengths
            max_pairs_per_sentence: Maximum ICT pairs per sentence
            max_total_pairs: Maximum total pairs to generate
            random_seed: Random seed for generation
            
        Returns:
            List of ICTPair objects
        """
        if random_seed is None:
            random_seed = self.random_seed
        
        self._set_seeds(random_seed)
        
        if not self.all_sentence_data:
            print("⚠ No sentence data available. Run load_all_sentences() first.")
            return []
        
        print(f"\n📧 Generating ICT pairs with RUNTIME MASKING SUPPORT...")
        print(f"📊 PARAMETERS:")
        print(f"   min_query_length: {min_query_length}")
        print(f"   max_query_length: {max_query_length}")
        print(f"   query_length_ratio: {query_length_ratio}")
        print(f"   min_sentence_length: {min_sentence_length}")
        print(f"   use_ratio_based_queries: {use_ratio_based_queries}")
        print(f"   max_pairs_per_sentence: {max_pairs_per_sentence}")
        print(f"🎭 MASKING: Will be applied at RUNTIME (supports 0%-100%)")
        
        ict_pairs = []
        
        # Store generation parameters
        generation_params = {
            'min_query_length': min_query_length,
            'max_query_length': max_query_length,
            'query_length_ratio': query_length_ratio,
            'min_sentence_length': min_sentence_length,
            'use_ratio_based_queries': use_ratio_based_queries,
            'max_pairs_per_sentence': max_pairs_per_sentence,
            'max_total_pairs': max_total_pairs,
            'random_seed': random_seed,
            'masking_applied_at_generation': False,
            'supports_runtime_masking': True
        }
        self.metadata['ict_generation_params'] = generation_params
        
        # Process sentences
        for sentence_data in tqdm(self.all_sentence_data, desc="Creating ICT pairs"):
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
                    sentence_data, min_query_length, max_query_length,
                    query_length_ratio, use_ratio_based_queries
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
        
        print(f"🎉 Generated {len(ict_pairs)} ICT pairs with RUNTIME MASKING SUPPORT")
        self._print_ict_statistics(ict_pairs)
        
        self.metadata['generated_ict_pairs'] = len(ict_pairs)
        return ict_pairs
    
    def _create_ict_pair_from_sentence(self, sentence_data: Dict, min_query_length: int,
                                      max_query_length: int, query_length_ratio: float,
                                      use_ratio_based_queries: bool) -> Optional[ICTPair]:
        """
        Create a single ICT pair with RUNTIME MASKING SUPPORT
        
        ✅ KEY: Always stores UNMASKED document (full sentence)
        ✅ Stores query span metadata for runtime masking
        """
        words = sentence_data['words']
        word_eegs = sentence_data['word_eegs']
        sentence_length = len(words)
        
        # Determine query length
        if use_ratio_based_queries:
            query_length = max(min_query_length, int(sentence_length * query_length_ratio))
            query_length = min(query_length, sentence_length - 1)
        else:
            query_length = min(random.randint(min_query_length, max_query_length), sentence_length - 1)
        
        # Select query span
        max_start_idx = sentence_length - query_length
        query_start_idx = random.randint(0, max_start_idx)
        query_end_idx = query_start_idx + query_length
        
        # Extract query
        query_words = words[query_start_idx:query_end_idx]
        query_word_eegs = word_eegs[query_start_idx:query_end_idx]
        query_text = ' '.join(query_words)
        
        # Convert query EEGs to numpy array: shape (num_words, time_samples, channels)
        try:
            query_eeg = np.array(query_word_eegs)
        except Exception as e:
            if self.verbose:
                print(f"⚠ Error stacking query EEGs: {e}")
            return None
        
        # ✅ RUNTIME MASKING: ALWAYS store UNMASKED document (full sentence)
        doc_words = words[:]
        doc_word_eegs = word_eegs[:]
        doc_text = ' '.join(doc_words)
        
        # Convert document EEGs to numpy array: shape (num_words, time_samples, channels)
        try:
            doc_eeg = np.array(doc_word_eegs)
        except Exception as e:
            if self.verbose:
                print(f"⚠ Error stacking doc EEGs: {e}")
            # Fallback: create dummy array
            doc_eeg = np.zeros((len(words), query_eeg.shape[1], query_eeg.shape[2]))
        
        # ✅ Create ICT pair with RUNTIME MASKING SUPPORT
        return ICTPair(
            query_text=query_text,
            query_eeg=query_eeg,
            query_words=query_words,
            doc_text=doc_text,  # ✅ ALWAYS UNMASKED
            doc_eeg=doc_eeg,  # ✅ ALWAYS UNMASKED
            doc_words=doc_words,  # ✅ ALWAYS UNMASKED
            participant_id=sentence_data['participant_id'],
            sentence_id=sentence_data['sentence_id'],
            query_start_idx=query_start_idx,  # ✅ For runtime masking
            query_end_idx=query_end_idx,  # ✅ For runtime masking
            full_sentence_text=doc_text,
            full_sentence_words=doc_words,
            fs=sentence_data['fs']
        )
    
    def _print_ict_statistics(self, ict_pairs: List[ICTPair]):
        """Print statistics about generated ICT pairs"""
        if not ict_pairs:
            return
        
        participants = set(pair.participant_id for pair in ict_pairs)
        
        query_lengths = [len(pair.query_words) for pair in ict_pairs]
        doc_lengths = [len(pair.doc_words) for pair in ict_pairs]
        
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
                'doc_text': pair.doc_text,  # UNMASKED
                'doc_eeg': pair.doc_eeg,  # UNMASKED
                'doc_words': pair.doc_words,  # UNMASKED
                'participant_id': pair.participant_id,
                'sentence_id': pair.sentence_id,
                'query_start_idx': pair.query_start_idx,  # For runtime masking
                'query_end_idx': pair.query_end_idx,  # For runtime masking
                'full_sentence_text': pair.full_sentence_text,
                'full_sentence_words': pair.full_sentence_words,
                'fs': pair.fs
            }
            pairs_data.append(pair_dict)
        
        # Create comprehensive dataset
        dataset = {
            'ict_pairs': pairs_data,
            'metadata': self.metadata,
            'version': '2.0',
            'description': 'DERCo EEG-text ICT pairs with RUNTIME MASKING SUPPORT - compatible with multi-masking validation'
        }
        
        # Save as .npy file
        if save_path.suffix != '.npy':
            save_path = save_path.with_suffix('.npy')
        
        np.save(save_path, dataset)
        
        # Also save metadata as JSON
        metadata_path = save_path.with_suffix('.json')
        with open(metadata_path, 'w') as f:
            json.dump(self.metadata, f, indent=2)
        
        print(f"✅ Saved ICT pairs to: {save_path}")
        print(f"✅ Saved metadata to: {metadata_path}")
        print(f"🎲 Reproducible with seed: {self.metadata['random_seed']}")
        print(f"🎭 Ready for multi-masking validation (0%, 25%, 50%, 75%, 90%, 100%)")


# Main execution
if __name__ == "__main__":
    print("🚀 DERCO EEG-READING ICT PAIR GENERATION - RUNTIME MASKING SUPPORT")
    print("=" * 80)
    
    # Configuration
    RANDOM_SEED = 42
    LIMIT_PARTICIPANTS = None  # None = ALL participants
    MAX_ICT_PAIRS = None  # None = ALL possible ICT pairs
    
    # File paths - ADJUST THESE!
    data_dir = "DERCo_preprocessed_rsvp"  # Path to downloaded DERCo data
    output_file = "derco_ict_pairs_RUNTIME_MASKING.npy"
    
    print(f"📧 RUNTIME MASKING FEATURES:")
    print(f"   ✅ No masking applied at generation time")
    print(f"   ✅ Stores query span metadata for runtime masking")
    print(f"   ✅ Compatible with any masking probability (0%-100%)")
    print(f"   ✅ Supports multi-masking validation during training")
    print(f"   ✅ Supports both word-level and sequence concatenation processing")
    
    # Initialize reader
    reader = DERCoEEGReader(
        data_dir=data_dir,
        random_seed=RANDOM_SEED,
        limit_participants=LIMIT_PARTICIPANTS,
        target_freq=128,  # Match Alice/Nieuwland
        target_channels=128,  # Match Alice/Nieuwland
        resample=True,
        pad_channels=True,
        verbose=True
    )
    
    # Step 1: Load all sentences
    print(f"\n📖 Step 1: Loading sentences from dataset...")
    reader.load_all_sentences(max_sentences_per_article=None)
    
    if not reader.all_sentence_data:
        print("⚠ No sentences found. Check data paths.")
        exit(1)
    
    # Step 2: Generate ICT pairs
    print(f"\n📧 Step 2: Generating ICT pairs with runtime masking support...")
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
        print(f"\n💾 Step 3: Saving ICT pairs with runtime masking support...")
        reader.save_ict_pairs_with_metadata(ict_pairs, output_file)
        
        # Final statistics
        print(f"\n📊 FINAL RESULTS:")
        print("=" * 40)
        participants = set(pair.participant_id for pair in ict_pairs)
        
        print(f"👥 Participants: {len(participants)}")
        print(f"🎭 All pairs UNMASKED (ready for runtime masking)")
        print(f"🧠 Total ICT pairs: {len(ict_pairs)}")
        print(f"✅ Saved to: {output_file}")
        
        print(f"\n🎉 SUCCESS - RUNTIME MASKING SUPPORT COMPLETE!")
        print(f"✅ Generated {len(ict_pairs)} ICT pairs ready for runtime masking")
        print(f"✅ Compatible with masking levels: 0%, 25%, 50%, 75%, 90%, 100%")
        print(f"✅ Supports both word-level and sequence concatenation")
        print(f"✅ Full reproducibility with seed: {RANDOM_SEED}")
        print(f"🚀 Ready for multi-masking validation during training!")
    
    else:
        print("⚠ No ICT pairs generated. Check data paths and parameters.")
