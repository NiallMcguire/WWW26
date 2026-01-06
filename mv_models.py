#!/usr/bin/env python3
"""
COMPLETE FIXED Multi-Vector Models for Brain Passage Retrieval
FIXES APPLIED:
1. EEG encoder initialized during __init__ (not during forward pass) ✓
2. Positional encodings in transformer ✓
3. True LaBraM-style channel patching (each channel processed independently) ✓
4. Spatial (channel) embeddings for 10-20 system ✓
5. Proper parameter registration for optimizer ✓
"""

import torch
import torch.nn as nn
from transformers import AutoModel
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model, TaskType
import math


class LaBramChannelPatcher(nn.Module):
    """
    TRUE LaBraM-style patching: processes each EEG channel independently
    Creates C × T patches (channels × time_patches) where each patch is single-channel
    """

    def __init__(self, patch_length=200, num_channels=64, max_time_patches=50):
        super().__init__()
        self.patch_length = patch_length
        self.num_channels = num_channels
        self.max_time_patches = max_time_patches

    def forward(self, eeg_input):
        """
        Convert EEG to channel-wise patches

        Args:
            eeg_input: [batch, num_words, time_samples, channels]
        Returns:
            patches: [batch, num_words, channels × time_patches, patch_length]
            channel_indices: [channels × time_patches] - which channel each patch came from
            time_indices: [channels × time_patches] - which time window each patch is
        """
        batch_size, num_words, time_samples, channels = eeg_input.shape

        # Calculate number of time patches per channel
        num_time_patches = time_samples // self.patch_length

        all_patches = []
        channel_indices = []
        time_indices = []

        for word_idx in range(num_words):
            word_data = eeg_input[:, word_idx, :, :]  # [batch, time, channels]

            word_patches = []

            # For each channel, create temporal patches
            for ch_idx in range(channels):
                channel_data = word_data[:, :, ch_idx]  # [batch, time]

                # Create temporal patches for this channel
                for t_idx in range(num_time_patches):
                    start_t = t_idx * self.patch_length
                    end_t = start_t + self.patch_length

                    if end_t <= time_samples:
                        patch = channel_data[:, start_t:end_t]  # [batch, patch_length]
                        word_patches.append(patch.unsqueeze(1))  # [batch, 1, patch_length]

                        if word_idx == 0:  # Only compute indices once
                            channel_indices.append(ch_idx)
                            time_indices.append(t_idx)

            # Stack patches for this word: [batch, C×T, patch_length]
            if len(word_patches) > 0:
                word_patches_stacked = torch.cat(word_patches, dim=1)
                all_patches.append(word_patches_stacked)

        # Stack across words: [batch, num_words, C×T, patch_length]
        patches = torch.stack(all_patches, dim=1)

        # Create index tensors
        channel_indices = torch.tensor(channel_indices, device=eeg_input.device, dtype=torch.long)
        time_indices = torch.tensor(time_indices, device=eeg_input.device, dtype=torch.long)

        return patches, channel_indices, time_indices


class LaBramCNNPreprocessor(nn.Module):
    """
    TRUE LaBraM-style CNN: processes single-channel patches
    Input: Single-channel patches
    Output: Hidden dimension features
    """

    def __init__(self, hidden_dim=768, num_layers=3):
        super().__init__()

        # CNN layers for single-channel input (input_channels=1)
        if num_layers == 3:
            self.cnn_blocks = nn.Sequential(
                nn.Conv1d(1, hidden_dim // 4, kernel_size=15, stride=8, padding=7),  # 1 input channel!
                nn.GroupNorm(4, hidden_dim // 4),
                nn.GELU(),
                nn.Conv1d(hidden_dim // 4, hidden_dim // 2, kernel_size=3, stride=1, padding=1),
                nn.GroupNorm(8, hidden_dim // 2),
                nn.GELU(),
                nn.Conv1d(hidden_dim // 2, hidden_dim, kernel_size=3, stride=1, padding=1),
                nn.GroupNorm(16, hidden_dim),
                nn.GELU()
            )
        else:
            self.cnn_blocks = nn.Sequential(
                nn.Conv1d(1, hidden_dim, kernel_size=15, stride=8, padding=7),  # 1 input channel!
                nn.GroupNorm(8, hidden_dim),
                nn.GELU()
            )

    def forward(self, x):
        """
        Args:
            x: [batch × num_words × num_patches, patch_length] - flattened single-channel patches
        Returns:
            [batch × num_words × num_patches, hidden_dim]
        """
        # Add channel dimension: [batch × num_words × num_patches, 1, patch_length]
        x = x.unsqueeze(1)

        # Apply CNN
        x = self.cnn_blocks(x)  # [batch × num_words × num_patches, hidden_dim, new_length]

        # Global average pooling over time
        x = x.mean(dim=2)  # [batch × num_words × num_patches, hidden_dim]

        return x


class LaBramPositionalEmbeddings(nn.Module):
    """
    LaBraM-style positional embeddings for EEG
    - Temporal: which time patch within a sequence
    - Spatial: which channel/electrode (mapped to 10-20 system)
    - Word: which word in the sequence
    """

    def __init__(self, hidden_dim, max_words=50, max_time_patches=50, max_channels=128):
        super().__init__()

        # Learnable embeddings
        self.temporal_embeddings = nn.Embedding(max_time_patches, hidden_dim)
        self.spatial_embeddings = nn.Embedding(max_channels, hidden_dim)
        self.word_embeddings = nn.Embedding(max_words, hidden_dim)

        self.max_words = max_words
        self.max_time_patches = max_time_patches
        self.max_channels = max_channels

    def forward(self, patch_features, num_words, channel_indices, time_indices):
        """
        Add positional embeddings to patch features

        Args:
            patch_features: [batch, num_words, C×T, hidden_dim]
            num_words: number of words
            channel_indices: [C×T] - which channel each patch came from
            time_indices: [C×T] - which time window each patch is
        Returns:
            [batch, num_words, C×T, hidden_dim] with positional encodings added
        """
        batch_size, num_words, num_patches, hidden_dim = patch_features.shape

        # Word embeddings: [batch, num_words, 1, hidden_dim]
        word_pos = torch.arange(num_words, device=patch_features.device).unsqueeze(0).expand(batch_size, -1)
        word_emb = self.word_embeddings(word_pos).unsqueeze(2)

        # Spatial embeddings: [1, 1, C×T, hidden_dim]
        spatial_emb = self.spatial_embeddings(channel_indices).unsqueeze(0).unsqueeze(0)

        # Temporal embeddings: [1, 1, C×T, hidden_dim]
        temporal_emb = self.temporal_embeddings(time_indices).unsqueeze(0).unsqueeze(0)

        # Add all embeddings
        return patch_features + word_emb + spatial_emb + temporal_emb


class SimpleTextEncoder(nn.Module):
    """Simple text encoder trained from scratch (similar complexity to EEG encoder)"""

    def __init__(self, vocab_size, hidden_dim=768, arch='simple'):
        super().__init__()
        self.arch = arch
        self.hidden_dim = hidden_dim

        # Embedding layer (trained from scratch)
        self.embedding = nn.Embedding(vocab_size, hidden_dim, padding_idx=0)

        if arch == 'simple':
            self.encoder = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            )
        elif arch == 'complex':
            self.encoder = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(hidden_dim, hidden_dim * 2),
                nn.LayerNorm(hidden_dim * 2),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(hidden_dim * 2, hidden_dim)
            )
        elif arch == 'transformer':
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=8,
                dim_feedforward=hidden_dim * 2,
                dropout=0.1,
                batch_first=True
            )
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)

    def forward(self, input_ids, attention_mask):
        embedded = self.embedding(input_ids)

        if self.arch == 'transformer':
            padding_mask = ~attention_mask.bool()
            encoded = self.encoder(embedded, src_key_padding_mask=padding_mask)
        else:
            mask_expanded = attention_mask.unsqueeze(-1).float()
            pooled = (embedded * mask_expanded).sum(dim=1) / (mask_expanded.sum(dim=1) + 1e-8)
            encoded = self.encoder(pooled).unsqueeze(1)

        return encoded


class SimplifiedBrainRetrieval(nn.Module):
    """
    COMPLETE FIXED Brain Retrieval Model with:
    1. EEG encoder initialized during __init__ (not lazily) ✓
    2. Positional encodings in transformer ✓
    3. Optional LaBraM-style channel patching ✓
    4. Proper parameter registration ✓
    """

    def __init__(self, colbert_model_name='colbert-ir/colbertv2.0',
                 hidden_dim=768, eeg_arch='simple', dropout=0.1,
                 use_lora=True, lora_r=16, lora_alpha=32,
                 pooling_strategy='multi', query_type='eeg',
                 use_pretrained_text=True, global_eeg_dims=None,
                 use_labram_patching=False, labram_patch_length=200):
        super().__init__()

        self.query_type = query_type
        self.pooling_strategy = pooling_strategy
        self.hidden_dim = hidden_dim
        self.eeg_arch = eeg_arch
        self.use_lora = use_lora
        self.use_pretrained_text = use_pretrained_text
        self.use_labram_patching = use_labram_patching

        # Validate pooling strategy
        if pooling_strategy not in ['multi', 'cls', 'max', 'mean']:
            raise ValueError(
                f"Only 'multi', 'cls', 'max', and 'mean' pooling strategies supported, got: {pooling_strategy}")

        # Text encoder
        if use_pretrained_text:
            print(f"Loading pretrained ColBERT model: {colbert_model_name}")
            try:
                self.text_encoder = AutoModel.from_pretrained(colbert_model_name)
            except:
                print(f"ColBERT model not found, falling back to bert-base-uncased")
                self.text_encoder = AutoModel.from_pretrained('bert-base-uncased')
                colbert_model_name = 'bert-base-uncased'

            encoder_dim = self.text_encoder.config.hidden_size
            self.text_projection = nn.Linear(encoder_dim, hidden_dim)

            if use_lora:
                print(f"Applying LoRA adaptation with r={lora_r}, alpha={lora_alpha}")
                lora_config = LoraConfig(
                    task_type=TaskType.FEATURE_EXTRACTION,
                    r=lora_r,
                    lora_alpha=lora_alpha,
                    lora_dropout=0.1,
                    target_modules=["query", "key", "value", "dense"]
                )
                self.text_encoder = get_peft_model(self.text_encoder, lora_config)
                print(f"LoRA parameters: {sum(p.numel() for p in self.text_encoder.parameters() if p.requires_grad)}")
            else:
                for param in self.text_encoder.parameters():
                    param.requires_grad = False
        else:
            print(f"Using simple text encoder trained from scratch (arch: {eeg_arch})")
            self.text_encoder = SimpleTextEncoder(
                vocab_size=30522,
                hidden_dim=hidden_dim,
                arch=eeg_arch
            )
            encoder_dim = hidden_dim
            self.text_projection = nn.Identity()

        # ===== FIX 1: INITIALIZE EEG ENCODER DURING __INIT__ =====
        if global_eeg_dims is not None and query_type == 'eeg':
            num_words, time_samples, channels = global_eeg_dims
            device = next(self.text_projection.parameters()).device if hasattr(self.text_projection,
                                                                               'weight') else torch.device(
                'cuda' if torch.cuda.is_available() else 'cpu')

            if use_labram_patching:
                # ===== FIX 2: LABRAM-STYLE CHANNEL PATCHING =====
                print(f"✓ Using LaBraM-style channel patching (patch_length={labram_patch_length})")

                # Create patcher
                self.channel_patcher = LaBramChannelPatcher(
                    patch_length=labram_patch_length,
                    num_channels=channels,
                    max_time_patches=time_samples // labram_patch_length
                )

                # Create CNN preprocessor (single-channel input)
                self.cnn_preprocessor = LaBramCNNPreprocessor(
                    hidden_dim=hidden_dim,
                    num_layers=3 if eeg_arch != 'simple' else 2
                )

                # Create positional embeddings
                self.positional_embeddings = LaBramPositionalEmbeddings(
                    hidden_dim=hidden_dim,
                    max_words=num_words,
                    max_time_patches=time_samples // labram_patch_length,
                    max_channels=channels
                )

                # Create transformer to process patches
                num_patches_per_word = channels * (time_samples // labram_patch_length)
                self.eeg_encoder = EEGTransformerEncoder(
                    input_size=hidden_dim,  # Already projected by CNN
                    hidden_dim=hidden_dim,
                    num_heads=8,
                    num_layers=2,
                    dropout=0.1,
                    max_seq_len=num_patches_per_word * num_words  # Total patches
                )

                print(f"  Channel patcher: {channels} channels × {time_samples // labram_patch_length} time patches")
                print(f"  Total patches per word: {num_patches_per_word}")
                print(f"  CNN preprocessor: Conv1d(1 → {hidden_dim})")
                print(f"  Positional embeddings: temporal + spatial + word")

            else:
                # Standard approach: flatten time×channels
                input_size = time_samples * channels
                self.eeg_encoder = self._create_eeg_encoder(input_size, device)
                print(f"✓ Initialized standard EEG encoder with input size {input_size}")

                self.channel_patcher = None
                self.cnn_preprocessor = None
                self.positional_embeddings = None
        else:
            self.eeg_encoder = None
            self.channel_patcher = None
            self.cnn_preprocessor = None
            self.positional_embeddings = None
            if query_type == 'eeg':
                raise ValueError("⚠ ERROR: global_eeg_dims MUST be provided for EEG query type!")

        # EEG projection
        if eeg_arch != 'transformer' and not use_labram_patching:
            self.eeg_projection = nn.Linear(hidden_dim, hidden_dim)
        else:
            self.eeg_projection = nn.Identity()

        # Components for CLS pooling
        if pooling_strategy == 'cls':
            self.eeg_cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=8,
                dim_feedforward=hidden_dim * 2,
                dropout=0.1,
                activation='relu',
                batch_first=True
            )
            self.eeg_cls_transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)
            print(f"Initialized learnable EEG CLS token: {self.eeg_cls_token.shape}")
        else:
            self.eeg_cls_token = None
            self.eeg_cls_transformer = None

        self.dropout = nn.Dropout(dropout)

        print(f"✓ Model initialized successfully")
        print(f"  Pooling: {pooling_strategy}, Query type: {query_type}")
        print(f"  Text encoder: {colbert_model_name if use_pretrained_text else 'SimpleTextEncoder'}")
        print(f"  EEG encoder: {eeg_arch} ({'LaBraM-style' if use_labram_patching else 'standard'})")

    def set_tokenizer_vocab_size(self, tokenizer_vocab_size):
        """Update text encoder vocab size after tokenizer is created"""
        if not self.use_pretrained_text:
            device = next(self.parameters()).device
            self.text_encoder.embedding = nn.Embedding(
                tokenizer_vocab_size,
                self.hidden_dim,
                padding_idx=0
            ).to(device)

    def _create_eeg_encoder(self, input_size, device):
        """Create EEG encoder based on architecture choice (standard approach)"""

        if self.eeg_arch == 'simple':
            encoder = nn.Sequential(
                nn.Linear(input_size, self.hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, self.hidden_dim)
            )
        elif self.eeg_arch == 'complex':
            encoder = nn.Sequential(
                nn.Linear(input_size, self.hidden_dim),
                nn.BatchNorm1d(self.hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(self.hidden_dim, self.hidden_dim * 2),
                nn.BatchNorm1d(self.hidden_dim * 2),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(self.hidden_dim * 2, self.hidden_dim)
            )
        elif self.eeg_arch == 'transformer':
            encoder = EEGTransformerEncoder(
                input_size=input_size,
                hidden_dim=self.hidden_dim,
                num_heads=4,
                num_layers=1,
                dropout=0.3,
                max_seq_len=50
            )
        else:
            raise ValueError(f"Unknown EEG architecture: {self.eeg_arch}")

        return encoder.to(device)

    def encode_text(self, input_ids, attention_mask):
        """Encode text with pooling strategy"""

        if self.use_pretrained_text:
            if self.use_lora:
                outputs = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
                hidden_states = outputs.last_hidden_state
            else:
                with torch.no_grad():
                    outputs = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
                    hidden_states = outputs.last_hidden_state
        else:
            hidden_states = self.text_encoder(input_ids, attention_mask)

        projected = self.text_projection(hidden_states)
        projected = self.dropout(projected)

        batch_size = input_ids.size(0)

        if self.pooling_strategy == 'multi':
            multi_vectors = []
            for i in range(batch_size):
                valid_positions = torch.where(attention_mask[i] == 1)[0][1:]
                if len(valid_positions) > 0:
                    sample_vectors = projected[i, valid_positions]
                else:
                    sample_vectors = torch.zeros(1, projected.size(-1), device=projected.device)
                multi_vectors.append(sample_vectors)
            return multi_vectors

        elif self.pooling_strategy == 'max':
            max_vectors = []
            for i in range(batch_size):
                valid_mask = attention_mask[i] == 1
                if valid_mask.sum() > 0:
                    max_vector = torch.max(projected[i, valid_mask], dim=0)[0]
                else:
                    max_vector = torch.zeros(projected.size(-1), device=projected.device)
                max_vectors.append(max_vector.unsqueeze(0))
            return torch.stack(max_vectors)

        elif self.pooling_strategy == 'mean':
            mean_vectors = []
            for i in range(batch_size):
                valid_mask = attention_mask[i] == 1
                if valid_mask.sum() > 0:
                    mean_vector = torch.mean(projected[i, valid_mask], dim=0)
                else:
                    mean_vector = torch.zeros(projected.size(-1), device=projected.device)
                mean_vectors.append(mean_vector.unsqueeze(0))
            return torch.stack(mean_vectors)

        elif self.pooling_strategy == 'cls':
            return projected[:, 0:1, :]

    def encode_eeg(self, eeg_input, eeg_mv_mask):
        """
        Encode EEG with either standard or LaBraM-style approach

        Args:
            eeg_input: [batch, num_words, time_samples, channels]
            eeg_mv_mask: mask for multi-vector
        """
        batch_size, num_words, time_samples, channels = eeg_input.shape

        # Compute padding mask
        eeg_padding_mask = (eeg_input.abs().sum(dim=(2, 3)) == 0)  # [batch, num_words]

        if self.use_labram_patching:
            # ===== LABRAM-STYLE PROCESSING =====
            # Step 1: Create channel-wise patches
            patches, channel_indices, time_indices = self.channel_patcher(eeg_input)
            # patches: [batch, num_words, C×T, patch_length]

            # Step 2: Flatten for CNN processing
            batch_size, num_words, num_patches, patch_length = patches.shape
            patches_flat = patches.view(batch_size * num_words * num_patches, patch_length)

            # Step 3: Apply CNN to each single-channel patch
            patch_features = self.cnn_preprocessor(patches_flat)
            patch_features = patch_features.view(batch_size, num_words, num_patches, self.hidden_dim)

            # Step 4: Add positional embeddings (temporal + spatial + word)
            patch_features = self.positional_embeddings(
                patch_features,
                num_words=num_words,
                channel_indices=channel_indices,
                time_indices=time_indices
            )

            # Step 5: Flatten words and patches for transformer
            patch_features_flat = patch_features.view(batch_size, num_words * num_patches, self.hidden_dim)

            # Create padding mask for patches
            patch_padding_mask = eeg_padding_mask.unsqueeze(2).expand(-1, -1, num_patches).reshape(batch_size, -1)

            # Step 6: Apply transformer
            word_representations = self.eeg_encoder(patch_features_flat, padding_mask=patch_padding_mask)

            # Step 7: Reshape back to word-level (average patches within each word)
            word_representations = word_representations.view(batch_size, num_words, num_patches, self.hidden_dim)
            word_representations = word_representations.mean(dim=2)  # [batch, num_words, hidden_dim]

        else:
            # Standard approach: flatten time×channels
            input_size = time_samples * channels

            if self.eeg_arch == 'transformer':
                eeg_reshaped = eeg_input.view(batch_size, num_words, input_size)
                word_representations = self.eeg_encoder(eeg_reshaped, padding_mask=eeg_padding_mask)
            else:
                eeg_flat = eeg_input.view(batch_size * num_words, input_size)
                encoded = self.eeg_encoder(eeg_flat)
                word_representations = encoded.view(batch_size, num_words, self.hidden_dim)

        # Apply projection and dropout
        word_representations = self.eeg_projection(word_representations)
        word_representations = self.dropout(word_representations)

        # Apply pooling strategy
        if self.pooling_strategy == 'multi':
            multi_vectors = []
            for i in range(batch_size):
                active_positions = torch.where(~eeg_padding_mask[i])[0]
                if len(active_positions) > 0:
                    sample_vectors = word_representations[i, active_positions]
                else:
                    sample_vectors = torch.zeros(1, self.hidden_dim, device=eeg_input.device)
                multi_vectors.append(sample_vectors)
            return multi_vectors

        elif self.pooling_strategy == 'max':
            max_vectors = []
            for i in range(batch_size):
                active_positions = torch.where(~eeg_padding_mask[i])[0]
                if len(active_positions) > 0:
                    max_vector = torch.max(word_representations[i, active_positions], dim=0)[0]
                else:
                    max_vector = torch.zeros(self.hidden_dim, device=eeg_input.device)
                max_vectors.append(max_vector.unsqueeze(0))
            return torch.stack(max_vectors)

        elif self.pooling_strategy == 'mean':
            mean_vectors = []
            for i in range(batch_size):
                active_positions = torch.where(~eeg_padding_mask[i])[0]
                if len(active_positions) > 0:
                    mean_vector = torch.mean(word_representations[i, active_positions], dim=0)
                else:
                    mean_vector = torch.zeros(self.hidden_dim, device=eeg_input.device)
                mean_vectors.append(mean_vector.unsqueeze(0))
            return torch.stack(mean_vectors)

        elif self.pooling_strategy == 'cls':
            cls_tokens = self.eeg_cls_token.expand(batch_size, -1, -1)
            cls_word_sequence = torch.cat([cls_tokens, word_representations], dim=1)
            cls_mask = torch.zeros(batch_size, 1, device=eeg_input.device, dtype=torch.bool)
            full_mask = torch.cat([cls_mask, eeg_padding_mask], dim=1)
            attended_sequence = self.eeg_cls_transformer(cls_word_sequence, src_key_padding_mask=full_mask)
            return attended_sequence[:, 0:1, :]

    def forward(self, eeg_queries, text_queries, docs, eeg_mv_masks):
        """Complete forward pass"""

        doc_vectors = self.encode_text(docs['input_ids'], docs['attention_mask'])

        if self.query_type == 'eeg':
            query_vectors = self.encode_eeg(eeg_queries, eeg_mv_masks)
        else:
            query_vectors = self.encode_text(text_queries['input_ids'], text_queries['attention_mask'])

        return {
            'query_vectors': query_vectors,
            'doc_vectors': doc_vectors,
            'eeg_vectors': None if self.query_type == 'text' else query_vectors
        }


class EEGTransformerEncoder(nn.Module):
    """
    FIXED Transformer encoder with positional encodings
    """

    def __init__(self, input_size, hidden_dim=768, num_heads=4, num_layers=1,
                 dropout=0.3, max_seq_len=50):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.max_seq_len = max_seq_len

        # Input projection
        if input_size != hidden_dim:
            self.input_projection = nn.Linear(input_size, hidden_dim)
        else:
            self.input_projection = nn.Identity()

        # Learnable positional encodings
        self.positional_encoding = nn.Embedding(max_seq_len, hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 2,
            dropout=dropout,
            activation='relu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.output_projection = nn.Linear(hidden_dim, hidden_dim)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

        print(f"✓ EEGTransformerEncoder: {num_layers} layers, {num_heads} heads, dropout={dropout}")
        print(f"  Positional encoding: learnable, max_len={max_seq_len}")

    def forward(self, x, padding_mask=None):
        """
        Args:
            x: [batch, seq_len, input_size]
            padding_mask: [batch, seq_len] - True = padded
        Returns:
            [batch, seq_len, hidden_dim]
        """
        batch_size, seq_len, _ = x.shape

        # Project input
        x = self.input_projection(x)

        # Add positional encodings
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, -1)
        positions = torch.clamp(positions, max=self.max_seq_len - 1)  # Clamp to max
        pos_encodings = self.positional_encoding(positions)
        x = x + pos_encodings
        x = self.dropout(x)

        # Create padding mask if not provided
        if padding_mask is None:
            padding_mask = (x.abs().sum(dim=-1) == 0)

        # Apply transformer
        x = self.transformer(x, src_key_padding_mask=padding_mask)

        # Final projection
        x = self.output_projection(x)
        x = self.layer_norm(x)
        x = self.dropout(x)

        return x


def compute_similarity(query_vectors, doc_vectors, pooling_strategy, temperature=1.0):
    """Compute similarity based on pooling strategy"""

    if pooling_strategy == 'multi':
        return compute_multi_vector_similarity(query_vectors, doc_vectors, temperature)
    elif pooling_strategy in ['cls', 'max', 'mean']:
        return compute_cls_similarity(query_vectors, doc_vectors, temperature)
    else:
        raise ValueError(f"Unknown pooling strategy: {pooling_strategy}")


def compute_multi_vector_similarity(query_vectors, doc_vectors, temperature=1.0):
    """Compute ColBERT-style MaxSim similarity"""

    if isinstance(query_vectors, list):
        similarities = []

        for i in range(len(query_vectors)):
            q_vecs = query_vectors[i]
            d_vecs = doc_vectors[i]

            q_vecs = F.normalize(q_vecs, p=2, dim=1)
            d_vecs = F.normalize(d_vecs, p=2, dim=1)

            q_nonzero = q_vecs[q_vecs.norm(dim=1) > 1e-6]
            d_nonzero = d_vecs[d_vecs.norm(dim=1) > 1e-6]

            if len(q_nonzero) == 0 or len(d_nonzero) == 0:
                similarities.append(torch.tensor(0.0, device=q_vecs.device))
                continue

            sim_matrix = torch.matmul(q_nonzero, d_nonzero.t())
            max_sims = sim_matrix.max(dim=1)[0]
            sim = max_sims.sum()
            similarities.append(sim)

        return torch.stack(similarities) / temperature
    else:
        raise ValueError("Multi-vector similarity requires list input")


def compute_cls_similarity(query_vectors, doc_vectors, temperature=1.0):
    """Compute cosine similarity for single vectors"""

    if isinstance(query_vectors, list):
        similarities = []
        for i in range(len(query_vectors)):
            q_vec = query_vectors[i].squeeze()
            d_vec = doc_vectors[i].squeeze()
            q_norm = F.normalize(q_vec, p=2, dim=0)
            d_norm = F.normalize(d_vec, p=2, dim=0)
            sim = torch.dot(q_norm, d_norm)
            similarities.append(sim)
        return torch.stack(similarities) / temperature
    else:
        batch_size = query_vectors.size(0)
        similarities = []
        for i in range(batch_size):
            q_vec = query_vectors[i].squeeze()
            d_vec = doc_vectors[i].squeeze()
            q_norm = F.normalize(q_vec, p=2, dim=0)
            d_norm = F.normalize(d_vec, p=2, dim=0)
            sim = torch.dot(q_norm, d_norm)
            similarities.append(sim)
        return torch.stack(similarities) / temperature


def create_model(colbert_model_name='colbert-ir/colbertv2.0', hidden_dim=768,
                 eeg_arch='simple', device='cuda', use_lora=True, lora_r=16,
                 lora_alpha=32, pooling_strategy='multi', encoder_type='dual',
                 global_eeg_dims=None, query_type='eeg',
                 use_pretrained_text=True, use_labram_patching=False,
                 labram_patch_length=200):
    """
    Create model with ALL FIXES:
    1. EEG encoder initialized during __init__ ✓
    2. Positional encodings ✓
    3. Optional LaBraM-style channel patching ✓
    4. Proper parameter registration ✓
    """

    if encoder_type == 'dual':
        model = SimplifiedBrainRetrieval(
            colbert_model_name=colbert_model_name,
            hidden_dim=hidden_dim,
            eeg_arch=eeg_arch,
            use_lora=use_lora,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            pooling_strategy=pooling_strategy,
            query_type=query_type,
            use_pretrained_text=use_pretrained_text,
            global_eeg_dims=global_eeg_dims,
            use_labram_patching=use_labram_patching,
            labram_patch_length=labram_patch_length
        )
    else:
        raise NotImplementedError("Only dual encoder supported in this fixed version")

    return model.to(device)