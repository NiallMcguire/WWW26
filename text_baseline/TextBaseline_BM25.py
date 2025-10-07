#!/usr/bin/env python3
"""
BM25 Text Retrieval Baseline for Brain Passage Retrieval - Pooled Documents
Evaluates BM25 with pooled documents from all datasets
Provides baseline comparison for neural EEG retrieval
"""

import numpy as np
import random
import argparse
from pathlib import Path
from typing import List, Dict
import wandb
from tqdm import tqdm
import tempfile
import os

# BM25 import
from rank_bm25 import BM25Okapi

# Import your existing dataloader and utilities
from mv_dataloader import (
    DynamicMaskingDataloader,
    compute_global_eeg_dimensions,
    load_combined_datasets,
    compute_combined_eeg_dimensions
)
from transformers import AutoTokenizer


def set_seeds(seed=42):
    """Set all random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    print(f"Set random seed to {seed}")


class BM25Retriever:
    """BM25 text retriever with preprocessing"""

    def __init__(self):
        self.bm25 = None
        self.documents = None
        print("BM25 retriever initialized")

    def fit(self, documents: List[str]):
        """Fit BM25 on document corpus"""
        tokenized_docs = [doc.lower().split() for doc in documents]
        self.bm25 = BM25Okapi(tokenized_docs)
        self.documents = documents

    def get_scores(self, query: str) -> List[float]:
        """Get BM25 scores for a query against all documents"""
        if self.bm25 is None:
            raise ValueError("BM25 not fitted. Call fit() first.")

        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)
        return scores.tolist()


def load_test_data(data_path: str, tokenizer, max_text_len: int = 256,
                   max_eeg_len: int = 50, dataset_type: str = 'auto',
                   holdout_subjects: bool = False, fold: int = None,
                   global_eeg_dims: tuple = None) -> DynamicMaskingDataloader:
    """Load test dataset"""

    print(f"Loading test data from {data_path}...")

    test_dataset = DynamicMaskingDataloader(
        data_path=data_path,
        tokenizer=tokenizer,
        max_text_len=max_text_len,
        max_eeg_len=max_eeg_len,
        train_ratio=0.8,
        debug=False,
        global_eeg_dims=global_eeg_dims,
        num_vectors=32,
        dataset_type=dataset_type,
        initial_masking_probability=0.9,
        split='val'
    )

    print(f"Loaded test dataset: {len(test_dataset)} samples")
    return test_dataset


def pool_test_data(test_dataset: DynamicMaskingDataloader, masking_level: int) -> Dict:
    """
    Pool all documents together regardless of dataset source
    Returns dict with queries, pooled documents, mappings, and dataset labels
    """

    original_prob = test_dataset.get_current_masking_probability()
    test_dataset.set_masking_probability(masking_level / 100.0)

    try:
        queries = []
        unique_docs = {}
        query_to_doc_mapping = {}
        query_dataset_labels = []

        for idx in range(len(test_dataset)):
            sample = test_dataset[idx]

            query_text = sample['metadata']['query_text']
            doc_text = sample['metadata']['document_text']
            dataset_source = sample['metadata'].get('dataset_source', 'unknown')

            if 'nieuwland' in dataset_source.lower() or 'dataset_1' in dataset_source.lower():
                dataset_name = 'nieuwland'
            elif 'alice' in dataset_source.lower() or 'dataset_2' in dataset_source.lower():
                dataset_name = 'alice'
            else:
                dataset_name = 'unknown'

            query_idx = len(queries)
            queries.append(query_text)
            query_dataset_labels.append(dataset_name)

            if doc_text.strip():
                if doc_text not in unique_docs:
                    unique_doc_idx = len(unique_docs)
                    unique_docs[doc_text] = unique_doc_idx
                else:
                    unique_doc_idx = unique_docs[doc_text]

                query_to_doc_mapping[query_idx] = unique_doc_idx

        doc_list = [''] * len(unique_docs)
        for text, idx in unique_docs.items():
            doc_list[idx] = text

        return {
            'queries': queries,
            'doc_list': doc_list,
            'query_to_doc_mapping': query_to_doc_mapping,
            'query_dataset_labels': query_dataset_labels
        }

    finally:
        test_dataset.set_masking_probability(original_prob)


def generate_consistent_subsets(doc_list: List[str], query_to_doc_mapping: Dict[int, int],
                                subset_size: int = 100, seed: int = 42) -> Dict[int, List[int]]:
    """Generate consistent document subsets for fair comparison"""

    random.seed(seed)
    query_subsets = {}

    for query_idx, correct_doc_idx in query_to_doc_mapping.items():
        doc_subset_indices = [correct_doc_idx]
        negative_candidates = [i for i in range(len(doc_list)) if i != correct_doc_idx]

        if negative_candidates:
            random_negatives = random.sample(negative_candidates,
                                             min(subset_size - 1, len(negative_candidates)))
            doc_subset_indices.extend(random_negatives)

        query_subsets[query_idx] = doc_subset_indices

    return query_subsets


def compute_ranking_metrics(ranked_doc_indices: List[int], correct_doc_idx: int,
                            k_values: List[int] = [1, 5, 10, 20]) -> Dict[str, float]:
    """Compute ranking metrics"""
    metrics = {}

    try:
        correct_rank = ranked_doc_indices.index(correct_doc_idx) + 1
    except ValueError:
        correct_rank = len(ranked_doc_indices) + 1

    metrics['rr'] = 1.0 / correct_rank if correct_rank <= len(ranked_doc_indices) else 0.0
    metrics['rank_of_correct'] = correct_rank

    for k in k_values:
        if k <= len(ranked_doc_indices):
            hit_at_k = 1.0 if correct_rank <= k else 0.0
            metrics[f'hit_at_{k}'] = hit_at_k
            metrics[f'precision_at_{k}'] = hit_at_k / k
            metrics[f'recall_at_{k}'] = hit_at_k
        else:
            metrics[f'hit_at_{k}'] = 0.0
            metrics[f'precision_at_{k}'] = 0.0
            metrics[f'recall_at_{k}'] = 0.0

    return metrics


def evaluate_bm25_ranking(bm25_retriever: BM25Retriever, queries: List[str],
                          doc_list: List[str], query_to_doc_mapping: Dict[int, int],
                          query_subsets: Dict[int, List[int]],
                          query_dataset_labels: List[str] = None) -> tuple:
    """Evaluate BM25 ranking performance"""

    all_metrics = []
    per_dataset_metrics = {}

    for query_idx, query_text in enumerate(tqdm(queries, desc="BM25 ranking")):
        if query_idx not in query_to_doc_mapping:
            continue

        correct_doc_idx = query_to_doc_mapping[query_idx]
        doc_subset_indices = query_subsets[query_idx]

        all_scores = bm25_retriever.get_scores(query_text)
        subset_scores = [all_scores[idx] for idx in doc_subset_indices]

        doc_score_pairs = list(zip(doc_subset_indices, subset_scores))
        doc_score_pairs.sort(key=lambda x: x[1], reverse=True)
        ranked_indices = [doc_idx for doc_idx, score in doc_score_pairs]

        query_metrics = compute_ranking_metrics(ranked_indices, correct_doc_idx)
        all_metrics.append(query_metrics)

        if query_dataset_labels:
            dataset_label = query_dataset_labels[query_idx]
            if dataset_label not in per_dataset_metrics:
                per_dataset_metrics[dataset_label] = []
            per_dataset_metrics[dataset_label].append(query_metrics)

    return all_metrics, per_dataset_metrics


def aggregate_metrics(all_metrics: List[Dict[str, float]], prefix: str) -> Dict[str, float]:
    """Aggregate ranking metrics across queries"""

    if not all_metrics:
        return {}

    aggregated = {}
    metric_names = all_metrics[0].keys()

    for metric_name in metric_names:
        values = [m[metric_name] for m in all_metrics]
        aggregated[f'{prefix}/{metric_name}'] = np.mean(values)
        if metric_name != 'rank_of_correct':
            aggregated[f'{prefix}/{metric_name}_std'] = np.std(values)

    return aggregated


def initialize_wandb(config: Dict):
    """Initialize wandb logging"""

    dataset_name = config.get('dataset_name', 'unknown')
    holdout_subjects = config.get('holdout_subjects', False)
    fold = config.get('fold', None)

    if holdout_subjects and fold is not None:
        split_suffix = f"_holdout_fold{fold}"
    elif holdout_subjects:
        split_suffix = "_holdout"
    else:
        split_suffix = "_random"

    run_name = f"text_baseline_bm25_pooled{split_suffix}_{dataset_name}"

    tags = ['text-retrieval-baseline', 'bm25', 'brain-retrieval-comparison',
            'holdout-subjects' if holdout_subjects else 'random-split',
            f'dataset-{dataset_name}', 'multi-masking-evaluation', 'pooled-documents']

    wandb.init(
        project="project",
        name=run_name,
        config={
            'experiment_type': 'text_retrieval_baseline_bm25_pooled',
            'retrieval_method': 'bm25',
            'dataset_name': dataset_name,
            'holdout_subjects': holdout_subjects,
            'fold': fold,
            'split_method': 'holdout_subjects' if holdout_subjects else 'random_samples',
            'test_masking_levels': config.get('test_masking_levels', []),
            'test_samples': config.get('test_samples', 0),
            'baseline_type': 'text_retrieval_pooled',
            'seed': config.get('seed', 42),
        },
        tags=tags
    )


def main():
    parser = argparse.ArgumentParser(description='BM25 Text Retrieval Baseline - Pooled Documents')

    parser.add_argument('--data_path', help='Path to single ICT pairs .npy file')
    parser.add_argument('--data_paths', nargs='+', help='Paths to multiple ICT pairs .npy files')
    parser.add_argument('--dataset_type', default='auto', choices=['auto', 'original', 'nieuwland'])
    parser.add_argument('--dataset_types', nargs='*', default=None)
    parser.add_argument('--test_masking_levels', nargs='+', type=int,
                        default=[0, 25, 50, 75, 90, 100])
    parser.add_argument('--subset_size', type=int, default=100)
    parser.add_argument('--max_text_len', type=int, default=256)
    parser.add_argument('--max_eeg_len', type=int, default=50)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--holdout_subjects', action='store_true')
    parser.add_argument('--fold', type=int, choices=[1, 2, 3, 4, 5], default=None)

    args = parser.parse_args()

    if not args.data_path and not args.data_paths:
        raise ValueError("Must specify either --data_path or --data_paths")

    if args.fold is not None and not args.holdout_subjects:
        raise ValueError("--fold parameter requires --holdout_subjects")

    set_seeds(args.seed)

    # Handle multiple datasets by combining them into a temp file
    temp_file = None
    if args.data_paths and len(args.data_paths) > 1:
        print(f"Combining {len(args.data_paths)} datasets...")

        all_ict_pairs, combined_metadata = load_combined_datasets(
            args.data_paths, args.dataset_types
        )

        combined_metadata['supports_runtime_masking'] = True
        combined_metadata['version'] = 'combined_dataset'

        combined_dataset = {
            'ict_pairs': all_ict_pairs,
            'metadata': combined_metadata
        }

        # Save to temp file
        temp_file = tempfile.NamedTemporaryFile(mode='wb', suffix='.npy', delete=False)
        np.save(temp_file.name, combined_dataset)
        temp_file.close()

        data_path = temp_file.name
        dataset_name = "combined"
        global_eeg_dims = compute_combined_eeg_dimensions(all_ict_pairs, args.max_eeg_len)
        dataset_type = 'original'

        print(f"Combined dataset saved to temporary file: {data_path}")
    else:
        data_path = args.data_path or args.data_paths[0]
        filename = Path(data_path).name.lower()

        if 'nieuwland' in filename and 'alice' in filename:
            dataset_name = "combined"
        elif 'nieuwland' in filename:
            dataset_name = "nieuwland"
        elif 'alice' in filename:
            dataset_name = "alice"
        else:
            dataset_name = "single"

        dataset_type = args.dataset_type
        global_eeg_dims = compute_global_eeg_dimensions(data_path, args.max_eeg_len, dataset_type)

    try:
        tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

        test_dataset = load_test_data(
            data_path, tokenizer, args.max_text_len, args.max_eeg_len,
            dataset_type, args.holdout_subjects, args.fold, global_eeg_dims
        )

        config = {
            'dataset_name': dataset_name,
            'holdout_subjects': args.holdout_subjects,
            'fold': args.fold,
            'test_masking_levels': args.test_masking_levels,
            'test_samples': len(test_dataset),
            'seed': args.seed,
            'subset_size': args.subset_size
        }

        initialize_wandb(config)

        print("\n" + "=" * 80)
        print("BM25 TEXT RETRIEVAL BASELINE - POOLED DOCUMENTS")
        print("=" * 80)
        print(f"Dataset: {dataset_name}")
        print(f"Test samples: {len(test_dataset)}")
        print(f"Masking levels: {args.test_masking_levels}")
        print(f"Subset size: {args.subset_size}")

        all_results = {}

        for masking_level in args.test_masking_levels:
            print(f"\n{'=' * 20} MASKING LEVEL: {masking_level}% {'=' * 20}")

            pooled_data = pool_test_data(test_dataset, masking_level)

            queries = pooled_data['queries']
            doc_list = pooled_data['doc_list']
            query_to_doc_mapping = pooled_data['query_to_doc_mapping']
            query_dataset_labels = pooled_data['query_dataset_labels']

            print(f"Pooled: {len(queries)} queries, {len(doc_list)} unique documents")

            dataset_counts = {}
            for label in query_dataset_labels:
                dataset_counts[label] = dataset_counts.get(label, 0) + 1
            print(f"Query distribution: {dataset_counts}")

            if len(doc_list) == 0 or len(queries) == 0:
                print(f"No valid data - skipping")
                continue

            query_subsets = generate_consistent_subsets(
                doc_list, query_to_doc_mapping, args.subset_size, args.seed
            )

            bm25_retriever = BM25Retriever()
            print(f"Fitting BM25 on {len(doc_list)} pooled documents...")
            bm25_retriever.fit(doc_list)

            bm25_metrics, per_dataset_metrics = evaluate_bm25_ranking(
                bm25_retriever, queries, doc_list, query_to_doc_mapping,
                query_subsets, query_dataset_labels
            )

            prefix = f'text_baseline/bm25/pooled/masking_{masking_level}'
            bm25_aggregated = aggregate_metrics(bm25_metrics, prefix)

            per_dataset_aggregated = {}
            for ds_name, ds_metrics in per_dataset_metrics.items():
                ds_prefix = f'text_baseline/bm25/{ds_name}/masking_{masking_level}'
                ds_aggregated = aggregate_metrics(ds_metrics, ds_prefix)
                per_dataset_aggregated.update(ds_aggregated)

            metadata = {
                f'{prefix}/num_queries': len(queries),
                f'{prefix}/num_unique_docs': len(doc_list),
                f'{prefix}/subset_size': args.subset_size,
                f'{prefix}/masking_level': masking_level
            }

            level_results = {**bm25_aggregated, **per_dataset_aggregated, **metadata}
            all_results.update(level_results)

            if bm25_metrics:
                mrr = np.mean([m['rr'] for m in bm25_metrics])
                hit1 = np.mean([m['hit_at_1'] for m in bm25_metrics])
                hit5 = np.mean([m['hit_at_5'] for m in bm25_metrics])
                hit10 = np.mean([m['hit_at_10'] for m in bm25_metrics])
                hit20 = np.mean([m['hit_at_20'] for m in bm25_metrics])
                p1 = np.mean([m['precision_at_1'] for m in bm25_metrics])
                p5 = np.mean([m['precision_at_5'] for m in bm25_metrics])
                p20 = np.mean([m['precision_at_20'] for m in bm25_metrics])
                avg_rank = np.mean([m['rank_of_correct'] for m in bm25_metrics])

                print(f"\nOverall Results (Pooled):")
                print(f"  MRR: {mrr:.4f}, H@1: {hit1:.4f}, H@5: {hit5:.4f}, H@10: {hit10:.4f}, H@20: {hit20:.4f}")
                print(f"  P@1: {p1:.4f}, P@5: {p5:.4f}, P@20: {p20:.4f}, Rank: {avg_rank:.2f}")

                for ds_name, ds_metrics in per_dataset_metrics.items():
                    ds_mrr = np.mean([m['rr'] for m in ds_metrics])
                    ds_hit1 = np.mean([m['hit_at_1'] for m in ds_metrics])
                    ds_hit5 = np.mean([m['hit_at_5'] for m in ds_metrics])
                    ds_hit10 = np.mean([m['hit_at_10'] for m in ds_metrics])
                    ds_hit20 = np.mean([m['hit_at_20'] for m in ds_metrics])
                    ds_p1 = np.mean([m['precision_at_1'] for m in ds_metrics])
                    ds_p5 = np.mean([m['precision_at_5'] for m in ds_metrics])
                    ds_p20 = np.mean([m['precision_at_20'] for m in ds_metrics])
                    ds_avg_rank = np.mean([m['rank_of_correct'] for m in ds_metrics])
                    print(f"\n  {ds_name} subset ({len(ds_metrics)} queries):")
                    print(
                        f"    MRR: {ds_mrr:.4f}, H@1: {ds_hit1:.4f}, H@5: {ds_hit5:.4f}, H@10: {ds_hit10:.4f}, H@20: {ds_hit20:.4f}")
                    print(f"    P@1: {ds_p1:.4f}, P@5: {ds_p5:.4f}, P@20: {ds_p20:.4f}, Rank: {ds_avg_rank:.2f}")

        if all_results:
            wandb.log(all_results)

            print(f"\n{'=' * 80}")
            print("BM25 BASELINE SUMMARY - POOLED DOCUMENTS")
            print(f"{'=' * 80}")

            print(f"\nOverall (Pooled):")
            print(
                f"{'Mask':>6} {'MRR':>8} {'H@1':>8} {'H@5':>8} {'H@10':>8} {'H@20':>8} {'P@1':>8} {'P@5':>8} {'P@20':>8} {'Rank':>8} {'Docs':>8}")
            print("-" * 98)

            # Collect metrics across masking levels for mean/std calculation
            mrr_vals, h1_vals, h5_vals, h10_vals, h20_vals = [], [], [], [], []

            for masking_level in args.test_masking_levels:
                mrr_key = f'text_baseline/bm25/pooled/masking_{masking_level}/rr'
                docs_key = f'text_baseline/bm25/pooled/masking_{masking_level}/num_unique_docs'
                rank_key = f'text_baseline/bm25/pooled/masking_{masking_level}/rank_of_correct'

                if mrr_key in all_results:
                    mrr = all_results[mrr_key]
                    hit1 = all_results[f'text_baseline/bm25/pooled/masking_{masking_level}/hit_at_1']
                    hit5 = all_results[f'text_baseline/bm25/pooled/masking_{masking_level}/hit_at_5']
                    hit10 = all_results[f'text_baseline/bm25/pooled/masking_{masking_level}/hit_at_10']
                    hit20 = all_results[f'text_baseline/bm25/pooled/masking_{masking_level}/hit_at_20']
                    p1 = all_results[f'text_baseline/bm25/pooled/masking_{masking_level}/precision_at_1']
                    p5 = all_results[f'text_baseline/bm25/pooled/masking_{masking_level}/precision_at_5']
                    p20 = all_results[f'text_baseline/bm25/pooled/masking_{masking_level}/precision_at_20']
                    rank = all_results.get(rank_key, 0)
                    n_docs = all_results.get(docs_key, 0)

                    mrr_vals.append(mrr)
                    h1_vals.append(hit1)
                    h5_vals.append(hit5)
                    h10_vals.append(hit10)
                    h20_vals.append(hit20)

                    print(
                        f"{masking_level:>4}% {mrr:>8.4f} {hit1:>8.4f} {hit5:>8.4f} {hit10:>8.4f} {hit20:>8.4f} {p1:>8.4f} {p5:>8.4f} {p20:>8.4f} {rank:>8.2f} {n_docs:>8.0f}")

            # Print mean ± std for LaTeX table
            print("\n" + "=" * 98)
            print("MEAN ± STD ACROSS ALL MASKING LEVELS (for LaTeX table):")
            print(f"MRR:   {np.mean(mrr_vals):.3f} ± {np.std(mrr_vals):.3f}")
            print(f"H@1:   {np.mean(h1_vals):.3f} ± {np.std(h1_vals):.3f}")
            print(f"H@5:   {np.mean(h5_vals):.3f} ± {np.std(h5_vals):.3f}")
            print(f"H@10:  {np.mean(h10_vals):.3f} ± {np.std(h10_vals):.3f}")
            print(f"H@20:  {np.mean(h20_vals):.3f} ± {np.std(h20_vals):.3f}")
            print("=" * 98)

            # Calculate per-dataset mean ± std across masking levels
            datasets_found = set()
            for key in all_results.keys():
                if 'alice' in key and 'pooled' not in key:
                    datasets_found.add('alice')
                if 'nieuwland' in key and 'pooled' not in key:
                    datasets_found.add('nieuwland')

            # Collect per-dataset metrics across masking levels
            per_dataset_stats = {}
            for ds in sorted(datasets_found):
                ds_mrr_vals, ds_h1_vals, ds_h5_vals, ds_h10_vals, ds_h20_vals = [], [], [], [], []

                for masking_level in args.test_masking_levels:
                    mrr_key = f'text_baseline/bm25/{ds}/masking_{masking_level}/rr'
                    if mrr_key in all_results:
                        ds_mrr_vals.append(all_results[mrr_key])
                        ds_h1_vals.append(all_results[f'text_baseline/bm25/{ds}/masking_{masking_level}/hit_at_1'])
                        ds_h5_vals.append(all_results[f'text_baseline/bm25/{ds}/masking_{masking_level}/hit_at_5'])
                        ds_h10_vals.append(all_results[f'text_baseline/bm25/{ds}/masking_{masking_level}/hit_at_10'])
                        ds_h20_vals.append(all_results[f'text_baseline/bm25/{ds}/masking_{masking_level}/hit_at_20'])

                if ds_mrr_vals:
                    per_dataset_stats[ds] = {
                        'mrr': (np.mean(ds_mrr_vals), np.std(ds_mrr_vals)),
                        'h1': (np.mean(ds_h1_vals), np.std(ds_h1_vals)),
                        'h5': (np.mean(ds_h5_vals), np.std(ds_h5_vals)),
                        'h10': (np.mean(ds_h10_vals), np.std(ds_h10_vals)),
                        'h20': (np.mean(ds_h20_vals), np.std(ds_h20_vals))
                    }

            # Print per-dataset summaries
            for ds in sorted(datasets_found):
                if ds in per_dataset_stats:
                    stats = per_dataset_stats[ds]
                    print(f"\n{ds.upper()} QUERIES - MEAN ± STD ACROSS ALL MASKING LEVELS:")
                    print(f"MRR:   {stats['mrr'][0]:.3f} ± {stats['mrr'][1]:.3f}")
                    print(f"H@1:   {stats['h1'][0]:.3f} ± {stats['h1'][1]:.3f}")
                    print(f"H@5:   {stats['h5'][0]:.3f} ± {stats['h5'][1]:.3f}")
                    print(f"H@10:  {stats['h10'][0]:.3f} ± {stats['h10'][1]:.3f}")
                    print(f"H@20:  {stats['h20'][0]:.3f} ± {stats['h20'][1]:.3f}")
                    print("=" * 98)
            for key in all_results.keys():
                if 'alice' in key and 'pooled' not in key:
                    datasets_found.add('alice')
                if 'nieuwland' in key and 'pooled' not in key:
                    datasets_found.add('nieuwland')

            for ds in sorted(datasets_found):
                print(f"\n{ds.upper()} queries (searching pooled docs):")
                print(
                    f"{'Mask':>6} {'MRR':>8} {'H@1':>8} {'H@5':>8} {'H@10':>8} {'H@20':>8} {'P@1':>8} {'P@5':>8} {'P@20':>8} {'Rank':>8}")
                print("-" * 88)

                for masking_level in args.test_masking_levels:
                    mrr_key = f'text_baseline/bm25/{ds}/masking_{masking_level}/rr'
                    rank_key = f'text_baseline/bm25/{ds}/masking_{masking_level}/rank_of_correct'

                    if mrr_key in all_results:
                        mrr = all_results[mrr_key]
                        hit1 = all_results[f'text_baseline/bm25/{ds}/masking_{masking_level}/hit_at_1']
                        hit5 = all_results[f'text_baseline/bm25/{ds}/masking_{masking_level}/hit_at_5']
                        hit10 = all_results[f'text_baseline/bm25/{ds}/masking_{masking_level}/hit_at_10']
                        hit20 = all_results[f'text_baseline/bm25/{ds}/masking_{masking_level}/hit_at_20']
                        p1 = all_results[f'text_baseline/bm25/{ds}/masking_{masking_level}/precision_at_1']
                        p5 = all_results[f'text_baseline/bm25/{ds}/masking_{masking_level}/precision_at_5']
                        p20 = all_results[f'text_baseline/bm25/{ds}/masking_{masking_level}/precision_at_20']
                        rank = all_results.get(rank_key, 0)

                        print(
                            f"{masking_level:>4}% {mrr:>8.4f} {hit1:>8.4f} {hit5:>8.4f} {hit10:>8.4f} {hit20:>8.4f} {p1:>8.4f} {p5:>8.4f} {p20:>8.4f} {rank:>8.2f}")

        wandb.finish()
        print(f"\n{'=' * 80}")
        print("BM25 POOLED BASELINE EVALUATION COMPLETE")
        print(f"{'=' * 80}")

    finally:
        # Clean up temp file
        if temp_file is not None:
            try:
                os.unlink(temp_file.name)
                print(f"Cleaned up temporary file: {temp_file.name}")
            except:
                pass


if __name__ == "__main__":
    main()