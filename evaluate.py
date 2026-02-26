"""
Standalone evaluation script for CrosswordTRM.

Usage:
    python evaluate.py \
        --checkpoint checkpoints/Crossword-1k-ACT-torch/crossword_overnight/step_40680 \
        --data data/crossword-1k \
        [--batch-size 64] [--device cuda]
"""
import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import DataLoader

from puzzle_dataset import PuzzleDataset, PuzzleDatasetConfig, PuzzleDatasetMetadata
from utils.functions import load_model_class
from models.losses import IGNORE_LABEL_ID


def load_model(checkpoint_path: str, metadata: PuzzleDatasetMetadata, device: str, batch_size: int):
    """Load CrosswordTRM + ACTLossHead from a checkpoint."""
    # Import classes
    model_cls = load_model_class("recursive_reasoning.trm_crossword@CrosswordTRM")
    loss_cls = load_model_class("losses@ACTLossHead")

    config_dict = dict(
        batch_size=batch_size,
        vocab_size=metadata.vocab_size,
        seq_len=metadata.seq_len,
        num_puzzle_identifiers=metadata.num_puzzle_identifiers,
        causal=False,
        # Architecture params (must match training)
        halt_exploration_prob=0.1,
        halt_max_steps=16,
        H_cycles=3,
        L_cycles=6,
        H_layers=0,
        L_layers=2,
        hidden_size=512,
        num_heads=8,
        expansion=4,
        puzzle_emb_ndim=0,
        puzzle_emb_len=0,
        pos_encodings="rope",
        forward_dtype="bfloat16",
        mlp_t=False,
        no_ACT_continue=True,
        clue_emb_dim=384,
    )

    with torch.device(device):
        model = model_cls(config_dict)
        model = loss_cls(model, loss_type="stablemax_cross_entropy")
        model = torch.compile(model)

    # Load weights
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict, assign=True)
    model.eval()
    print(f"Loaded checkpoint from {checkpoint_path}")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

    return model


def evaluate(model, eval_loader, device: str):
    """Run evaluation and return metrics."""
    total_cells = 0
    total_correct = 0
    total_exact = 0
    total_puzzles = 0
    total_steps = 0
    total_batches = 0

    # Per-puzzle tracking for detailed output
    puzzle_results = []

    with torch.inference_mode():
        for set_name, batch, global_batch_size in eval_loader:
            total_batches += 1
            batch = {k: v.to(device) for k, v in batch.items()}

            # Initialize carry (must match device context used in training)
            with torch.device(device):
                carry = model.initial_carry(batch)

            # Run ACT inference loop
            inference_steps = 0
            while True:
                carry, loss, metrics, preds, all_finish = model(
                    carry=carry, batch=batch, return_keys=["preds"]
                )
                inference_steps += 1
                if all_finish:
                    break

            # Extract per-example stats
            labels = carry.current_data["labels"]
            predictions = preds["preds"]
            mask = (labels != IGNORE_LABEL_ID)

            batch_size = labels.shape[0]
            for i in range(batch_size):
                cell_mask = mask[i]
                n_cells = cell_mask.sum().item()
                if n_cells == 0:
                    continue  # padding example

                correct = (predictions[i][cell_mask] == labels[i][cell_mask]).sum().item()
                is_exact = (correct == n_cells)

                total_cells += n_cells
                total_correct += correct
                total_exact += int(is_exact)
                total_puzzles += 1
                total_steps += carry.steps[i].item()

                puzzle_results.append({
                    "cell_accuracy": correct / n_cells,
                    "exact_match": is_exact,
                    "n_cells": n_cells,
                    "steps": carry.steps[i].item(),
                })

            print(f"  Batch {total_batches} ({set_name}): "
                  f"{inference_steps} ACT steps, "
                  f"running accuracy={total_correct/max(total_cells,1):.4f}")

    # Aggregate
    cell_accuracy = total_correct / max(total_cells, 1)
    exact_accuracy = total_exact / max(total_puzzles, 1)
    avg_steps = total_steps / max(total_puzzles, 1)

    return {
        "cell_accuracy": cell_accuracy,
        "exact_accuracy": exact_accuracy,
        "total_puzzles": total_puzzles,
        "total_cells": total_cells,
        "total_correct": total_correct,
        "total_exact": total_exact,
        "avg_act_steps": avg_steps,
        "puzzle_results": puzzle_results,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate CrosswordTRM on test set")
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint file (e.g. step_40680)")
    parser.add_argument("--data", required=True, help="Path to dataset dir (e.g. data/crossword-1k)")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for evaluation")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    # Create test dataloader
    print(f"Loading test data from {args.data} ...")
    eval_dataset = PuzzleDataset(PuzzleDatasetConfig(
        seed=0,
        dataset_paths=[args.data],
        global_batch_size=args.batch_size,
        test_set_mode=True,
        epochs_per_iter=1,
        rank=0,
        num_replicas=1,
    ), split="test")
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=None,
        num_workers=0,  # simpler for eval
    )
    metadata = eval_dataset.metadata
    print(f"  vocab_size={metadata.vocab_size}, seq_len={metadata.seq_len}, "
          f"puzzles={metadata.total_puzzles}, groups={metadata.total_groups}")

    # Load model
    print(f"\nLoading model from {args.checkpoint} ...")
    model = load_model(args.checkpoint, metadata, args.device, args.batch_size)

    # Evaluate
    print(f"\nEvaluating on test set ...")
    results = evaluate(model, eval_loader, args.device)

    # Print results
    print("\n" + "=" * 60)
    print("TEST SET RESULTS")
    print("=" * 60)
    print(f"  Puzzles evaluated:   {results['total_puzzles']}")
    print(f"  Total cells:         {results['total_cells']}")
    print(f"  Per-cell accuracy:   {results['cell_accuracy']:.4f} ({results['total_correct']}/{results['total_cells']})")
    print(f"  Exact puzzle match:  {results['exact_accuracy']:.4f} ({results['total_exact']}/{results['total_puzzles']})")
    print(f"  Avg ACT steps:       {results['avg_act_steps']:.2f}")
    print("=" * 60)

    # Distribution of cell accuracy
    accs = [r["cell_accuracy"] for r in results["puzzle_results"]]
    if accs:
        accs_arr = np.array(accs)
        print(f"\n  Cell accuracy distribution:")
        print(f"    Min:    {accs_arr.min():.4f}")
        print(f"    25th:   {np.percentile(accs_arr, 25):.4f}")
        print(f"    Median: {np.median(accs_arr):.4f}")
        print(f"    75th:   {np.percentile(accs_arr, 75):.4f}")
        print(f"    Max:    {accs_arr.max():.4f}")

    # Save results to JSON
    out_dir = os.path.dirname(args.checkpoint)
    out_path = os.path.join(out_dir, "test_results.json")
    save_results = {k: v for k, v in results.items() if k != "puzzle_results"}
    save_results["per_puzzle"] = results["puzzle_results"]
    with open(out_path, "w") as f:
        json.dump(save_results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
