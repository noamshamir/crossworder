"""
Build a crossword dataset for TRM from .puz files.

Uses a frozen sentence-transformer to encode clue texts as per-cell embeddings,
which are injected into the TRM model during training.

Usage:
    python dataset/build_crossword_dataset.py --output-dir data/crossword-1k --num-aug 100

Token vocab (29 tokens):
    0 = PAD
    1 = BLACK (block cell)
    2 = BLANK (unfilled white cell)
    3..28 = A..Z
"""
from typing import Optional
import os
import glob
import json
import numpy as np

from argdantic import ArgParser
from pydantic import BaseModel
from tqdm import tqdm

import puz  # puzpy library for reading .puz files
from sentence_transformers import SentenceTransformer

from dataset.common import PuzzleDatasetMetadata


cli = ArgParser()


class CrosswordDataConfig(BaseModel):
    input_dirs: list = []
    test_input_dirs: list = []  # if set, use these dirs for test instead of splitting input_dirs
    output_dir: str = "data/crossword-1k"
    test_ratio: float = 0.1
    num_aug: int = 100        # augmented variants per puzzle (random partial reveals)
    reveal_min: float = 0.0   # min fraction of white cells to reveal
    reveal_max: float = 0.5   # max fraction of white cells to reveal
    clue_model: str = "all-MiniLM-L6-v2"  # sentence-transformer model name
    seed: int = 42


# ---------------------------------------------------------------------------
# .puz parsing
# ---------------------------------------------------------------------------

def parse_puz_file(fname: str) -> dict:
    """Parse a .puz file and return grid + clue information."""
    p = puz.read(fname)
    numbering = p.clue_numbering()

    rows, cols = p.height, p.width

    # Build solution grid: None for black, uppercase letter for white
    grid = []
    for r in range(rows):
        row = []
        for c in range(cols):
            ch = p.solution[r * cols + c]
            if ch == '.':
                row.append(None)
            else:
                ch = ch.upper()
                if not ch.isalpha():
                    raise ValueError(f"Non-alpha character '{ch}' in solution — skipping rebus/meta puzzle")
                row.append(ch)
        grid.append(row)

    # Extract across clues with cell positions
    across_clues = {}
    for clue in numbering.across:
        r, c = clue['cell'] // cols, clue['cell'] % cols
        length = clue['len']
        answer = ''.join(
            p.solution[clue['cell'] + i] for i in range(length)
        ).upper()
        cells = [(r, c + i) for i in range(length)]
        across_clues[f"{clue['num']}A"] = {
            'text': clue['clue'],
            'answer': answer,
            'cells': cells,
        }

    # Extract down clues with cell positions
    down_clues = {}
    for clue in numbering.down:
        r, c = clue['cell'] // cols, clue['cell'] % cols
        length = clue['len']
        answer = ''.join(
            p.solution[clue['cell'] + i * cols] for i in range(length)
        ).upper()
        cells = [(r + i, c) for i in range(length)]
        down_clues[f"{clue['num']}D"] = {
            'text': clue['clue'],
            'answer': answer,
            'cells': cells,
        }

    # Build cell → clue mappings
    cell_to_across = {}
    for clue_id, info in across_clues.items():
        for pos_in_word, (r, c) in enumerate(info['cells']):
            cell_to_across[(r, c)] = clue_id

    cell_to_down = {}
    for clue_id, info in down_clues.items():
        for pos_in_word, (r, c) in enumerate(info['cells']):
            cell_to_down[(r, c)] = clue_id

    return {
        'rows': rows,
        'cols': cols,
        'grid': grid,
        'across_clues': across_clues,
        'down_clues': down_clues,
        'cell_to_across': cell_to_across,
        'cell_to_down': cell_to_down,
        'filename': os.path.basename(fname),
    }


# ---------------------------------------------------------------------------
# Clue encoding
# ---------------------------------------------------------------------------

def encode_clues(puzzles: list, model_name: str):
    """Encode all unique clue texts with a sentence-transformer."""
    all_clue_texts = set()
    for puzzle in puzzles:
        for info in puzzle['across_clues'].values():
            all_clue_texts.add(info['text'])
        for info in puzzle['down_clues'].values():
            all_clue_texts.add(info['text'])

    all_clue_texts = sorted(all_clue_texts)
    print(f"Encoding {len(all_clue_texts)} unique clue texts with {model_name}...")

    model = SentenceTransformer(model_name)
    embeddings = model.encode(all_clue_texts, show_progress_bar=True, batch_size=256)

    clue_to_embedding = {
        text: emb for text, emb in zip(all_clue_texts, embeddings)
    }
    clue_dim = embeddings.shape[1]
    return clue_to_embedding, clue_dim


# ---------------------------------------------------------------------------
# Grid encoding helpers
# ---------------------------------------------------------------------------

def letter_to_token(ch: str) -> int:
    """Convert an uppercase letter A-Z to token ID 3-28."""
    return ord(ch) - ord('A') + 3


def build_cell_clue_embeddings(
    puzzle: dict,
    clue_to_embedding: dict,
    clue_dim: int,
    max_rows: int,
    max_cols: int,
) -> np.ndarray:
    """
    Build per-cell clue embedding matrix.
    For each white cell: clue_emb = across_clue_emb + down_clue_emb.
    Returns shape [max_rows * max_cols, clue_dim].
    """
    emb = np.zeros((max_rows * max_cols, clue_dim), dtype=np.float32)

    for r in range(puzzle['rows']):
        for c in range(puzzle['cols']):
            if puzzle['grid'][r][c] is None:
                continue  # black cell → zeros

            flat_idx = r * max_cols + c

            # Add across clue embedding
            if (r, c) in puzzle['cell_to_across']:
                clue_id = puzzle['cell_to_across'][(r, c)]
                clue_text = puzzle['across_clues'][clue_id]['text']
                emb[flat_idx] += clue_to_embedding[clue_text]

            # Add down clue embedding
            if (r, c) in puzzle['cell_to_down']:
                clue_id = puzzle['cell_to_down'][(r, c)]
                clue_text = puzzle['down_clues'][clue_id]['text']
                emb[flat_idx] += clue_to_embedding[clue_text]

    return emb


def build_grid_arrays(
    puzzle: dict,
    max_rows: int,
    max_cols: int,
    rng: Optional[np.random.Generator] = None,
    reveal_frac: float = 0.0,
):
    """
    Build input and label token arrays for one crossword example.

    Args:
        puzzle: parsed puzzle dict
        max_rows, max_cols: padded grid dimensions
        rng: numpy RNG for random reveals
        reveal_frac: fraction of white cells to reveal in the input

    Returns:
        input_arr: [max_cells] uint8
        label_arr: [max_cells] uint8
    """
    seq_len = max_rows * max_cols
    input_arr = np.zeros(seq_len, dtype=np.uint8)   # PAD by default
    label_arr = np.zeros(seq_len, dtype=np.uint8)    # ignore (0) by default

    white_cells = []

    for r in range(puzzle['rows']):
        for c in range(puzzle['cols']):
            flat_idx = r * max_cols + c
            ch = puzzle['grid'][r][c]

            if ch is None:
                # Black cell
                input_arr[flat_idx] = 1   # BLACK token
                label_arr[flat_idx] = 0   # ignore in loss
            else:
                # White cell
                letter_tok = letter_to_token(ch)
                input_arr[flat_idx] = 2   # BLANK
                label_arr[flat_idx] = letter_tok
                white_cells.append(flat_idx)

    # Random partial reveal augmentation
    if reveal_frac > 0 and len(white_cells) > 0 and rng is not None:
        num_reveal = max(1, int(reveal_frac * len(white_cells)))
        reveal_indices = rng.choice(white_cells, size=num_reveal, replace=False)
        for idx in reveal_indices:
            input_arr[idx] = label_arr[idx]  # copy letter token into input
            label_arr[idx] = 0               # mask from loss — no free credit

    return input_arr, label_arr


# ---------------------------------------------------------------------------
# Dataset conversion
# ---------------------------------------------------------------------------

def convert_puzzles(
    puzzles: list,
    config: CrosswordDataConfig,
    clue_to_embedding: dict,
    clue_dim: int,
    max_rows: int,
    max_cols: int,
    set_name: str,
    is_train: bool,
):
    """Convert a list of parsed puzzles to TRM dataset format and save."""
    num_augments = config.num_aug if is_train else 0
    rng = np.random.default_rng(config.seed + (0 if is_train else 99))

    results = {
        "inputs": [],
        "labels": [],
        "puzzle_identifiers": [],
        "puzzle_indices": [0],
        "group_indices": [0],
    }
    example_id = 0
    puzzle_id = 0
    num_groups = 0

    # Pre-create save directory so we can stream clue embeddings to disk
    seq_len = max_rows * max_cols
    save_dir = os.path.join(config.output_dir, set_name)
    os.makedirs(save_dir, exist_ok=True)

    # Open a .npy file for streaming clue embeddings (one row per group)
    clue_emb_path = os.path.join(save_dir, "all__clue_embeddings.npy")
    clue_emb_shape = (len(puzzles), seq_len, clue_dim)
    clue_emb_fp = open(clue_emb_path, "wb")
    # Write npy header with known shape
    np.lib.format.write_array_header_2_0(
        clue_emb_fp,
        {"descr": np.dtype(np.float16).str, "fortran_order": False, "shape": clue_emb_shape},
    )

    for puzzle in tqdm(puzzles, desc=f"Building {set_name}"):
        cell_clue_emb = build_cell_clue_embeddings(
            puzzle, clue_to_embedding, clue_dim, max_rows, max_cols
        )
        # Stream clue embedding directly to disk
        cell_clue_emb.astype(np.float16).tofile(clue_emb_fp)
        num_groups += 1

        for aug_idx in range(1 + num_augments):
            if aug_idx == 0:
                # Base version: no reveals
                inp, lbl = build_grid_arrays(puzzle, max_rows, max_cols)
            else:
                # Augmented: random partial reveal
                reveal_frac = rng.uniform(config.reveal_min, config.reveal_max)
                inp, lbl = build_grid_arrays(
                    puzzle, max_rows, max_cols, rng=rng, reveal_frac=reveal_frac
                )

            results["inputs"].append(inp)
            results["labels"].append(lbl)
            results["puzzle_identifiers"].append(0)

            example_id += 1
            puzzle_id += 1
            results["puzzle_indices"].append(example_id)

        # Close group (one group = one original puzzle + its augments)
        results["group_indices"].append(puzzle_id)

    clue_emb_fp.close()
    clue_emb_bytes = os.path.getsize(clue_emb_path)

    # Convert to numpy --------------------------------------------------
    final = {
        "inputs": np.array(results["inputs"], dtype=np.uint8),
        "labels": np.array(results["labels"], dtype=np.uint8),
        "puzzle_indices": np.array(results["puzzle_indices"], dtype=np.int32),
        "group_indices": np.array(results["group_indices"], dtype=np.int32),
        "puzzle_identifiers": np.array(results["puzzle_identifiers"], dtype=np.int32),
    }

    # Metadata
    metadata = PuzzleDatasetMetadata(
        seq_len=seq_len,
        vocab_size=29,  # PAD + BLACK + BLANK + A-Z
        pad_id=0,
        ignore_label_id=0,
        blank_identifier_id=0,
        num_puzzle_identifiers=1,
        total_groups=len(final["group_indices"]) - 1,
        mean_puzzle_examples=1,
        total_puzzles=len(final["group_indices"]) - 1,
        sets=["all"],
    )

    # Save ---------------------------------------------------------------
    with open(os.path.join(save_dir, "dataset.json"), "w") as f:
        json.dump(metadata.model_dump(), f)

    for k, v in final.items():
        np.save(os.path.join(save_dir, f"all__{k}.npy"), v)

    # clue embeddings already saved via streaming above

    # Save identifiers mapping (for consistency with other datasets)
    with open(os.path.join(config.output_dir, "identifiers.json"), "w") as f:
        json.dump(["<blank>"], f)

    print(
        f"Saved {set_name}: {len(results['inputs'])} examples "
        f"from {len(puzzles)} puzzles  →  {save_dir}"
        f"  (clue_embeddings: {clue_emb_shape}, {clue_emb_bytes / 1e6:.1f} MB)"
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

@cli.command(singleton=True)
def preprocess_data(config: CrosswordDataConfig):
    rng = np.random.default_rng(config.seed)

    use_separate_test = len(config.test_input_dirs) > 0

    # 1. Collect .puz files for train
    puz_files = []
    for input_dir in config.input_dirs:
        puz_files.extend(
            sorted(glob.glob(os.path.join(input_dir, "**/*.puz"), recursive=True))
        )
    print(f"Found {len(puz_files)} train .puz files")

    # Collect .puz files for test (if separate dirs specified)
    test_puz_files = []
    if use_separate_test:
        for input_dir in config.test_input_dirs:
            test_puz_files.extend(
                sorted(glob.glob(os.path.join(input_dir, "**/*.puz"), recursive=True))
            )
        print(f"Found {len(test_puz_files)} test .puz files")

    # 2. Parse all puzzles
    # Tag files with source for later splitting
    tagged_files = [(f, 'train') for f in puz_files] + [(f, 'test') for f in test_puz_files]
    all_puzzles = []
    max_rows = 0
    max_cols = 0
    errors = 0

    for f, source in tqdm(tagged_files, desc="Parsing .puz files"):
        try:
            puzzle = parse_puz_file(f)
            puzzle['_source'] = source
            all_puzzles.append(puzzle)
            max_rows = max(max_rows, puzzle['rows'])
            max_cols = max(max_cols, puzzle['cols'])
        except Exception as e:
            print(f"Error parsing {f}: {e}")
            errors += 1

    print(
        f"Parsed {len(all_puzzles)} puzzles "
        f"(max grid: {max_rows}×{max_cols}), {errors} errors"
    )

    # 3. Encode clues with sentence-transformer (over ALL puzzles for consistency)
    clue_to_embedding, clue_dim = encode_clues(all_puzzles, config.clue_model)
    print(f"Clue embedding dim: {clue_dim}")

    # Save crossword-specific metadata
    os.makedirs(config.output_dir, exist_ok=True)
    with open(os.path.join(config.output_dir, "crossword_meta.json"), "w") as f:
        json.dump({
            "max_rows": max_rows,
            "max_cols": max_cols,
            "clue_dim": clue_dim,
            "num_puzzles": len(all_puzzles),
            "clue_model": config.clue_model,
        }, f)

    # 4. Train / test split
    if use_separate_test:
        # test_input_dirs puzzles are the "test pool" — split them into
        # train/test so the model sees some of the test-distribution during
        # training.  All input_dirs-only puzzles go to train.
        test_pool = [p for p in all_puzzles if p['_source'] == 'test']
        other_puzzles = [p for p in all_puzzles if p['_source'] == 'train']

        # Deduplicate: remove from other_puzzles any that also appear in test_pool
        test_pool_filenames = {p['filename'] for p in test_pool}
        pre_dedup = len(other_puzzles)
        other_puzzles = [p for p in other_puzzles if p['filename'] not in test_pool_filenames]
        print(f"Removed {pre_dedup - len(other_puzzles)} duplicates between train and test sources")

        # Split the test pool
        rng.shuffle(test_pool)
        test_count = max(1, int(len(test_pool) * config.test_ratio))
        test_puzzles = test_pool[:test_count]
        test_pool_train = test_pool[test_count:]

        # Train = non-test-pool puzzles + train portion of test pool
        train_puzzles = other_puzzles + test_pool_train
        rng.shuffle(train_puzzles)
        print(f"Test pool: {len(test_pool)} → {len(test_pool_train)} train + {len(test_puzzles)} test")
    else:
        rng.shuffle(all_puzzles)
        test_count = max(1, int(len(all_puzzles) * config.test_ratio))
        test_puzzles = all_puzzles[:test_count]
        train_puzzles = all_puzzles[test_count:]
    print(f"Train: {len(train_puzzles)}, Test: {len(test_puzzles)}")

    # 5. Build & save datasets
    convert_puzzles(
        train_puzzles, config, clue_to_embedding, clue_dim,
        max_rows, max_cols, "train", is_train=True,
    )
    convert_puzzles(
        test_puzzles, config, clue_to_embedding, clue_dim,
        max_rows, max_cols, "test", is_train=False,
    )

    print("Done!")


if __name__ == "__main__":
    cli()
