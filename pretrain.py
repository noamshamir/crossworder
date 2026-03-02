from typing import Optional, Any, Sequence, List
from dataclasses import dataclass
import os
import math
import yaml
import shutil
import copy
import time

import torch
import torch.distributed as dist
from torch import nn
from torch.utils.data import DataLoader

import tqdm
import wandb
import coolname
import hydra
import pydantic
from omegaconf import DictConfig
try:
    from adam_atan2 import AdamATan2
except Exception as e:
    AdamATan2 = None
    print(f"WARNING: adam_atan2 not available ({e}); falling back to torch.optim.AdamW")

from puzzle_dataset import PuzzleDataset, PuzzleDatasetConfig, PuzzleDatasetMetadata
from utils.functions import load_model_class, get_model_source_path
from models.sparse_embedding import CastedSparseEmbeddingSignSGD_Distributed
from models.ema import EMAHelper

def make_adam_optimizer(params, *, lr: float, weight_decay: float, betas):
    """Create AdamATan2 if available, otherwise fall back to torch.optim.AdamW.

    This makes Colab runs robust when the adam_atan2 backend extension is not built.
    """
    if AdamATan2 is None:
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay, betas=betas)
    return AdamATan2(params, lr=lr, weight_decay=weight_decay, betas=betas)


class LossConfig(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra='allow')
    name: str


class ArchConfig(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra='allow')
    name: str
    loss: LossConfig


class EvaluatorConfig(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra="allow")
    name: str


class PretrainConfig(pydantic.BaseModel):
    # Config
    arch: ArchConfig
    # Data
    data_paths: List[str]
    data_paths_test: List[str] = []
    # Evaluators
    evaluators: List[EvaluatorConfig] = []

    # Hyperparams
    global_batch_size: int
    epochs: int

    lr: float
    lr_min_ratio: float
    lr_warmup_steps: int

    weight_decay: float
    beta1: float
    beta2: float

    # Puzzle embedding
    puzzle_emb_lr: float
    puzzle_emb_weight_decay: float

    # Names
    project_name: Optional[str] = None
    run_name: Optional[str] = None
    load_checkpoint: Optional[str] = None
    checkpoint_path: Optional[str] = None

    # Extras
    seed: int = 0
    checkpoint_every_eval: bool = False
    eval_interval: Optional[int] = None
    min_eval_interval: Optional[int] = 0 # when to start eval
    eval_save_outputs: List[str] = []

    ema: bool = False # use Exponential-Moving-Average
    ema_rate: float = 0.999 # EMA-rate
    freeze_weights: bool = False # If True, freeze weights and only learn the embeddings

@dataclass
class TrainState:
    model: nn.Module
    optimizers: Sequence[torch.optim.Optimizer]
    optimizer_lrs: Sequence[float]
    carry: Any

    step: int
    total_steps: int


def create_dataloader(config: PretrainConfig, split: str, rank: int, world_size: int, **kwargs):
    dataset = PuzzleDataset(PuzzleDatasetConfig(
        seed=config.seed,
        dataset_paths=config.data_paths_test if len(config.data_paths_test)>0 and split=="test" else config.data_paths,
        rank=rank,
        num_replicas=world_size,
        **kwargs
    ), split=split)
    # Try to keep the GPU fed; Colab/remote runtimes often benefit from >1 worker.
    # Cap to avoid runaway RAM usage.
    cpu_count = os.cpu_count() or 4
    num_workers = max(2, min(8, cpu_count // 2))

    dataloader = DataLoader(
        dataset,
        batch_size=None,
        num_workers=1,
        pin_memory=True,
        persistent_workers=1
    )
    return dataloader, dataset.metadata


def create_model(config: PretrainConfig, train_metadata: PuzzleDatasetMetadata, rank: int, world_size: int):
    model_cfg = dict(
        **config.arch.__pydantic_extra__,  # type: ignore
        batch_size=config.global_batch_size // world_size,
        vocab_size=train_metadata.vocab_size,
        seq_len=train_metadata.seq_len,
        num_puzzle_identifiers=train_metadata.num_puzzle_identifiers,
        causal=False  # Non-autoregressive
    )

    # Instantiate model with loss head
    model_cls = load_model_class(config.arch.name)
    loss_head_cls = load_model_class(config.arch.loss.name)

    with torch.device("cuda"):
        model: nn.Module = model_cls(model_cfg)
        print(model)
        model = loss_head_cls(model, **config.arch.loss.__pydantic_extra__)  # type: ignore
        if "DISABLE_COMPILE" not in os.environ:
            model = torch.compile(model)  # type: ignore

        # Load checkpoint
        if rank == 0:
            load_checkpoint(model, config)

        # Broadcast parameters from rank 0
        if world_size > 1:
            with torch.no_grad():
                for param in list(model.parameters()) + list(model.buffers()):
                    dist.broadcast(param, src=0)

    # Optimizers and lr
    if config.arch.puzzle_emb_ndim == 0:
        optimizers = [
            make_adam_optimizer(
                model.parameters(),
                lr=0,  # Needs to be set by scheduler
                weight_decay=config.weight_decay,
                betas=(config.beta1, config.beta2)
            )
        ]
        optimizer_lrs = [
            config.lr
        ]
    elif config.freeze_weights:
        optimizers = [
            CastedSparseEmbeddingSignSGD_Distributed(
                model.model.puzzle_emb.buffers(),  # type: ignore
                lr=0,  # Needs to be set by scheduler
                weight_decay=config.puzzle_emb_weight_decay,
                world_size=world_size
            )
        ]
        optimizer_lrs = [
            config.puzzle_emb_lr
        ]
    else:
        optimizers = [
            CastedSparseEmbeddingSignSGD_Distributed(
                model.model.puzzle_emb.buffers(),  # type: ignore
                lr=0,  # Needs to be set by scheduler
                weight_decay=config.puzzle_emb_weight_decay,
                world_size=world_size
            ),
            make_adam_optimizer(
                model.parameters(),
                lr=0,  # Needs to be set by scheduler
                weight_decay=config.weight_decay,
                betas=(config.beta1, config.beta2)
            )
        ]
        optimizer_lrs = [
            config.puzzle_emb_lr,
            config.lr
        ]

    return model, optimizers, optimizer_lrs

def mix_weights_direct(device, alpha, net, nets):
    sd = []
    for i in range(len(nets)):
        sd += [nets[i].state_dict()]
    sd_alpha = {}
    for k in sd[0].keys():
        comb_net = alpha[0]*sd[0][k].to(device)
        for i in range(1,len(nets)):
            comb_net += alpha[i]*sd[i][k].to(device)
        sd_alpha[k] =  comb_net
    net.load_state_dict(sd_alpha)
    return net

def cosine_schedule_with_warmup_lr_lambda(
    current_step: int, *, base_lr: float, num_warmup_steps: int, num_training_steps: int, min_ratio: float = 0.0, num_cycles: float = 0.5
):
    if current_step < num_warmup_steps:
        return base_lr * float(current_step) / float(max(1, num_warmup_steps))

    progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
    return base_lr * (min_ratio + max(0.0, (1 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress))))


def init_train_state(config: PretrainConfig, train_metadata: PuzzleDatasetMetadata, rank: int, world_size: int):
    # Estimated total training steps
    total_steps = int(config.epochs * train_metadata.total_groups * train_metadata.mean_puzzle_examples / config.global_batch_size)

    # Model
    model, optimizers, optimizer_lrs = create_model(config, train_metadata, rank=rank, world_size=world_size)

    return TrainState(
        step=0,
        total_steps=total_steps,

        model=model,
        optimizers=optimizers,
        optimizer_lrs=optimizer_lrs,
        carry=None
    )


def save_train_state(config: PretrainConfig, train_state: TrainState):
    # FIXME: Only saved model.
    if config.checkpoint_path is None:
        return

    os.makedirs(config.checkpoint_path, exist_ok=True)
    torch.save(train_state.model.state_dict(), os.path.join(config.checkpoint_path, f"step_{train_state.step}"))

def load_checkpoint(model: nn.Module, config: PretrainConfig):
    if config.load_checkpoint is None:
        return

    print(f"Loading checkpoint {config.load_checkpoint}")
    state_dict = torch.load(config.load_checkpoint, map_location="cuda")

    # Check whether this model actually has puzzle_emb.weights
    has_puzzle_emb = (
        hasattr(model, "model")
        and hasattr(model.model, "puzzle_emb")  # type: ignore
        and hasattr(model.model.puzzle_emb, "weights")  # type: ignore
    )

    if not has_puzzle_emb:
        # Drop any puzzle_emb keys from checkpoint, then load the rest
        drop_keys = [k for k in list(state_dict.keys()) if "puzzle_emb" in k]
        if drop_keys:
            print(f"Checkpoint contains puzzle_emb but model doesn't; dropping {len(drop_keys)} keys.")
            for k in drop_keys:
                state_dict.pop(k, None)

        model.load_state_dict(state_dict, strict=False, assign=True)
        return

    # Resize/reset puzzle emb if present and shape differs
    expected_shape: torch.Size = model.model.puzzle_emb.weights.shape  # type: ignore

    # Your old key was hardcoded; make it robust to different prefixes
    candidate_keys = [k for k in state_dict.keys() if k.endswith("puzzle_emb.weights")]
    for k in candidate_keys:
        puzzle_emb = state_dict[k]
        if hasattr(puzzle_emb, "shape") and puzzle_emb.shape != expected_shape:
            print(f"Resetting puzzle embedding for {k}. Found {puzzle_emb.shape}, Expected {expected_shape}")
            state_dict[k] = (
                torch.mean(puzzle_emb, dim=0, keepdim=True)
                .expand(expected_shape)
                .contiguous()
            )

    model.load_state_dict(state_dict, strict=False, assign=True)

def compute_lr(base_lr: float, config: PretrainConfig, train_state: TrainState):
    return cosine_schedule_with_warmup_lr_lambda(
        current_step=train_state.step,
        base_lr=base_lr,
        num_warmup_steps=round(config.lr_warmup_steps),
        num_training_steps=train_state.total_steps,
        min_ratio=config.lr_min_ratio
    )



def create_evaluators(config: PretrainConfig, eval_metadata: PuzzleDatasetMetadata) -> List[Any]:
    data_paths =config.data_paths_test if len(config.data_paths_test)>0 else config.data_paths
    # Initialize evaluators
    evaluators = []
    for cfg in config.evaluators:
        for data_path in data_paths:
            cls = load_model_class(cfg.name, "evaluators.")(
                data_path=data_path, eval_metadata=eval_metadata, **cfg.__pydantic_extra__
            )  # type: ignore
            evaluators.append(cls)

    return evaluators

def train_batch(config: PretrainConfig, train_state: TrainState, batch: Any, global_batch_size: int, rank: int, world_size: int):
    train_state.step += 1
    if train_state.step > train_state.total_steps:  # At most train_total_steps
        return

    # To device
    batch = {k: v.cuda() for k, v in batch.items()}

    # Init carry if it is None
    if train_state.carry is None:
        with torch.device("cuda"):
            train_state.carry = train_state.model.initial_carry(batch)  # type: ignore

    # Forward (fp16 autocast for speed; keep backward in fp32 as usual)
    #with torch.autocast(device_type="cuda", dtype=torch.bfloat16):  # H100
    with torch.autocast(device_type="cuda", dtype=torch.float16):    # T4
        train_state.carry, loss, metrics, _, _ = train_state.model(
            carry=train_state.carry, batch=batch, return_keys=[]
        )

    ((1 / global_batch_size) * loss).backward()

    # Allreduce
    if world_size > 1:
        for param in train_state.model.parameters():
            if param.grad is not None:
                dist.all_reduce(param.grad)
            
    # Apply optimizer
    lr_this_step = None    
    for optim, base_lr in zip(train_state.optimizers, train_state.optimizer_lrs):
        lr_this_step = compute_lr(base_lr, config, train_state)

        for param_group in optim.param_groups:
            param_group['lr'] = lr_this_step
            
        optim.step()
        optim.zero_grad()

    # Reduce metrics
    if len(metrics):
        assert not any(v.requires_grad for v in metrics.values())

        metric_keys = list(sorted(metrics.keys()))  # Sort keys to guarantee all processes use the same order.
        # Reduce and reconstruct
        metric_values = torch.stack([metrics[k] for k in metric_keys])
        if world_size > 1:
            dist.reduce(metric_values, dst=0)

        if rank == 0:
            metric_values = metric_values.cpu().numpy()
            reduced_metrics = {k: metric_values[i] for i, k in enumerate(metric_keys)}
            
            # Postprocess
            count = max(reduced_metrics["count"], 1)  # Avoid NaNs
            reduced_metrics = {f"train/{k}": v / (global_batch_size if k.endswith("loss") else count) for k, v in reduced_metrics.items()}

            reduced_metrics["train/lr"] = lr_this_step
            return reduced_metrics

def evaluate(
    config: PretrainConfig,
    train_state: TrainState,
    eval_loader: torch.utils.data.DataLoader,
    eval_metadata: PuzzleDatasetMetadata,
    evaluators: List[Any],
    rank: int,
    world_size: int,
    cpu_group: Optional[dist.ProcessGroup],
):
    reduced_metrics = None

    with torch.inference_mode():
        return_keys = set(config.eval_save_outputs)
        for evaluator in evaluators:
            evaluator.begin_eval()
            return_keys.update(evaluator.required_outputs)

        # Run evaluation
        set_ids = {k: idx for idx, k in enumerate(eval_metadata.sets)}

        save_preds = {}

        metric_keys = []
        metric_values = None

        carry = None
        processed_batches = 0
        
        for set_name, batch, global_batch_size in eval_loader:
            processed_batches += 1
            if rank == 0:
                print(f"Processing batch {processed_batches}: {set_name}")
            
            # To device
            batch = {k: v.cuda() for k, v in batch.items()}
            with torch.device("cuda"):
                carry = train_state.model.initial_carry(batch)  # type: ignore

            # Forward
            inference_steps = 0
            while True:
                #with torch.autocast(device_type="cuda", dtype=torch.bfloat16):  # H100
                with torch.autocast(device_type="cuda", dtype=torch.float16):    # T4
                    carry, loss, metrics, preds, all_finish = train_state.model(
                        carry=carry, batch=batch, return_keys=return_keys
                    )
                inference_steps += 1

                if all_finish:
                    break

            if rank == 0:
                print(f"  Completed inference in {inference_steps} steps")

            for collection in (batch, preds):
                for k, v in collection.items():
                    if k in config.eval_save_outputs:
                        save_preds.setdefault(k, [])
                        save_preds[k].append(v.cpu())  # Move to CPU for saving GPU memory

            for evaluator in evaluators:
                evaluator.update_batch(batch, preds)

            del carry, loss, preds, batch, all_finish

            # Aggregate metrics
            set_id = set_ids[set_name]

            if metric_values is None:
                metric_keys = list(
                    sorted(metrics.keys())
                )  # Sort keys to guarantee all processes use the same order.
                metric_values = torch.zeros(
                    (len(set_ids), len(metrics.values())), dtype=torch.float32, device="cuda"
                )

            metric_values[set_id] += torch.stack([metrics[k] for k in metric_keys])

            del metrics

        # concatenate save preds
        save_preds = {k: torch.cat(v, dim=0) for k, v in save_preds.items()}

        # Save preds
        if config.checkpoint_path is not None and len(save_preds):
            # Each rank save predictions independently
            os.makedirs(os.path.dirname(config.checkpoint_path), exist_ok=True)
            torch.save(
                save_preds, os.path.join(config.checkpoint_path, f"step_{train_state.step}_all_preds.{rank}")
            )

        del save_preds

        # Reduce to rank 0
        if metric_values is not None:
            if world_size > 1:
                dist.reduce(metric_values, dst=0)

            if rank == 0:
                reduced_metrics = metric_values.cpu().numpy()
                reduced_metrics = {
                    set_name: {
                        metric_name: reduced_metrics[set_id, metric_id]
                        for metric_id, metric_name in enumerate(metric_keys)
                    }
                    for set_id, set_name in enumerate(set_ids)
                }

                # Postprocess
                for set_name, m in reduced_metrics.items():
                    count = m.pop("count")
                    reduced_metrics[set_name] = {k: v / count for k, v in m.items()}

        # Run evaluators
        if rank == 0:
            print(f"\nRunning {len(evaluators)} evaluator(s)...")
            
        for i, evaluator in enumerate(evaluators):
            if rank == 0:
                print(f"Running evaluator {i+1}/{len(evaluators)}: {evaluator.__class__.__name__}")
                
            # Path for saving
            evaluator_save_path = None
            if config.checkpoint_path is not None:
                evaluator_save_path = os.path.join(
                    config.checkpoint_path,
                    f"evaluator_{evaluator.__class__.__name__}_step_{train_state.step}",
                )
                os.makedirs(evaluator_save_path, exist_ok=True)

            # Run and log
            metrics = evaluator.result(evaluator_save_path, rank=rank, world_size=world_size, group=cpu_group)
            if rank == 0 and metrics is not None:
                if reduced_metrics is None:
                    reduced_metrics = {}

                reduced_metrics.update(metrics)
                print(f"  Completed {evaluator.__class__.__name__}")
                
        if rank == 0:
            print("All evaluators completed!")

    return reduced_metrics

def save_code_and_config(config: PretrainConfig):
    if config.checkpoint_path is None or wandb.run is None:
        return

    os.makedirs(config.checkpoint_path, exist_ok=True)

    # Copy code
    code_list = [
        get_model_source_path(config.arch.name),
        get_model_source_path(config.arch.loss.name)
    ]
    for code_file in code_list:
        if code_file is not None:
            code_name = os.path.basename(code_file)

            shutil.copy(code_file, os.path.join(config.checkpoint_path, code_name))

    # Dump config as yaml
    config_file = os.path.join(config.checkpoint_path, "all_config.yaml")
    with open(config_file, "wt") as f:
        yaml.dump(config.model_dump(), f)

    # Log code
    wandb.run.log_code(config.checkpoint_path)


def load_synced_config(hydra_config: DictConfig, rank: int, world_size: int) -> PretrainConfig:
    objects = [None]
    if rank == 0:
        config = PretrainConfig(**hydra_config)  # type: ignore

        # Naming
        if config.project_name is None:
            config.project_name = f"{os.path.basename(config.data_paths[0]).capitalize()}-ACT-torch"
        if config.run_name is None:
            config.run_name = f"{config.arch.name.split('@')[-1]} {coolname.generate_slug(2)}"
        if config.checkpoint_path is None:
            config.checkpoint_path = os.path.join("checkpoints", config.project_name, config.run_name)

        objects = [config]

    if world_size > 1:
        dist.broadcast_object_list(objects, src=0)

    return objects[0]  # type: ignore


def print_final_summary(
    config: PretrainConfig,
    train_state: TrainState,
    train_metadata: PuzzleDatasetMetadata,
    wall_time: float,
    train_history: list,
    eval_history: list,
):
    """Print and save a comprehensive training summary."""
    num_params = sum(p.numel() for p in train_state.model.parameters())
    num_trainable = sum(p.numel() for p in train_state.model.parameters() if p.requires_grad)

    hours = wall_time / 3600
    steps_per_sec = train_state.step / wall_time if wall_time > 0 else 0

    # Best eval metrics
    best_eval = {}
    if eval_history:
        # Collect all metric keys across eval history
        all_keys = set()
        for entry in eval_history:
            for set_name, metrics in entry["metrics"].items():
                for k in metrics:
                    all_keys.add(f"{set_name}/{k}")

        for key in sorted(all_keys):
            set_name, metric_name = key.split("/", 1)
            vals = []
            for entry in eval_history:
                if set_name in entry["metrics"] and metric_name in entry["metrics"][set_name]:
                    vals.append((entry["step"], entry["metrics"][set_name][metric_name]))
            if vals:
                # For loss: best = min; for accuracy-like: best = max
                is_loss = "loss" in metric_name.lower()
                best_step, best_val = min(vals, key=lambda x: x[1]) if is_loss else max(vals, key=lambda x: x[1])
                best_eval[key] = {"value": best_val, "step": best_step}

    # Last train metrics
    last_train = train_history[-1] if train_history else {}

    sep = "=" * 72
    summary_lines = [
        "",
        sep,
        "  TRAINING COMPLETE — FINAL SUMMARY",
        sep,
        "",
        f"  Run name:        {config.run_name}",
        f"  Project:         {config.project_name}",
        f"  Checkpoint:      {config.checkpoint_path}",
        "",
        "  --- Model ---",
        f"  Architecture:    {config.arch.name}",
        f"  Parameters:      {num_params:,} ({num_params/1e6:.2f}M)",
        f"  Trainable:       {num_trainable:,} ({num_trainable/1e6:.2f}M)",
        f"  EMA:             {config.ema} (rate={config.ema_rate})",
        "",
        "  --- Data ---",
        f"  Train paths:     {config.data_paths}",
        f"  Test paths:      {config.data_paths_test}",
        f"  Train groups:    {train_metadata.total_groups:,}",
        f"  Seq length:      {train_metadata.seq_len}",
        f"  Vocab size:      {train_metadata.vocab_size}",
        "",
        "  --- Training ---",
        f"  Epochs:          {config.epochs:,}",
        f"  Batch size:      {config.global_batch_size}",
        f"  Total steps:     {train_state.step:,} / {train_state.total_steps:,}",
        f"  Learning rate:   {config.lr} (min_ratio={config.lr_min_ratio})",
        f"  Warmup steps:    {config.lr_warmup_steps}",
        f"  Weight decay:    {config.weight_decay}",
        f"  Wall time:       {hours:.2f} hours ({wall_time:.0f}s)",
        f"  Steps/sec:       {steps_per_sec:.2f}",
        f"  Seed:            {config.seed}",
    ]

    if last_train:
        summary_lines += [
            "",
            "  --- Last Train Metrics ---",
        ]
        for k, v in sorted(last_train.items()):
            summary_lines.append(f"    {k}: {v:.6f}" if isinstance(v, float) else f"    {k}: {v}")

    if best_eval:
        summary_lines += [
            "",
            "  --- Best Eval Metrics ---",
        ]
        for k, info in sorted(best_eval.items()):
            summary_lines.append(f"    {k}: {info['value']:.6f}  (step {info['step']})")

    if eval_history:
        last_eval = eval_history[-1]
        summary_lines += [
            "",
            f"  --- Final Eval (step {last_eval['step']}) ---",
        ]
        for set_name, metrics in sorted(last_eval["metrics"].items()):
            for k, v in sorted(metrics.items()):
                summary_lines.append(f"    {set_name}/{k}: {v:.6f}")

    summary_lines += [
        "",
        sep,
    ]

    summary_text = "\n".join(summary_lines)
    print(summary_text)

    # Save to file
    if config.checkpoint_path is not None:
        os.makedirs(config.checkpoint_path, exist_ok=True)
        summary_path = os.path.join(config.checkpoint_path, "training_summary.txt")
        with open(summary_path, "w") as f:
            f.write(summary_text + "\n")
        print(f"\n  Summary saved to: {summary_path}\n")


@hydra.main(config_path="config", config_name="cfg_pretrain", version_base=None)
def launch(hydra_config: DictConfig):
    RANK = 0
    WORLD_SIZE = 1
    CPU_PROCESS_GROUP = None

    # Initialize distributed training if in distributed environment (e.g. torchrun)
    if "LOCAL_RANK" in os.environ:
        # Initialize distributed, default device and dtype
        dist.init_process_group(backend="nccl")

        RANK = dist.get_rank()
        WORLD_SIZE = dist.get_world_size()

        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        
        # CPU GLOO process group
        CPU_PROCESS_GROUP = dist.new_group(backend="gloo")
        assert (
            dist.get_rank(CPU_PROCESS_GROUP) == RANK and dist.get_world_size(CPU_PROCESS_GROUP) == WORLD_SIZE
        )

    # Load sync'ed config
    config = load_synced_config(hydra_config, rank=RANK, world_size=WORLD_SIZE)

    # Seed RNGs to ensure consistency
    torch.random.manual_seed(config.seed + RANK)
    # Performance knobs (safe on Ampere+; especially beneficial on H100)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

    # Dataset
    train_epochs_per_iter = config.eval_interval if config.eval_interval is not None else config.epochs
    total_iters = config.epochs // train_epochs_per_iter

    assert config.epochs % train_epochs_per_iter == 0, "Eval interval must be a divisor of total epochs."

    train_loader, train_metadata = create_dataloader(config, "train", test_set_mode=False, epochs_per_iter=train_epochs_per_iter, global_batch_size=config.global_batch_size, rank=RANK, world_size=WORLD_SIZE)
    try:
        eval_loader,  eval_metadata  = create_dataloader(config, "test", test_set_mode=True, epochs_per_iter=1, global_batch_size=config.global_batch_size, rank=RANK, world_size=WORLD_SIZE)
    except:
        print("NO EVAL DATA FOUND")
        eval_loader = eval_metadata = None

    try:
        evaluators = create_evaluators(config, eval_metadata)
    except:
        print("No evaluator found")
        evaluators = []

    # Train state
    train_state = init_train_state(config, train_metadata, rank=RANK, world_size=WORLD_SIZE)

    # Progress bar and logger
    progress_bar = None
    ema_helper = None
    if RANK == 0:
        progress_bar = tqdm.tqdm(total=train_state.total_steps)
        wandb.init(project=config.project_name, name=config.run_name, config=config.model_dump(), settings=wandb.Settings(_disable_stats=True))  # type: ignore
        wandb.log({"num_params": sum(x.numel() for x in train_state.model.parameters())}, step=0)
        save_code_and_config(config)
    if config.ema:
        print('Setup EMA')
        ema_helper = EMAHelper(mu=config.ema_rate)
        ema_helper.register(train_state.model)

    # Training Loop
    train_history = []
    eval_history = []
    start_time = time.time()

    for _iter_id in range(total_iters):
        print (f"[Rank {RANK}, World Size {WORLD_SIZE}]: Epoch {_iter_id * train_epochs_per_iter}")

        ############ Train Iter
        if RANK == 0:
            print("TRAIN")
        train_state.model.train()
        epoch_metrics_accum = {}
        epoch_batch_count = 0
        for set_name, batch, global_batch_size in train_loader:
            metrics = train_batch(config, train_state, batch, global_batch_size, rank=RANK, world_size=WORLD_SIZE)

            if RANK == 0 and metrics is not None:
                wandb.log(metrics, step=train_state.step)
                progress_bar.update(train_state.step - progress_bar.n)  # type: ignore
                train_history.append(metrics)
                # Accumulate for epoch summary
                epoch_batch_count += 1
                for k, v in metrics.items():
                    if isinstance(v, (int, float)):
                        epoch_metrics_accum[k] = epoch_metrics_accum.get(k, 0) + v
            if config.ema:
                ema_helper.update(train_state.model)

        # Print epoch summary
        if RANK == 0 and epoch_batch_count > 0:
            epoch_num = (_iter_id + 1) * train_epochs_per_iter
            avg_metrics = {k: v / epoch_batch_count for k, v in epoch_metrics_accum.items()}
            summary_parts = [f"Epoch {epoch_num}/{config.epochs}  step={train_state.step}"]
            for k in ["train/lm_loss", "train/accuracy", "train/exact_accuracy", "train/q_halt_loss", "train/steps"]:
                if k in avg_metrics:
                    summary_parts.append(f"{k.split('/')[-1]}={avg_metrics[k]:.4f}")
            if "train/lr" in avg_metrics:
                summary_parts.append(f"lr={avg_metrics['train/lr']:.2e}")
            print("  >>> " + "  |  ".join(summary_parts))

        if _iter_id >= config.min_eval_interval:
            ############ Evaluation
            if RANK == 0:
                print("EVALUATE")
            if config.ema:
                print("SWITCH TO EMA")
                ema_model = ema_helper.ema_copy(train_state.model)
                train_state_eval = TrainState(
                    model=ema_model,
                    optimizers=train_state.optimizers,
                    optimizer_lrs=train_state.optimizer_lrs,
                    carry=None,
                    step=train_state.step,
                    total_steps=train_state.total_steps,
                )
            else:
                train_state_eval = train_state
            train_state_eval.model.eval()
            metrics = evaluate(config, 
                train_state_eval, 
                eval_loader, 
                eval_metadata, 
                evaluators,
                rank=RANK, 
                world_size=WORLD_SIZE,
                cpu_group=CPU_PROCESS_GROUP)

            if RANK == 0 and metrics is not None:
                wandb.log(metrics, step=train_state.step)
                eval_history.append({"step": train_state.step, "metrics": metrics})
                
            ############ Checkpointing
            if RANK == 0:
                print("SAVE CHECKPOINT")
            if RANK == 0 and (config.checkpoint_every_eval or (_iter_id == total_iters - 1)):
                save_train_state(config, train_state_eval)

            if config.ema:
                del train_state_eval

    # Print final summary
    wall_time = time.time() - start_time
    if RANK == 0:
        print_final_summary(
            config=config,
            train_state=train_state,
            train_metadata=train_metadata,
            wall_time=wall_time,
            train_history=train_history,
            eval_history=eval_history,
        )

    # finalize
    if dist.is_initialized():
        dist.destroy_process_group()
    wandb.finish()


if __name__ == "__main__":
    launch()
