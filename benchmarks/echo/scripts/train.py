## add parent directory to path
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, help="Task to run: [sssp, ecc, diam, chem]",)
parser.add_argument("--seed", type=int, default=1, help="Random seed for reproducibility")
parser.add_argument("--device", type=str, default="auto", help="Device to use for training (gpu, cpu, mps, auto)")
parser.add_argument("--resume_from_checkpoint", type=str, default=None, help="Path to checkpoint to resume training")
# general gnn parameters
parser.add_argument("--conv_layer", type=str)
parser.add_argument("--num_layers", type=int, default=1, help="Number of layers in the GNN")
parser.add_argument("--hidden_dim", type=int, help="Hidden dimension of the GNN")
parser.add_argument("--lr", type=float, help="Learning rate for the optimizer")
parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay for the optimizer")
parser.add_argument("--batch_size", type=int, default=256, help="Batch size for the DataLoader")
parser.add_argument("--gnn_type", type=str)

# GHR-specific parameters
parser.add_argument("--l_steps", type=int, default=4, help="Fine message-passing steps per outer iteration")
parser.add_argument("--h_steps", type=int, default=4, help="Outer (coarse) iterations per reasoning step")
parser.add_argument("--use_swiglu", action="store_true", help="Use SwiGLU activation instead of SiLU+Linear")

# adgn, swan specific params
parser.add_argument("--epsilon", type=float, default=0.1, help="Epsilon for the ADGN model")
parser.add_argument("--gamma", type=float, default=0.1, help="Gamma for the ADGN model")
parser.add_argument("--activ_fun", type=str, default="tanh", help="Activation function for the ADGN model")
parser.add_argument("--graph_conv", type=str, default="GCNConv", help="Graph convolution layer for the ADGN model")
parser.add_argument("--bias", type=bool, help="Use bias in the ADGN model")
parser.add_argument("--train_weights", type=bool)
parser.add_argument("--weight_sharing", type=bool, help="Use weight sharing in the ADGN model")

# drew specific parameters
parser.add_argument("--khop", type=int)
parser.add_argument("--delay", type=bool)
parser.add_argument("--constant_feature", type=float, default=1.0, help="Constant feature")

# gcn2 params
parser.add_argument("--alpha", type=float, help="Alpha for the GCN2 model")

# phdgn specific parameters
parser.add_argument("--beta", type=float, help="Beta parameter for the PHDGN model")
parser.add_argument("--p_conv_mode", type=str, choices=["naive", "gcn"], help="P convolution mode for the PhDGN model")
parser.add_argument("--q_conv_mode", type=str, choices=["naive", "gcn"], help="Q convolution mode for the PhDGN model")
parser.add_argument("--doubled_dim", type=bool, choices=[True, False], help="Whether to double the dimension in the PhDGN model")
parser.add_argument("--final_state", type=str, choices=["p", "q", "pq"], help="Final state mode for the PhDGN model")
parser.add_argument("--dampening_mode", type=str, choices=["param", "param+", "MLP4ReLU", "DGNReLU", "none"], help="Dampening mode for the PhDGN model")
parser.add_argument("--external_mode", type=str, choices=["MLP4Sin", "DGNtanh", "none"], help="External mode for the PhDGN model")
parser.add_argument("--smoke-test", action="store_true", dest="smoke_test",
                    help="Run 1 train+val batch to verify the pipeline (no checkpoints, no wandb)")

from torch_geometric.loader import DataLoader
import torch
import lightning as L
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
import os
from utils import get_dataset, KHopTransform
from utils.litmodels import LitGraphNN
import re

# Import your custom transform here

try:
    from utils.ghrtransform import GHRTransform
except ImportError:
    print("Warning: GHRTransform not found. Make sure ghrtransform.py is in the utils directory.")

torch.set_float32_matmul_precision("high")
get_epoch = lambda path: int(re.findall(r"epoch=(\d+)", path)[0])

def train(seed, config):
    """Train and validate the model"""
    task = config.task

    L.seed_everything(seed) 
    batch_size = config.batch_size

    print("Current directory: ", os.getcwd())
    chem_task = False
    # Safely assign the correct pre_transform based on the model
    if config.gnn_type in ("GHR", "GHRModel"):
        print("Applying GHR Graclus Transform...")
        transform_func = GHRTransform()
    elif config.gnn_type == "DRew_GCN":
        transform_func = KHopTransform(k=config.khop)
    else:
        transform_func = None
    if task in ["charge", "energy"]:
        config.constant_feature = None
        chem_task = True

    data_train, data_val, data_test, num_feat, num_class = get_dataset(
        root="./data/",
        task=task,
        pre_transform=None, 
        constant_feature=config.constant_feature,
    )
    fraction = 1
    print(f"\n🚀 INITIATING SCOUT MODE: Using {fraction * 100}% of training data")
    original_size = len(data_train)
    subset_size = int(fraction * original_size)
    
    # 2. FIX: Only apply randperm if fraction < 1 to prevent messing up the RNG state
    if fraction < 1.0:
        indices = torch.randperm(original_size)[:subset_size]
        data_train = data_train[indices]        
        print(f"📉 Reduced training set from {original_size} down to {len(data_train)} molecules.")
    # DYNAMICALLY ATTACH YOUR TRANSFORM HERE
    if transform_func is not None:
        data_train.transform = transform_func
        data_val.transform = transform_func
        data_test.transform = transform_func
    scaling_factor = data_train.scaling_factor[task]

    # Safely catch the un-normalized chemical datasets
    if scaling_factor is None and task in ["charge", "energy"]:
        scaling_factor = 1.0

    print(f"Scaling factor for {task}: {scaling_factor}")
    num_workers = 0 if config.smoke_test else min(4, os.cpu_count() - 1)
    train_loader = DataLoader(
        data_train, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=False, persistent_workers=num_workers > 0
    )
    val_loader = DataLoader(
        data_val, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=False, persistent_workers=num_workers > 0
    )
    test_loader = DataLoader(
        data_test, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=False, persistent_workers=num_workers > 0
    )

    print("Data loaded")

    hp_conf = vars(config)

    
    model = LitGraphNN(
        input_dim=num_feat,
        output_dim=num_class,
        node_level_task=False if task == "diam" or task == "energy" else True,
        scaling_factor=scaling_factor,
        chem_task=chem_task,
        **hp_conf,
    )

    # Lightning Trainer configured to auto-detect Mac MPS / GPU / CPU
    trainer = L.Trainer(
        fast_dev_run=config.smoke_test,
        max_epochs=500,
        accelerator=config.device,
        devices=1,
        gradient_clip_val=1.0,
        callbacks=[
            EarlyStopping(monitor="val_loss", patience=150),
            ModelCheckpoint(monitor="val_loss", save_top_k=1),
        ],
    )

    trainer.fit(model, train_loader, val_loader, ckpt_path=config.resume_from_checkpoint)
    if config.smoke_test:
        print("Smoke test passed.")
        return {}
    best_epoch = get_epoch(trainer.checkpoint_callback.best_model_path) #type: ignore
    print(f"Best epoch: {best_epoch}")
    print("Best checkpoint path: ", trainer.checkpoint_callback.best_model_path) #type: ignore

    trainer.validate(model, val_loader, ckpt_path="best")
    print("Testing model")
    trainer.test(model, test_loader, ckpt_path="best")

    # Safely extract metrics using .get() to prevent crashes on missing keys (like test_acc in SSSP)
    cb_metrics = trainer.callback_metrics
    metrics = {
        "train_loss": cb_metrics.get("train_loss", torch.tensor(0.0)).item(),
        "val_loss": cb_metrics.get("val_loss", torch.tensor(0.0)).item(),
        "val_mse": cb_metrics.get("val_mse", torch.tensor(0.0)).item(),
        "val_mae": cb_metrics.get("val_mae", torch.tensor(0.0)).item(),
        "test_loss": cb_metrics.get("test_loss", torch.tensor(0.0)).item(),
        "test_mse": cb_metrics.get("test_mse", torch.tensor(0.0)).item(),
        "test_mae": cb_metrics.get("test_mae", torch.tensor(0.0)).item(),
        "test_acc": cb_metrics.get("test_acc", torch.tensor(0.0)).item(),
        "best_epoch": best_epoch,
        "best_checkpoint_path": trainer.checkpoint_callback.best_model_path, #type: ignore
    }

    return metrics

if __name__ == "__main__":
    args = parser.parse_args()
    metrics = train(
        seed=args.seed,
        config=args,
    )

    print("Metrics: ", metrics)