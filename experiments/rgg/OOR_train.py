import os
import time
import csv
import pathlib
import random
import numpy as np

import torch
import torch.nn.functional as F
import lightning as L
from torch_geometric.loader import DataLoader
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint

# Local imports
from OOR_dataset import FixedRGGDataset
from ablation_models import *

models_map = {
    "DeepGIN": DeepGIN,
    "RecursiveGIN": RecursiveGIN,
    "ReasoningDeepGIN": ReasoningDeepGIN,
    "ReasoningRecursiveGIN": ReasoningRecursiveGIN,
    "GHR": GHR,
    "GHRAntiSymmetric": GHRAntiSymmetric,
    "GHRGCN": GHRGCN,   
    "GHRGAT": GHRGAT,   
    "SimpleGPS": SimpleGPS,
    "SimpleDeepGIN": SimpleDeepGIN

}

class LitAblationModel(L.LightningModule):
    def __init__(
        self,
        gnn_type: str,
        input_dim: int,
        hidden_dim: int = 32,
        num_layers: int = 20,
        reasoning_steps: int = 4, 
        use_swiglu: bool = False,
        gamma: float = 0.8,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        enable_timing: bool = True,
        timing_csv_base_path: str = "./range_test/training_timings",
    ):
        super().__init__()
        self.save_hyperparameters()
        
        self.gnn_type = gnn_type
        self.lr = lr
        self.weight_decay = weight_decay
        self.reasoning_steps = reasoning_steps
        self.gamma = gamma
        self.enable_timing = enable_timing
        self.use_swiglu = use_swiglu
        self._epoch_start_time = None
        self.timing_csv_file = None

        if "GHR" in gnn_type:
            self.model = models_map[gnn_type](
                input_dim=input_dim, 
                hidden_dim=hidden_dim, 
                L_steps=2, 
                H_steps=2,
                use_swiglu=self.use_swiglu
            )
        elif "Recursive" in gnn_type:
            self.model = models_map[gnn_type](
                input_dim=input_dim, hidden_dim=hidden_dim, num_iterations=num_layers, use_swiglu=self.use_swiglu
            )
        else:
            self.model = models_map[gnn_type](
                input_dim=input_dim, hidden_dim=hidden_dim, num_layers=num_layers, use_swiglu=self.use_swiglu
            )

        if self.enable_timing:
            self.timing_csv_base_path = pathlib.Path(timing_csv_base_path)
            self.timing_csv_base_path.mkdir(parents=True, exist_ok=True)
            
            suffix = "_SwiGLU" if self.use_swiglu else ""
            filename = f"{self.gnn_type}{suffix}_timing.csv"
            self.timing_csv_file = self.timing_csv_base_path / filename

            if not self.timing_csv_file.exists():
                with open(self.timing_csv_file, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(["epoch", "training_time_seconds"])

    def on_train_epoch_start(self):
        if self.enable_timing:
            self._epoch_start_time = time.time()

    def on_train_epoch_end(self):
        if self.enable_timing and self._epoch_start_time is not None and self.timing_csv_file:
            epoch_duration = time.time() - self._epoch_start_time
            with open(self.timing_csv_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([self.current_epoch, epoch_duration])
            self._epoch_start_time = None

    def forward(self, batch):
        state = None
        pred = None
        
        if "Reasoning" in self.gnn_type or "GHR" in self.gnn_type:
            for _ in range(self.reasoning_steps):
                pred, state = self.model(batch, state)
            return pred, state
        else:
            pred, state = self.model(batch, state=None)
            return pred, state

    def _shared_step(self, batch, step_name):
        if batch.mask.sum() == 0:
            return None
            
        total_loss = 0.0
        final_pred = None
        state = None
        
        if "Reasoning" in self.gnn_type or "GHR" in self.gnn_type:
            # Fix: Only randomize reasoning steps during training to keep eval deterministic
            if step_name == "train" and self.reasoning_steps > 1:
                steps = random.randint(max(1, self.reasoning_steps - 1), self.reasoning_steps + 1)
            else:
                steps = self.reasoning_steps

            for r in range(steps):
                pred, state = self.model(batch, state)
                
                step_loss = F.mse_loss(pred[batch.mask], batch.y[batch.mask])
                
                weight = self.gamma ** (steps - r - 1)
                total_loss = total_loss + (weight * step_loss)
                final_pred = pred 
                
        else:
            final_pred, state = self.model(batch, state=None)
            total_loss = F.mse_loss(final_pred[batch.mask], batch.y[batch.mask])

        with torch.no_grad():
            mae_loss = F.l1_loss(final_pred[batch.mask], batch.y[batch.mask])

        self.log(f"{step_name}_mse_loss", total_loss, batch_size=batch.num_graphs, prog_bar=False, sync_dist=False)
        self.log(f"{step_name}_mae", mae_loss, batch_size=batch.num_graphs, prog_bar=True, sync_dist=False)
        
        return total_loss  

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, "val")

    def test_step(self, batch, batch_idx):
        return self._shared_step(batch, "test")

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

def train_and_test_model(gnn_type: str, train_loader, val_loader, test_loader, use_swiglu: bool = False, smoke_test: bool = False):
    suffix = "_SwiGLU" if use_swiglu else ""
    print(f"\n{'='*50}")
    print(f"       Training {gnn_type}       ")
    print(f"{'='*50}")

    folder_name = f"{gnn_type}{suffix}"
    ckpt_dir = os.path.join("./range_test/checkpoints", folder_name)
    os.makedirs(ckpt_dir, exist_ok=True)
    
    ckpt_callback = ModelCheckpoint(
        dirpath=ckpt_dir,
        monitor="val_mae",          
        mode="min",                 
        save_top_k=1, 
        filename="best_{epoch:02d}-{val_mae:.4f}" 
    )

    model = LitAblationModel(
        gnn_type=gnn_type,
        input_dim=1,
        hidden_dim=32,
        num_layers=10,
        reasoning_steps=4, 
        lr=1e-3,
        use_swiglu=use_swiglu
    )

    trainer = L.Trainer(
        fast_dev_run=smoke_test,
        max_epochs=300,
        devices=1,
        accelerator="auto",
        precision=32,
        gradient_clip_val=1.0,
        callbacks=[
            EarlyStopping(monitor="val_mae", mode="min", patience=15),
            ckpt_callback
        ],
        enable_progress_bar=True,
        log_every_n_steps=10
    )

    trainer.fit(model, train_loader, val_loader)

    if smoke_test:
        print("Smoke test passed.")
        return 0.0
    print(f"\nTesting {gnn_type} ({suffix})...")
    metrics = trainer.test(model, test_loader, ckpt_path="best")

    return metrics[0]['test_mae']

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-test", action="store_true", dest="smoke_test",
                        help="Run 1 train+val batch on one model/seed to verify the pipeline")
    args, _ = parser.parse_known_args()
    smoke_test = args.smoke_test
    seeds = [42,43,44]
    torch.set_float32_matmul_precision('high')
    os.makedirs("./range_test", exist_ok=True)

    results = {}

    for use_swiglu in [True,False]:
        for model_name in models_map.keys():
            maes = []
            for seed in seeds:
                L.seed_everything(seed)
                print(f"\nRunning {model_name} ({'SwiGLU' if use_swiglu else 'ReLU'}) with seed {seed}")

                if smoke_test:
                    train_dataset = FixedRGGDataset(
                        num_samples=4, min_nodes=20, max_nodes=30, max_distance=20, target_avg_degree=6
                    )
                    val_dataset = FixedRGGDataset(
                        num_samples=4, min_nodes=20, max_nodes=30, max_distance=20, target_avg_degree=6
                    )
                    test_dataset = FixedRGGDataset(
                        num_samples=4, min_nodes=20, max_nodes=40, max_distance=40, target_avg_degree=6
                    )
                else:
                    train_dataset = FixedRGGDataset(
                        num_samples=6000, min_nodes=300, max_nodes=350, max_distance=20, target_avg_degree=12, cache_path=f"./ablation_data/data_train_seed{seed}.pt"
                    )
                    val_dataset = FixedRGGDataset(
                        num_samples=1000, min_nodes=300, max_nodes=350, max_distance=20, target_avg_degree=12, cache_path=f"./ablation_data/data_val_seed{seed}.pt"
                    )
                    test_dataset = FixedRGGDataset(
                        num_samples=1000, min_nodes=300, max_nodes=500, max_distance=40, target_avg_degree=12, cache_path=f"./ablation_data/data_test_ood_seed{seed}.pt"
                    )

                num_workers = 0 if smoke_test else (min(4, os.cpu_count() - 1) if os.cpu_count() else 0)
                train_loader = DataLoader(train_dataset, batch_size=4 if smoke_test else 32, shuffle=True, num_workers=num_workers, pin_memory=False, persistent_workers=num_workers > 0)
                val_loader = DataLoader(val_dataset, batch_size=4 if smoke_test else 32, shuffle=False, num_workers=num_workers, pin_memory=False, persistent_workers=num_workers > 0)
                test_loader = DataLoader(test_dataset, batch_size=4 if smoke_test else 32, shuffle=False, num_workers=num_workers, pin_memory=False, persistent_workers=num_workers > 0)

                test_mae = train_and_test_model(
                    gnn_type=model_name,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    test_loader=test_loader,
                    use_swiglu=use_swiglu,
                    smoke_test=smoke_test
                )
                if smoke_test:
                    return
                maes.append(test_mae)
            
            maes = np.array(maes)
            suffix = "SwiGLU" if use_swiglu else "ReLU"
            results[f"{model_name}_{suffix}"] = (maes.mean(), maes.std())

    print(f"\n{'='*15} FINAL RESULTS {'='*15}")
    for model_name, (mean_mae, std_mae) in results.items():
        print(f"{model_name:25s} Test MAE: {mean_mae:.4f} ± {std_mae:.4f}")
    print(f"{'='*45}")

if __name__ == "__main__":
    main()