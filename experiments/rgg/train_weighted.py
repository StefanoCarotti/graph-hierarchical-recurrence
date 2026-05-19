import os
import time
import csv
import pathlib
import torch
import torch.nn.functional as F
import lightning as L
from torch_geometric.loader import DataLoader
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
import random
from dist_dataset import FixedRGGDataset
from ablation_models import *
# 1. Model Registry
models_map = {
    "DeepGIN": DeepGIN,
    "RecursiveGIN": RecursiveGIN,
    "ReasoningDeepGIN": ReasoningDeepGIN,
    "ReasoningRecursiveGIN": ReasoningRecursiveGIN,
    "GHR": GHR,
    "GHRAntiSymmetric": GHRAntiSymmetric,
    "GHRGCN": GHRGCN,   
    "GHRGAT": GHRGAT    
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
        timing_csv_base_path: str = "./ablation_data/training_timings",
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

        # 2. Dynamic Model Instantiation
        if "GHR" in gnn_type:
            self.model = models_map[gnn_type](
                input_dim=input_dim, 
                hidden_dim=hidden_dim, 
                L_steps=6, 
                H_steps=3,
                use_swiglu=self.use_swiglu
            )
        elif "Recursive" in gnn_type:
            # Recursive forms use num_iterations
            self.model = models_map[gnn_type](input_dim=input_dim, hidden_dim=hidden_dim, num_iterations=num_layers, use_swiglu=self.use_swiglu)
        else:
            # Deep forms use num_layers
            self.model = models_map[gnn_type](input_dim=input_dim, hidden_dim=hidden_dim, num_layers=num_layers, use_swiglu=self.use_swiglu)

        # 3. Timing CSV Setup
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
        # Forward is purely for inference , training is in _shared_step
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
            steps = random.randint(self.reasoning_steps -1 , self.reasoning_steps +1)  # Randomize steps only during training
            for r in range(steps):
                pred, state = self.model(batch, state)
                
                # Masked MSE for this specific step
                step_loss = F.mse_loss(pred[batch.mask], batch.y[batch.mask])
                
                # Discounted weight accumulation
                weight = self.gamma ** (steps - r - 1)
                total_loss = total_loss + (weight * step_loss)
                final_pred = pred  # Store the last pred for MAE tracking
                
        # --- STANDARD GIN MODELS ---
        else:
            final_pred, state = self.model(batch, state=None)
            total_loss = F.mse_loss(final_pred[batch.mask], batch.y[batch.mask])

        with torch.no_grad():
            mae_loss = F.l1_loss(final_pred[batch.mask], batch.y[batch.mask])

        # Logging
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
    ckpt_dir = os.path.join("./ablation_data/checkpoints_weighted", folder_name)    
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
        num_layers=20,
        reasoning_steps=4, 
        lr=1e-3,
        use_swiglu= use_swiglu

    )

    trainer = L.Trainer(
        fast_dev_run=smoke_test,
        max_epochs=300,
        devices=1 if torch.cuda.is_available() else "auto",
        accelerator="gpu" if torch.cuda.is_available() else 'cpu',
        precision=32 if torch.cuda.is_available() else None,
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
                        help="Run 1 train+val batch on one model to verify the pipeline")
    args, _ = parser.parse_known_args()
    smoke_test = args.smoke_test
    torch.set_float32_matmul_precision('high')
    L.seed_everything(42)
    os.makedirs("./ablation_data", exist_ok=True)

    # 1. Dataset Generation/Loading
    print("Pre-loading FixedRGG Datasets...")
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
            num_samples=8000, min_nodes=300, max_nodes=350, max_distance=20, target_avg_degree=12,
            cache_path="./ablation_data/dist_train.pt"
        )
        val_dataset = FixedRGGDataset(
            num_samples=1000, min_nodes=300, max_nodes=350, max_distance=20, target_avg_degree=12,
            cache_path="./ablation_data/dist_val.pt"
        )
        test_dataset = FixedRGGDataset(
            num_samples=1000, min_nodes=300, max_nodes=500, max_distance=40, target_avg_degree=12,
            cache_path="./ablation_data/dist_test_ood.pt"
        )

    num_workers = 0 if smoke_test else (min(4, os.cpu_count() - 1) if os.cpu_count() else 0)
    train_loader = DataLoader(train_dataset, batch_size=4 if smoke_test else 32, shuffle=True, num_workers=num_workers, pin_memory=False, persistent_workers=num_workers > 0)
    val_loader = DataLoader(val_dataset, batch_size=4 if smoke_test else 32, shuffle=False, num_workers=num_workers, pin_memory=False, persistent_workers=num_workers > 0)
    test_loader = DataLoader(test_dataset, batch_size=4 if smoke_test else 32, shuffle=False, num_workers=num_workers, pin_memory=False, persistent_workers=num_workers > 0)

    # 2. Iterate through ALL models in the map
    results = {}
    for use_swiglu in [True, False]:
        for model_name in models_map.keys():
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
            suffix = "SwiGLU" if use_swiglu else "ReLU"
            results[f"{model_name}_{suffix}"] = test_mae

    # 3. Final Summary Output
    print(f"\n{'='*15} FINAL RESULTS {'='*15}")
    for model_name, mae in results.items():
        print(f"{model_name:25s} Test MAE: {mae:.4f}")
    print(f"{'='*45}")

if __name__ == "__main__":
    main()