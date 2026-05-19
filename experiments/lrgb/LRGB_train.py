import os
import argparse
import yaml
import torch
import torch.nn.functional as F
import lightning as L
from torch_geometric.loader import DataLoader
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
import random
from torchmetrics.regression import MeanAbsoluteError
from torch_geometric.transforms import BaseTransform, AddLaplacianEigenvectorPE
from peptides_structural import PeptidesStructuralDataset
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..'))
from ghr_model import GHRModel

class PaddedLaplacianPE(BaseTransform):
    def __init__(self, k, attr_name='pe'):
        self.k = k
        self.attr_name = attr_name

    def forward(self, data):
        safe_k = min(self.k, data.num_nodes - 1)
        
        if safe_k <= 0:
            pe_zeros = torch.zeros((data.num_nodes, self.k), dtype=torch.float32)
            setattr(data, self.attr_name, pe_zeros)
            return data

        temp_transform = AddLaplacianEigenvectorPE(
            k=safe_k, attr_name=self.attr_name, is_undirected=True
        )
        data = temp_transform(data)
        
        pe = getattr(data, self.attr_name)
        if pe.size(1) < self.k:
            padding = torch.zeros((pe.size(0), self.k - pe.size(1)), dtype=pe.dtype, device=pe.device)
            padded_pe = torch.cat([pe, padding], dim=1)
            setattr(data, self.attr_name, padded_pe)
            
        return data

class LitPeptidesModel(L.LightningModule):
    def __init__(
        self,
        hidden_dim: int = 150,
        L_steps: int = 4,
        H_steps: int = 4,
        use_swiglu: bool = True,
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
        reasoning_steps: int = 1,
        dropout: float = 0.0
    ):
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr
        self.weight_decay = weight_decay
        self.reasoning_steps = reasoning_steps
        self.gamma = 0.8
        
        self.model = GHRModel(
            output_dim=11,
            hidden_dim=hidden_dim,
            l_steps=L_steps,
            h_steps=H_steps,
            use_swiglu=use_swiglu,
            dropout=dropout,
            node_level_task=False,
            scatter_agg='add',
            pe_dim=16,
            linear_graph_head=True,
        )
        
        self.train_mae = MeanAbsoluteError()
        self.val_mae = MeanAbsoluteError()
        self.test_mae = MeanAbsoluteError()

    def forward(self, batch):
        state = None
        pred = None
        for _ in range(self.reasoning_steps):
            pred, state = self.model(batch, state)
        return pred, state

    def _shared_step(self, batch, step_name):
        y = batch.y.float()
        
        state = None
        total_loss = 0.0
        final_pred = None
        
        if step_name == "train" and self.reasoning_steps > 1:
            steps = random.randint(self.reasoning_steps - 1, self.reasoning_steps + 1)
        else:
            steps = self.reasoning_steps
            
        for r in range(steps):
            pred, state = self.model(batch, state)
            
            step_loss = F.l1_loss(pred, y)
            
            weight = self.gamma ** (steps - r - 1)
            total_loss = total_loss + (weight * step_loss)
            final_pred = pred
            
        metric_fn = getattr(self, f"{step_name}_mae")
        metric_fn.update(final_pred, y)
        
        self.log(f"{step_name}_loss", total_loss, batch_size=batch.num_graphs, prog_bar=True, sync_dist=False)
        return total_loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, "val")

    def test_step(self, batch, batch_idx):
        return self._shared_step(batch, "test")

    def on_train_epoch_end(self):
        self.log('train_mae', self.train_mae.compute(), prog_bar=True, sync_dist=False)
        self.train_mae.reset()

    def on_validation_epoch_end(self):
        self.log('val_mae', self.val_mae.compute(), prog_bar=True, sync_dist=False)
        self.val_mae.reset()

    def on_test_epoch_end(self):
        self.log('test_mae', self.test_mae.compute(), prog_bar=True, sync_dist=False)
        self.test_mae.reset()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='min', 
            factor=0.5, 
            patience=10, 
            min_lr=1e-6
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_mae"
            }
        }

def main():
    parser = argparse.ArgumentParser(description="Train GHR on LRGB Peptides-struct (3 seeds, results averaged).")
    parser.add_argument("--config", type=str, default="LRGB.yaml", help="Path to YAML config file")
    parser.add_argument("--smoke-test", action="store_true", dest="smoke_test",
                        help="Run 1 train+val batch on a single seed to verify the pipeline")
    args = parser.parse_args()

    torch.set_float32_matmul_precision('high')
    os.makedirs("./checkpoints", exist_ok=True)

    with open(args.config, "r") as file:
        config = yaml.safe_load(file)

    lap_transform = PaddedLaplacianPE(k=16, attr_name='pe')
    dataset = PeptidesStructuralDataset(root='datasets', pre_transform=lap_transform)
    splits = dataset.get_idx_split()

    train_dataset = dataset[splits['train']]
    val_dataset = dataset[splits['val']]
    test_dataset = dataset[splits['test']]

    num_workers = min(4, os.cpu_count() - 1) if os.cpu_count() else 0
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=num_workers)

    results = {}

    for seed in ([1, 2, 3] if not args.smoke_test else [1]):
        L.seed_everything(seed)
        run_name = f"seed_{seed}"
        
        ckpt_dir = f"./checkpoints/{run_name}"
        os.makedirs(ckpt_dir, exist_ok=True)

        ckpt_callback = ModelCheckpoint(
            dirpath=ckpt_dir,
            monitor="val_mae",          
            mode="min",                 
            save_top_k=1, 
            filename="best_model-{epoch:02d}-{val_mae:.4f}" 
        )
        early_stop = EarlyStopping(monitor="val_mae", mode="min", patience=25)

        model = LitPeptidesModel(
            hidden_dim=config["hidden_dim"], 
            L_steps=config["L_steps"],
            H_steps=config["H_steps"],
            use_swiglu=config["use_swiglu"],
            lr=config["lr"],
            weight_decay=config["weight_decay"],
            reasoning_steps=config["reasoning_steps"],
            dropout=config["dropout"]
        )

        trainer = L.Trainer(
            fast_dev_run=args.smoke_test,
            max_epochs=200,
            devices=1,
            accelerator="auto",
            precision=32,
            gradient_clip_val=1.0,
            callbacks=[early_stop, ckpt_callback],
            enable_progress_bar=True,
            log_every_n_steps=10
        )

        trainer.fit(model, train_loader, val_loader)

        if not args.smoke_test:
            metrics = trainer.test(model, test_loader, ckpt_path="best")
            results[run_name] = metrics[0]['test_mae']
        else:
            print("Smoke test passed.")

    if not args.smoke_test:
        print(results)

if __name__ == "__main__":
    main()