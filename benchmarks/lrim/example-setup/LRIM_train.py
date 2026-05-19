import os
import argparse
import yaml
import random
import torch
import torch.nn.functional as F
import lightning as L
from torch_geometric.loader import DataLoader
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from torchmetrics.regression import MeanSquaredError

# Local imports from the same folder
from lrim_loader import LRIM
from LRIM_transform import GridBlockTransform
from LRIM_model import GHRModel

class LitLRIMModel(L.LightningModule):
    def __init__(
        self,
        hidden_dim: int = 32,
        L_steps: int = 4,
        H_steps: int = 4,
        use_swiglu: bool = False,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        reasoning_steps: int = 4,
        gamma: float = 0.4,
        conv_type: str = 'GINE',
    ):
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr
        self.weight_decay = weight_decay
        self.reasoning_steps = reasoning_steps
        self.gamma = gamma
        
        self.model = GHRModel(
            input_dim=1, 
            output_dim=1,
            hidden_dim=hidden_dim, 
            l_steps=L_steps, 
            h_steps=H_steps,
            use_swiglu=use_swiglu,
            chem_task=False,
            node_level_task=True,
            conv_type=conv_type
        )
        
        self.train_mse = MeanSquaredError()
        self.val_mse = MeanSquaredError()
        self.test_mse = MeanSquaredError()

    def forward(self, batch, state=None):
        return self.model(batch, state)

    def _shared_step(self, batch, step_name):
        y = batch.y.float().view(-1)
        state = None
        total_loss = 0.0
        final_pred = None
        
        if step_name == "train" and self.reasoning_steps > 1:
            steps = max(1, random.randint(self.reasoning_steps - 1, self.reasoning_steps + 1))
        else:
            steps = self.reasoning_steps
            
        for r in range(steps):
            pred, state = self.model(batch, state=state)
            pred = pred.view(-1)
            
            step_loss = F.mse_loss(pred, y)
            
            weight = self.gamma ** (steps - r - 1)
            total_loss = total_loss + (weight * step_loss)
            final_pred = pred
            
        metric_fn = getattr(self, f"{step_name}_mse")
        metric_fn.update(final_pred, y)
        
        self.log(f"{step_name}_batch_loss", total_loss, batch_size=batch.num_graphs, prog_bar=False)
        return total_loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, "val")

    def test_step(self, batch, batch_idx):
        return self._shared_step(batch, "test")

    def on_train_epoch_end(self):
        mse = self.train_mse.compute()
        log_mse = torch.log10(mse)
        self.log('train_log_mse', log_mse, prog_bar=True, sync_dist=False)
        self.train_mse.reset()

    def on_validation_epoch_end(self):
        mse = self.val_mse.compute()
        log_mse = torch.log10(mse)
        self.log('val_log_mse', log_mse, prog_bar=True, sync_dist=False)
        self.val_mse.reset()

    def on_test_epoch_end(self):
        mse = self.test_mse.compute()
        log_mse = torch.log10(mse)
        self.log('test_log_mse', log_mse, prog_bar=True, sync_dist=False)
        self.test_mse.reset()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='min', 
            factor=0.5, 
            patience=20, 
            min_lr=1e-6
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_log_mse"
            }
        }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config file")
    
    # Dataset and Training settings
    parser.add_argument("--dataset_name", type=str, default="lrim_16_0.6_10k")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--block_size", type=int, default=2)
    parser.add_argument("--seed", type=int, default=41)
    
    # Model Hyperparameters
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--L_steps", type=int, default=3)
    parser.add_argument("--H_steps", type=int, default=3)
    parser.add_argument("--use_swiglu", action="store_true")
    parser.add_argument("--lr", type=float, default=0.0003)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--reasoning_steps", type=int, default=3)
    parser.add_argument("--gamma", type=float, default=0.8)
    parser.add_argument("--conv_type", type=str, default="GatedGCN")
    parser.add_argument("--smoke-test", action="store_true", dest="smoke_test",
                        help="Run 1 train+val batch to verify the pipeline")

    args, _ = parser.parse_known_args()

    # Load from YAML if provided
    if args.config and os.path.exists(args.config):
        with open(args.config, "r") as file:
            yaml_config = yaml.safe_load(file)
            for key, value in yaml_config.items():
                setattr(args, key, value)

    torch.set_float32_matmul_precision('high')
    L.seed_everything(args.seed)
    os.makedirs("./checkpoints", exist_ok=True)

    print(f"Loading {args.dataset_name} Dataset...")
    dataset = LRIM(
        root='data/', 
        name=args.dataset_name,
        pre_transform=GridBlockTransform(block_size=args.block_size)
    )
    splits = dataset.get_idx_split()

    train_dataset = dataset[splits['train']]
    val_dataset = dataset[splits['val']]
    test_dataset = dataset[splits['test']]

    num_workers = min(4, os.cpu_count() - 1) if os.cpu_count() else 0
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=num_workers)

    # Dynamic run name based on config
    run_name = f"GHR_{args.dataset_name}_b{args.block_size}_h{args.hidden_dim}_L{args.L_steps}_H{args.H_steps}_{args.conv_type}" 
    print(f"\n{'='*50}\n Starting Run: {run_name} \n{'='*50}\n")
    
    ckpt_dir = f"./checkpoints/{run_name}"
    os.makedirs(ckpt_dir, exist_ok=True)

    ckpt_callback = ModelCheckpoint(
        dirpath=ckpt_dir,
        monitor="val_log_mse",          
        mode="min",                 
        save_top_k=1, 
        filename="best_model-{epoch:02d}-{val_log_mse:.4f}" 
    )
    early_stop = EarlyStopping(monitor="val_log_mse", mode="min", patience=80)

    model = LitLRIMModel(
        hidden_dim=args.hidden_dim, 
        L_steps=args.L_steps,
        H_steps=args.H_steps,
        use_swiglu=args.use_swiglu,
        lr=args.lr,
        weight_decay=args.weight_decay,
        reasoning_steps=args.reasoning_steps,
        gamma=args.gamma,
        conv_type=args.conv_type
    )

    trainer = L.Trainer(
        fast_dev_run=args.smoke_test,
        max_epochs=2000,
        devices=1,
        accelerator="auto",
        precision=32,
        gradient_clip_val=1.0,
        accumulate_grad_batches=1,
        callbacks=[early_stop, ckpt_callback],
        enable_progress_bar=True,
        log_every_n_steps=10
    )

    trainer.fit(model, train_loader, val_loader)

    if not args.smoke_test:
        metrics = trainer.test(model, test_loader, ckpt_path="best")
        print(f"\n{'='*50}\nFinal Results Summary\n{'='*50}")
        print(f"GHR Test Log-MSE: {metrics[0]['test_log_mse']:.4f}")
    else:
        print("Smoke test passed.")

if __name__ == "__main__":
    main()