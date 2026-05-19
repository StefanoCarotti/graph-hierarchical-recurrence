import torch
import lightning as L
import time
import csv
import pathlib
import random
from typing import Optional
from torch_geometric.data.lightning import LightningDataset

from models.ghr_model import GHRModel

def convert_to_lit_dataset(data):
    return LightningDataset(data)

class LitGraphNN(L.LightningModule):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: Optional[int] = None,
        num_layers: int = 1,
        node_level_task: bool = False,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        scaling_factor: float = 1.0,
        enable_timing: bool = False,
        timing_csv_base_path: str = "training_timings",
        task: str = "sssp",
        num_reasoning_steps: int = 4,
        gamma: float = 0.8,
        **kwargs,
    ) -> None:
        super().__init__()
        self.num_reasoning_steps = num_reasoning_steps
        self.gamma = gamma
        self.conv_layer = kwargs.get("conv_layer")
        self.enable_timing = enable_timing
        self._epoch_start_time = None
        self.timing_csv_file = None
        self.task = task
        self.scaling_factor = scaling_factor

        self.model = GHRModel(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            node_level_task=node_level_task,
            **kwargs,
        )
            
        self.criterion = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=lr, weight_decay=weight_decay
        )

        self.save_hyperparameters()

        if self.enable_timing:
            self.timing_csv_base_path = pathlib.Path(timing_csv_base_path)
            try:
                self.timing_csv_base_path.mkdir(parents=True, exist_ok=True)
            except OSError:
                self.enable_timing = False

        if self.enable_timing:
            timing_filename_parts = ["GHRModel"]
            if self.conv_layer:
                timing_filename_parts.append(str(self.conv_layer))
            timing_filename_parts.append(str(self.task))
            timing_filename_parts.append("timing.csv")
            filename = "_".join(filter(None, timing_filename_parts))
            self.timing_csv_file = self.timing_csv_base_path / filename

            if not self.timing_csv_file.exists():
                try:
                    with open(self.timing_csv_file, 'w', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow(["epoch", "training_time_seconds"])
                except OSError:
                    self.timing_csv_file = None
                    self.enable_timing = False
    
    def on_train_epoch_start(self):
        if self.enable_timing:
            self._epoch_start_time = time.time()

    def on_train_epoch_end(self):
        if self.enable_timing and self._epoch_start_time is not None and self.timing_csv_file:
            epoch_duration = time.time() - self._epoch_start_time
            current_epoch_to_log = self.current_epoch
            try:
                with open(self.timing_csv_file, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([current_epoch_to_log, epoch_duration])
            except Exception:
                pass
            self._epoch_start_time = None

    def forward(self, data):
        return self.model(data)

    def training_step(self, batch, batch_idx):
        state = None
        total_loss = 0
        
        steps = max(1, random.randint(self.num_reasoning_steps - 1, self.num_reasoning_steps + 1))
        
        for r in range(steps):
            out, state = self.model(batch, state=state)
            out = out.squeeze(-1)
            
            step_loss = torch.nn.functional.mse_loss(out, batch.y)
            
            weight = self.gamma ** (steps - r - 1)
            total_loss = total_loss + (weight * step_loss)
        
        loss = torch.log10(total_loss)
        final_out = out 

        self.log("train_loss", loss, sync_dist=True, batch_size=batch.y.size(0))
        self.log(
            "train_mae",
            torch.nn.functional.l1_loss(final_out.detach() * self.scaling_factor, batch.y * self.scaling_factor),
            sync_dist=True, batch_size=batch.y.size(0),
        )
        self.log(
            "train_mse",
            torch.nn.functional.mse_loss(final_out.detach() * self.scaling_factor, batch.y * self.scaling_factor), 
            sync_dist=True, batch_size=batch.y.size(0),
        )
        return loss

    def validation_step(self, batch, batch_idx):
        if batch.x.dtype == torch.float64:
            batch.x = batch.x.float()
        if batch.edge_attr is not None and batch.edge_attr.dtype == torch.float64:
            batch.edge_attr = batch.edge_attr.float()
        if batch.y.dtype == torch.float64:
            batch.y = batch.y.float()
        
        state = None
        for _ in range(self.num_reasoning_steps):
            final_out, state = self.model(batch, state=state)
        final_out = final_out.squeeze(-1)

        loss = self.criterion(final_out, batch.y)
        loss = torch.log10(loss)

        if self.task == "energy":
            final_out = 10**final_out.detach()
            batch.y = 10**batch.y

        self.log("val_loss", loss, sync_dist=True, batch_size=batch.y.size(0))
        self.log(
            "val_mae",
            torch.nn.functional.l1_loss(
                final_out.detach() * self.scaling_factor, batch.y * self.scaling_factor
            ),
            sync_dist=True,
            prog_bar=True,
            batch_size=batch.y.size(0),
        )
        self.log(
            "val_mse",
            torch.nn.functional.mse_loss(
                final_out.detach() * self.scaling_factor, batch.y * self.scaling_factor
            ),
            sync_dist=True,
            batch_size=batch.y.size(0),
        )
        return loss

    def test_step(self, batch, batch_idx):        
        state = None
        for _ in range(self.num_reasoning_steps):
            final_out, state = self.model(batch, state=state)
        final_out = final_out.squeeze(-1)

        loss = self.criterion(final_out, batch.y)
        loss = torch.log10(loss)

        if self.task == "energy":
            final_out = 10**final_out.detach()
            batch.y = 10**batch.y

        self.log("test_loss", loss, sync_dist=True, batch_size=batch.y.size(0))
        self.log(
            "test_mae",
            torch.nn.functional.l1_loss(
                final_out.detach() * self.scaling_factor, batch.y * self.scaling_factor
            ),
            sync_dist=True,
            batch_size=batch.y.size(0),
        )
        self.log(
            "test_mse",
            torch.nn.functional.mse_loss(
                final_out.detach() * self.scaling_factor, batch.y * self.scaling_factor
            ),
            sync_dist=True,
            batch_size=batch.y.size(0),
        )
        return loss

    def configure_optimizers(self):
        return self.optimizer
    
    def __str__(self):
        params = ", ".join(f"{k}={v}" for k, v in self.hparams.items())
        return f"LitGraphNN({params})" + f" with model: {str(self.model)}"