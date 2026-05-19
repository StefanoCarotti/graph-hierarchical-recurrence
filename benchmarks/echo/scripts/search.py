import os
import sys
sys.path.append('..')

import yaml
import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.optuna import OptunaSearch
import pandas as pd
import argparse
from pathlib import Path



SEEDS = [43]


def create_data_loaders(task, config):
    """Create train, validation, and test data loaders."""
    sys.path.append('..')
    from utils import get_dataset, KHopTransform
    from torch_geometric.loader import DataLoader
    
    try:
        from utils.ghrtransform import GHRTransform
    except ImportError:
        GHRTransform = None
    # Figure out the absolute path to the main project's data folder
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    data_dir = os.path.join(project_root, "data")


    batch_size = config.get("batch_size", 256)
    
    # Figure out which transform to apply dynamically (but DO NOT pass as pre_transform)
    transform_func = None
    if config.get("gnn_type") in ("GHR", "GHRModel") and GHRTransform is not None:
        transform_func = GHRTransform()
    elif config.get("gnn_type") == "DRew_GCN":
        transform_func = KHopTransform(k=config.get("k_hop", 1))

    data_train, data_val, data_test, num_feat, num_class = get_dataset(
        root=data_dir,   # <-- Use absolute path here instead of "./data"
        task=task,
        pre_transform=None, # Keep None to protect the disk cache!
        constant_feature=None if task in ["charge", "energy"] else 1.0
    )

    # Attach transforms dynamically just like in train.py
    if transform_func is not None:
        data_train.transform = transform_func
        data_val.transform = transform_func
        data_test.transform = transform_func

    scaling_factor = data_train.scaling_factor[task]
    if scaling_factor is None and task in ["charge", "energy"]:
        scaling_factor = 1.0
    
    loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": min(4, os.cpu_count() - 1),
        "pin_memory": True,
        "persistent_workers": True
    }
    
    train_loader = DataLoader(data_train, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(data_val, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(data_test, shuffle=False, **loader_kwargs)
    
    return train_loader, val_loader, test_loader, num_feat, num_class, scaling_factor


def create_model_and_trainer(config, num_feat, num_class, scaling_factor, task, logger_type="wandb"):
    """Create the model and trainer with appropriate callbacks."""
    import lightning as L
    from lightning.pytorch.loggers import WandbLogger, CSVLogger
    from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
    from utils.litmodels import LitGraphNN
    from ray.tune.integration.pytorch_lightning import TuneReportCallback # <-- Add this

    
    logger = CSVLogger(save_dir=f"./logs/csv/{task}", name="logs")
    experiment_name = f"version_{logger.version}"
    
    model = LitGraphNN(
        input_dim=num_feat,
        output_dim=num_class,
        node_level_task=False if task in ["diam", "energy"] else True,
        scaling_factor=scaling_factor or 1.0,
        chem_task=(task in ["charge", "energy"]),  
        **config,
    )
    
    # checkpoint callback
    layer_name = config.get("conv_layer", config.get("gnn_type", "unknown"))
    
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    ckpt_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath=os.path.join(project_root, "checkpoints", task, layer_name, experiment_name),
        save_top_k=3,
        filename="{epoch:02d}-{val_loss:.4f}",
        mode="min",
    )
    tune_callback = TuneReportCallback(
        {
            "val_loss": "val_loss",
        },
        on="validation_end"
    )

    es_callback = EarlyStopping(
        monitor="val_loss",
        patience=50,
    )

    callbacks = [es_callback, ckpt_callback, tune_callback]

    
    trainer = L.Trainer(
        max_epochs=config.get("max_epochs", 500),
        accelerator="gpu",
        logger=logger,
        log_every_n_steps=50,
        gradient_clip_val=1.0,
        callbacks=callbacks,
        enable_progress_bar=False,
    )
    
    return model, trainer


def train_model_tune(config):
    """Training function compatible with Ray Tune"""
    import lightning as L
    #import wandb
    import torch
    from ray.air import session
    
    #set matmul precision
    torch.set_float32_matmul_precision("high")

    L.seed_everything(SEEDS[0])
    
    
    task = config["task"]
    
    # Create data loaders
    train_loader, val_loader, test_loader, num_feat, num_class, scaling_factor = create_data_loaders(task, config)
    
    # Create model and trainer
    model, trainer = create_model_and_trainer(config, num_feat, num_class, scaling_factor, task)
    
    # Train model
    trainer.fit(model, train_loader, val_loader)
    
    # Get results
    val_results = trainer.validate(model, val_loader, ckpt_path="best")
    test_results = trainer.test(model, test_loader, ckpt_path="best")
    
    # Report to Ray Tune
    session.report({
        "val_loss": val_results[0]["val_loss"],
        "val_mae": val_results[0].get("val_mae", val_results[0]["val_loss"]),
        "test_loss": test_results[0]["test_loss"],
        "test_mae": test_results[0].get("test_mae", test_results[0]["test_loss"])
    })
    
    

def load_search_space(config_path):
    """Load YAML config and convert to Ray Tune search space."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    search_space = {}
    for param, values in config["parameters"].items():
        if isinstance(values, dict):
            if "min" in values and "max" in values:
                if isinstance(values["min"], int) and isinstance(values["max"], int):
                    search_space[param] = tune.randint(values["min"], values["max"] + 1)
                else:
                    search_space[param] = tune.uniform(values["min"], values["max"])
            elif "values" in values:
                search_space[param] = tune.choice(values["values"])
        elif isinstance(values, list):
            search_space[param] = tune.choice(values)
    
    model_name = config.get("gnn_type")
    if model_name:
        search_space["gnn_type"] = model_name
    
    return search_space, model_name


def save_trial_results(trials, task, output_dir):
    """Save all trial results to CSV file."""
    csv_file = os.path.join(output_dir, f"search_{task}.csv")
    
    results = []
    for trial in trials:
        if not trial.last_result:
            continue
            
        # Combine config and results
        row_data = {**trial.config, **trial.last_result}
        row_data["trial_id"] = getattr(trial, 'trial_id', "")
        results.append(row_data)
    
    if not results:
        return
    
    df = pd.DataFrame(results)
    
    # Reorder columns for better readability
    priority_cols = ['conv_layer', 'gnn_type', 'val_mae', 'val_loss', 'test_mae', 'test_loss']
    other_cols = [col for col in df.columns if col not in priority_cols]
    column_order = [col for col in priority_cols if col in df.columns] + other_cols
    
    df = df[column_order]
    df.to_csv(csv_file, index=False)


def get_config_files(task, model_names=None):
    """Get YAML config files for a task, optionally filtered by model names."""
    
    # Map 'charge' task to the 'chem' search space directory
    search_dir_name = "chem" if task == "charge" else task
    config_dir = Path(f"../search-space/{search_dir_name}")
    
    if not config_dir.exists():
        print(f"Warning: Config directory {config_dir} does not exist")
        return []
    
    yaml_files = list(config_dir.glob("*.yaml"))
    
    if model_names:
        yaml_files = [f for f in yaml_files if f.stem in model_names]
        if not yaml_files:
            print(f"No config files found for models {model_names} in task {task} (using folder: {search_dir_name})")
    
    return yaml_files


def create_scheduler_and_search_alg(args):
    """Create scheduler and search algorithm based on arguments."""
    scheduler = None
    if args.scheduler == "asha":
        scheduler = ASHAScheduler(
            time_attr="training_iteration",
            metric="val_loss",
            mode="min",
            max_t=1000,
            grace_period=50,
            reduction_factor=2,
        )
    
    search_alg = None
    if args.search_alg == "optuna":
        search_alg = OptunaSearch(metric="val_loss", mode="min")
    
    return scheduler, search_alg


def run_experiments_for_task(task, args, scheduler, search_alg):
    """Run all experiments for a single task."""
    config_files = get_config_files(task, args.models)
    if not config_files:
        print(f"No config files found for task {task}")
        return
    
    all_trials = []
    
    for config_file in config_files:
        search_space, gnn_type = load_search_space(config_file)
        print(f"Processing {gnn_type} for task {task}")
        
        # Add task and override parameters
        search_space["task"] = task
        search_space["max_epochs"] = args.max_epochs

        
        # Run Ray Tune
        analysis = tune.run(
            train_model_tune,
            config=search_space,
            num_samples=args.n_samples,
            scheduler=scheduler,
            search_alg=search_alg,
            resources_per_trial={"cpu": 8, "gpu": 1.0},
            name=f"{task}_{gnn_type}",
            storage_path="/tmp/ray_results",
            verbose=1,
        )
        
        all_trials.extend(analysis.trials)
    
    # Save all results for this task
    save_trial_results(all_trials, task, args.output_dir)
    print(f"Saved {len(all_trials)} trials for task {task}")


def print_experiment_summary(tasks, output_dir):
    """Print summary of all experiments."""
    print("\nExperiment Summary:")
    for task in tasks:
        csv_file = os.path.join(output_dir, f"search_{task}.csv")
        if not os.path.exists(csv_file):
            continue
            
        df = pd.read_csv(csv_file)
        print(f"\nTask {task}:")
        print(f"  Total experiments: {len(df)}")
        
        if "gnn_type" in df.columns:
            print(f"  GNN types: {df['gnn_type'].unique()}")
        
        if "val_loss" in df.columns:
            valid_val_loss = df["val_loss"].dropna()
            if len(valid_val_loss) > 0:
                print(f"  Best validation loss: {valid_val_loss.min():.4f}")


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_samples", type=int, default=32, help="Number of random samples per configuration")
    parser.add_argument("--tasks", nargs="+", default=["diam", "sssp", "ecc"], help="Tasks to run experiments on")
    parser.add_argument("--output_dir", type=str, default="./results", help="Directory to save results CSV files")
    parser.add_argument("--num_cpus", type=int, default=None, help="Number of CPUs to use for Ray")
    parser.add_argument("--num_gpus", type=int, default=None, help="Number of GPUs to use for Ray")
    parser.add_argument("--models", nargs="+", default=None, help="Specific model names to test")
    #parser.add_argument("--entity_name", type=str, default=ENTITY_NAME, help="Wandb entity name for logging")
    parser.add_argument("--scheduler", type=str, default="asha", choices=["asha", "none"], help="Scheduler to use")
    parser.add_argument("--search_alg", type=str, default="optuna", choices=["optuna", "random"], help="Search algorithm")
    #parser.add_argument("--logger", type=str, default="csv", choices=["wandb", "csv"], help="Logger to use (wandb or csv)")
    parser.add_argument("--max_epochs", type=int, default=500, help="Maximum number of epochs for training")
    
    args = parser.parse_args()
    
    return args


def main():
    args = parse_arguments()
    
    # Initialize Ray
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    ray.init(
        num_cpus=args.num_cpus, 
        num_gpus=args.num_gpus,
        runtime_env={
            "working_dir": project_root,
            "excludes": ["data/", "checkpoints/", "logs/", ".git/", "__pycache__/", ".vscode/"]
        }
    )
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create scheduler and search algorithm
    scheduler, search_alg = create_scheduler_and_search_alg(args)
    
    # Run experiments for each task
    for task in args.tasks:
        print(f"Processing task: {task}")
        run_experiments_for_task(task, args, scheduler, search_alg)
    
    # Print summary
    print_experiment_summary(args.tasks, args.output_dir)


if __name__ == "__main__":
    main()
    ray.shutdown()

