import torch
import torch.nn as nn
import numpy as np
import time
from city_dataset2 import CityNetworkH3
from MODEL import GHRModel

# --- HELPER: Convert Continuous Distance to Classes ---
def get_quantile_thresholds(raw_eccentricities, n_bins=10):
    """
    Computes the boundaries for the 10 classes based on the WHOLE graph 
    (mimicking the original paper's setup).
    """
    quantiles = np.linspace(0, 1, n_bins + 1)
    thresholds = np.quantile(raw_eccentricities.cpu().numpy(), quantiles)
    # Extend range to -inf and +inf to avoid boundary errors
    thresholds[0] = -np.inf
    thresholds[-1] = np.inf
    return thresholds

def compute_accuracy(pred_dist, true_dist, thresholds, mask):
    """
    Bins the continuous predictions and ground truth into classes and compares.
    """
    # 1. Digitize both predictions and ground truth
    pred_classes = np.digitize(pred_dist.detach().cpu().numpy(), thresholds) - 1
    true_classes = np.digitize(true_dist.detach().cpu().numpy(), thresholds) - 1
    
    # 2. Clip (Just in case)
    pred_classes = np.clip(pred_classes, 0, 9)
    true_classes = np.clip(true_classes, 0, 9)
    
    # 3. Compare
    pred = torch.tensor(pred_classes, device=mask.device)
    true = torch.tensor(true_classes, device=mask.device)
    correct = (pred[mask] == true[mask]).float().sum()
    total = mask.sum().float()
    
    return (correct / total).item()

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-test", action="store_true", dest="smoke_test",
                        help="Run 2 epochs with 1 reasoning step to verify the pipeline")
    args, _ = parser.parse_known_args()

    # --- CONFIGURATION ---
    CITY_NAME = 'paris'
    H3_RES = 9
    HIDDEN_DIM = 32
    LR = 0.001
    EPOCHS = 2 if args.smoke_test else 2000

    # REASONING CONFIG
    MODEL_INTERNAL_STEPS = 2
    REASONING_STEPS = 1 if args.smoke_test else 4
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # --- LOAD DATA ---
    dataset = CityNetworkH3(root='./data', name=CITY_NAME, resolution=H3_RES, minimalist=True, regression=True)
    data = dataset[0].to(device)
    thresholds = get_quantile_thresholds(data.y_raw) # For accuracy checking
    print(f"Graph Loaded: {data.num_nodes} nodes.")
    print(f"Input Features: {data.x.shape[1]} (Should be 2: x, y)")
    print(f"Edge Features: {data.edge_attr.shape[1] if data.edge_attr is not None else 'None'} (Should be 1: Length)")
    
    # --- MODEL ---
    model = GHRModel(
        input_dim=2, edge_dim=1, hidden_dim=HIDDEN_DIM,
        L_steps=3,
        H_steps=2 
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.L1Loss()

    print(f"\nStarting Training...")
    best_val_mae = float('inf')
    for epoch in range(1, EPOCHS + 1):
        model.train()
        optimizer.zero_grad()
        
        state = None
        loss_total = 0
        
        # --- REASONING LOOP ---
        for step in range(REASONING_STEPS):
            pred_distance, state = model(
                data.x, data.edge_index, data.edge_attr, 
                data.cluster, data.coarse_edge_index, data.coarse_pos, 
                state=state
            )
            
            # Loss on NORMALIZED targets
            step_loss = criterion(pred_distance[data.train_mask], data.y_norm[data.train_mask])
            loss_total += step_loss
        
        loss_total.backward()
        optimizer.step()
        
        # --- EVALUATION ---
        if epoch % 5 == 0:
            model.eval()
            with torch.no_grad():
                eval_state = None
                # Run full reasoning chain
                for _ in range(REASONING_STEPS):
                    pred_dist_eval, eval_state = model(
                        data.x, data.edge_index, data.edge_attr, 
                        data.cluster, data.coarse_edge_index, data.coarse_pos, 
                        state=eval_state
                    )
                
                # Denormalize predictions to original scale
                pred_meters = pred_dist_eval * data.y_std + data.y_mean
                
                # Calculate MAE on validation set
                val_mae = criterion(pred_meters[data.val_mask], data.y_raw[data.val_mask]).item()
                if val_mae < best_val_mae:
                    model_path = f"best_model_{CITY_NAME}_h3_{H3_RES}.pt"
                    torch.save(model.state_dict(), model_path)
                    best_val_mae = val_mae

                # Calculate accuracy (binned classification)
                val_acc = compute_accuracy(pred_meters, data.y_raw, thresholds, data.val_mask)
                
                print(f"Epoch {epoch:03d} | Loss: {loss_total.item()/REASONING_STEPS:.3f} | "
                      f"Val MAE: {val_mae:.3f}m | Val Acc: {val_acc*100:.1f}%")

if __name__ == "__main__":
    main()