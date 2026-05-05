import os
import argparse
import glob
from ultralytics import YOLO
import wandb
import pandas as pd

def parse_args():
    parser = argparse.ArgumentParser(prog="eval-suite", description="Evaluate YOLO models and log to WandB.")
    parser.add_argument("--data", type=str, default="data/data.yaml", help="Path to data.yaml")
    parser.add_argument("--project", type=str, default="Object-Detection-Human-Tent", help="WandB project name")
    parser.add_argument("--weights_dir", type=str, default="runs/detect", help="Root directory to search for best.pt files")
    return parser.parse_args()

def find_models(root_dir):
    """Finds all best.pt files in the ultralytics runs directory."""
    pattern = os.path.join(root_dir, "**", "weights", "best.pt")
    models = glob.glob(pattern, recursive=True)
    return sorted(models)

def main():
    args = parse_args()
    
    # Initialize WandB for the evaluation summary
    run = wandb.init(project=args.project, job_type="evaluation", name="model-comparison")
    
    model_paths = find_models(args.weights_dir)
    
    if not model_paths:
        print(f"No models found in {args.weights_dir}. Make sure you have completed at least one training run.")
        return

    print(f"Found {len(model_paths)} models to evaluate.")
    
    results_data = []

    for path in model_paths:
        run_name = os.path.basename(os.path.dirname(os.path.dirname(path)))
        print(f"\n>>> Evaluating model: {run_name} ({path})")
        
        # Load model
        model = YOLO(path)
        
        # Run validation on TEST split
        # imgsz should match training (usually 640)
        results = model.val(data=args.data, split="test", imgsz=640, plots=True, save_json=True)
        
        # Extract metrics
        metrics = results.results_dict
        
        # results_dict typically contains:
        # metrics/precision(B), metrics/recall(B), metrics/mAP50(B), metrics/mAP50-95(B)
        row = {
            "Model Name": run_name,
            "mAP50": metrics.get("metrics/mAP50(B)", 0),
            "mAP50-95": metrics.get("metrics/mAP50-95(B)", 0),
            "Precision": metrics.get("metrics/precision(B)", 0),
            "Recall": metrics.get("metrics/recall(B)", 0),
            "Fitness": results.fitness,
        }
        results_data.append(row)
        
        # Log individual run details to WandB as artifacts or media if desired
        # Ultralytics already logs some of these if wandb is integrated, 
        # but here we are doing a dedicated comparison.
        
        # Optional: Log the confusion matrix plot if it exists
        cm_path = os.path.join(results.save_dir, "confusion_matrix.png")
        if os.path.exists(cm_path):
            wandb.log({f"Confusion_Matrix/{run_name}": wandb.Image(cm_path)})

    # Create and log comparison table
    df = pd.DataFrame(results_data)
    table = wandb.Table(dataframe=df)
    wandb.log({"Model_Comparison_Table": table})
    
    print("\nEvaluation complete. Results logged to WandB.")
    wandb.finish()

if __name__ == "__main__":
    main()
