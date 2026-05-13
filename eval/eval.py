import os
import argparse
import glob

from ultralytics import YOLO
import wandb
import pandas as pd


def parse_args():
    """
    Parse command-line arguments.

    Returns:
        argparse.Namespace: Parsed arguments object
    """

    parser = argparse.ArgumentParser(
        prog="eval-suite",
        description="Evaluate YOLO models and log results to WandB."
    )

    # Path to YOLO dataset configuration
    parser.add_argument(
        "--data",
        type=str,
        default="data/data.yaml",
        help="Path to data.yaml"
    )

    # WandB project name
    parser.add_argument(
        "--project",
        type=str,
        default="Object-Detection-Human-Tent",
        help="WandB project name"
    )

    # Root directory containing YOLO training runs
    parser.add_argument(
        "--weights_dir",
        type=str,
        default="runs/detect",
        help="Root directory to search for best.pt files"
    )

    return parser.parse_args()


def find_models(root_dir):
    """
    Find all trained YOLO model weights (best.pt)
    inside the Ultralytics runs directory.

    Example:
        runs/detect/train/weights/best.pt

    Args:
        root_dir (str): Root search directory

    Returns:
        list[str]: Sorted list of model paths
    """

    # Recursive search pattern
    pattern = os.path.join(root_dir, "**", "weights", "best.pt")

    # Search recursively
    models = glob.glob(pattern, recursive=True)

    return sorted(models)


def main():

    # Parse command-line arguments
    args = parse_args()

    # Initialize WandB experiment
    run = wandb.init(
        project=args.project,
        job_type="evaluation",
        name="model-comparison"
    )

    # Find all trained models
    model_paths = find_models(args.weights_dir)

    # Stop if no models are found
    if not model_paths:
        print(
            f"No models found in {args.weights_dir}. "
            "Make sure you have completed at least one training run."
        )
        return

    print(f"Found {len(model_paths)} models to evaluate.")

    # Store evaluation metrics for all models
    results_data = []

    # Evaluate each model
    for path in model_paths:

        # Extract training run name
        # Example:
        # runs/detect/train5/weights/best.pt -> train5
        run_name = os.path.basename(
            os.path.dirname(
                os.path.dirname(path)
            )
        )

        print(f"\n>>> Evaluating model: {run_name} ({path})")

        # Load YOLO model
        model = YOLO(path)

        # Run evaluation on test dataset
        results = model.val(
            data=args.data,
            split="test",
            imgsz=640,
            plots=True,
            save_json=True
        )

        # Extract evaluation metrics dictionary
        metrics = results.results_dict

        # Save selected metrics
        row = {
            "Model Name": run_name,

            # Mean Average Precision at IoU = 0.5
            "mAP50": metrics.get("metrics/mAP50(B)", 0),

            # Mean Average Precision averaged from 0.5 to 0.95
            "mAP50-95": metrics.get("metrics/mAP50-95(B)", 0),

            # Detection precision
            "Precision": metrics.get("metrics/precision(B)", 0),

            # Detection recall
            "Recall": metrics.get("metrics/recall(B)", 0),

            # Combined fitness metric used internally by YOLO
            "Fitness": results.fitness,
        }

        # Add model results to comparison list
        results_data.append(row)

        # Log confusion matrix image to WandB
        cm_path = os.path.join(
            results.save_dir,
            "confusion_matrix.png"
        )

        if os.path.exists(cm_path):
            wandb.log({
                f"Confusion_Matrix/{run_name}": wandb.Image(cm_path)
            })

   # Create comparison table
    df = pd.DataFrame(results_data)

    # Convert dataframe into WandB table
    table = wandb.Table(dataframe=df)

    # Log comparison table
    wandb.log({
        "Model_Comparison_Table": table
    })

    print("\nEvaluation complete. Results logged to WandB.")

    # Finish WandB session
    wandb.finish()


# Script entry point
if __name__ == "__main__":
    main()