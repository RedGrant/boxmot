import os
import shutil
import subprocess
import cv2
from pathlib import Path


def get_video_fps(video_path):
    """Get FPS from video file"""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return int(fps)


def get_model_weights(yolo_model_type="mediumaug"):
    """
    Get model weights paths - rename only if not already renamed
    """
    weights_dir = "renamed_weights"
    os.makedirs(weights_dir, exist_ok=True)

    model_names = [
        "normal_950", "normal_1200", "normal_1400",
        "binary_50_closing_KNN_950", "binary_50_closing_KNN_1200", "binary_50_closing_KNN_1400",
        "binary_250_closing_KNN_950", "binary_250_closing_KNN_1200", "binary_250_closing_KNN_1400",
        "adaptive_gaussian_11_opening_KNN_950", "adaptive_gaussian_11_opening_KNN_1200",
        "adaptive_gaussian_11_opening_KNN_1400"
    ]

    renamed_models = {}
    models_to_copy = {}

    # Check which models already exist in renamed format
    for model_name in model_names:
        new_name = f"yolov8_{model_name}_{yolo_model_type}.pt"
        new_path = os.path.join(weights_dir, new_name)

        if os.path.exists(new_path):
            renamed_models[model_name] = new_path
            print(f"Found existing renamed model: {new_path}")
        else:
            # Need to copy from original location
            original_path = f"../../yolo8{yolo_model_type}_results/{model_name}/weights/best.pt"
            models_to_copy[model_name] = (original_path, new_path)

    # Copy models that don't exist in renamed format
    if models_to_copy:
        print(f"Copying {len(models_to_copy)} models that haven't been renamed yet...")
        for model_name, (original_path, new_path) in models_to_copy.items():
            if os.path.exists(original_path):
                shutil.copy2(original_path, new_path)
                renamed_models[model_name] = new_path
                print(f"Copied {original_path} -> {new_path}")
            else:
                print(f"Warning: Model file not found: {original_path}")
    else:
        print(f"All models for {yolo_model_type} are already renamed - skipping copy step")

    return renamed_models


def run_boxmot_tracking(yolo_model_path, video_path, method, conf_level=0.25):
    """
    Run BoxMOT tracking on a single video
    """
    # Get video FPS
    fps = get_video_fps(video_path)

    # BoxMOT command (let it use default output location)
    cmd = [
        "boxmot", "track",
        "--yolo-model", yolo_model_path,
        "--source", video_path,
        "--tracking-method", method,
        "--save",
        "--save-txt",
        "--show-trajectories",
        "--fps", str(fps),
        "--conf", str(conf_level)
    ]

    try:
        print(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print(f"Success: {video_path} with {method}")

        # BoxMOT saves to runs/track by default
        default_output = "runs/track"
        return default_output if os.path.exists(default_output) else None

    except subprocess.CalledProcessError as e:
        print(f"Error processing {video_path} with {method}: {e}")
        print(f"stdout: {e.stdout}")
        print(f"stderr: {e.stderr}")
        return None


def organize_results(boxmot_output_dir, final_output_dir, model_name, method, video_filename):
    """
    Organize results from BoxMOT default output to structured folders
    """
    if not os.path.exists(boxmot_output_dir):
        print(f"Warning: BoxMOT output directory not found: {boxmot_output_dir}")
        return False

    # Final output structure: boxmot_results/{model_name}/{method}/{video_name}/
    video_name = os.path.splitext(video_filename)[0]  # Remove .mp4 extension
    final_dir = os.path.join(final_output_dir, model_name, method, video_name)
    os.makedirs(final_dir, exist_ok=True)

    # Copy video file (if exists)
    for item in os.listdir(boxmot_output_dir):
        if item.endswith(('.avi', '.mp4', '.mov')):
            src_path = os.path.join(boxmot_output_dir, item)
            dst_path = os.path.join(final_dir, item)
            shutil.copy2(src_path, dst_path)
            print(f"Copied video: {src_path} -> {dst_path}")

    # Copy labels directory
    labels_src = os.path.join(boxmot_output_dir, "labels")
    if os.path.exists(labels_src):
        labels_dst = os.path.join(final_dir, "labels")
        if os.path.exists(labels_dst):
            shutil.rmtree(labels_dst)
        shutil.copytree(labels_src, labels_dst)
        print(f"Copied labels: {labels_src} -> {labels_dst}")

    print(f"Results organized in: {final_dir}")
    return True


def clean_boxmot_runs():
    """
    Clean the runs directory after processing
    """
    runs_dir = "runs"
    if os.path.exists(runs_dir):
        shutil.rmtree(runs_dir)
        print("Cleaned BoxMOT runs directory")


def main():
    # Configuration
    yolo_model_types = ["medium", "mediumaug", "nano"]  # Add your model types
    video_types = ["normal", "adaptive_gaussian_11_opening_KNN", "binary_50_closing_KNN", "binary_250_closing_KNN"]
    frequencies = ["950", "1200", "1400"]
    tracking_methods = ["deepocsort", "strongsort", "ocsort", "bytetrack", "botsort", "boosttrack"]
    conf_level = 0.6  # Confidence threshold

    class_names = ['Aluminum_tube', 'Fish_cage', 'Fish_net', 'Floating_floor_wood', 'Plastic_Deck', 'PVC_cone',
                   'PVC_perforated_deck', 'PVC_Square', 'Vinyl', 'Wooden_deck', 'PVC_Blue_Square', 'rope']

    # Main processing loop
    for yolo_model_type in yolo_model_types:
        print(f"\n=== Processing YOLO model type: {yolo_model_type} ===")

        # Get model weights (rename only if needed)
        renamed_models = get_model_weights(yolo_model_type)

        if not renamed_models:
            print(f"No models found for {yolo_model_type}, skipping...")
            continue

        # Process each model
        for model_name, model_path in renamed_models.items():
            print(f"\n--- Processing model: {model_name} ---")

            # Extract video type and frequency from model name
            parts = model_name.split('_')
            frequency = parts[-1]  # Last part is frequency
            video_type = '_'.join(parts[:-1])  # Everything except last part

            # Input video directory
            input_video_dir = f"../../input_videos/{video_type}"

            if not os.path.exists(input_video_dir):
                print(f"Video directory not found: {input_video_dir}")
                continue

            # Process each video in the directory
            for filename in os.listdir(input_video_dir):
                if not filename.endswith(".mp4"):
                    continue

                # Check if video matches the frequency
                video_file, video_freq = filename.rsplit("_", 1)
                video_freq = video_freq.split(".")[0]

                if video_freq != frequency:
                    continue

                video_path = os.path.join(input_video_dir, filename)
                print(f"\n  Processing video: {filename}")

                # Run tracking with each method
                for method in tracking_methods:
                    print(f"    Tracking method: {method}")

                    # Run BoxMOT (it will output to runs/track by default)
                    boxmot_output_dir = run_boxmot_tracking(
                        yolo_model_path=model_path,
                        video_path=video_path,
                        method=method,
                        conf_level=conf_level
                    )

                    if boxmot_output_dir:
                        # Organize results
                        final_output_dir = "/media/pedroguedes/GUEDES_PHD/box_mot_new_results"
                        model_full_name = f"{yolo_model_type}_{model_name}"

                        organize_results(
                            boxmot_output_dir=boxmot_output_dir,
                            final_output_dir=final_output_dir,
                            model_name=model_full_name,
                            method=method,
                            video_filename=filename
                        )

                        # Clean up BoxMOT's default runs directory after organizing
                        clean_boxmot_runs()

    print("\n=== Processing completed ===")


if __name__ == "__main__":
    main()