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


def rename_model_weights(yolo_model_type="mediumaug"):
    """
    Rename best.pt files to yolov8_{model_name}.pt format
    """
    models = {
        "normal_950": f"../yolo8{yolo_model_type}_results/normal_950/weights/best.pt",
        "normal_1200": f"../yolo8{yolo_model_type}_results/normal_1200/weights/best.pt",
        "normal_1400": f"../yolo8{yolo_model_type}_results/normal_1400/weights/best.pt",
        "binary_50_closing_KNN_950": f"../yolo8{yolo_model_type}_results/binary_50_closing_KNN_950/weights/best.pt",
        "binary_50_closing_KNN_1200": f"../yolo8{yolo_model_type}_results/binary_50_closing_KNN_1200/weights/best.pt",
        "binary_50_closing_KNN_1400": f"../yolo8{yolo_model_type}_results/binary_50_closing_KNN_1400/weights/best.pt",
        "binary_250_closing_KNN_950": f"../yolo8{yolo_model_type}_results/binary_250_closing_KNN_950/weights/best.pt",
        "binary_250_closing_KNN_1200": f"../yolo8{yolo_model_type}_results/binary_250_closing_KNN_1200/weights/best.pt",
        "binary_250_closing_KNN_1400": f"../yolo8{yolo_model_type}_results/binary_250_closing_KNN_1400/weights/best.pt",
        "adaptive_gaussian_11_opening_KNN_950": f"../yolo8{yolo_model_type}_results/adaptive_gaussian_11_opening_KNN_950/weights/best.pt",
        "adaptive_gaussian_11_opening_KNN_1200": f"../yolo8{yolo_model_type}_results/adaptive_gaussian_11_opening_KNN_1200/weights/best.pt",
        "adaptive_gaussian_11_opening_KNN_1400": f"../yolo8{yolo_model_type}_results/adaptive_gaussian_11_opening_KNN_1400/weights/best.pt"
    }

    renamed_models = {}
    weights_dir = "renamed_weights"
    os.makedirs(weights_dir, exist_ok=True)

    for model_name, model_path in models.items():
        if os.path.exists(model_path):
            new_name = f"yolov8_{model_name}_{yolo_model_type}.pt"
            new_path = os.path.join(weights_dir, new_name)
            shutil.copy2(model_path, new_path)
            renamed_models[model_name] = new_path
            print(f"Copied {model_path} -> {new_path}")
        else:
            print(f"Warning: Model file not found: {model_path}")

    return renamed_models


def run_boxmot_tracking(yolo_model_path, video_path, method, output_dir, conf_level=0.25):
    """
    Run BoxMOT tracking on a single video
    """
    # Get video FPS
    fps = get_video_fps(video_path)

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # BoxMOT command
    cmd = [
        "boxmot", "track",
        "--yolo-model", yolo_model_path,
        "--source", video_path,
        "--tracking-method", method,
        "--save",
        "--save-txt",
        "--fps", str(fps),
        "--conf", str(conf_level),
        "--project", output_dir,
        "--name", "track"
    ]

    try:
        print(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print(f"Success: {video_path} with {method}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error processing {video_path} with {method}: {e}")
        print(f"stdout: {e.stdout}")
        print(f"stderr: {e.stderr}")
        return False


def organize_results(temp_output_dir, final_output_dir, model_name, video_type, frequency, method):
    """
    Organize results into the desired folder structure
    """
    # Source directories
    track_dir = os.path.join(temp_output_dir, "track")

    if not os.path.exists(track_dir):
        print(f"Warning: Track directory not found: {track_dir}")
        return

    # Final output structure: results/{yolo_model_type}/{video_type}_{frequency}/{method}/
    final_dir = os.path.join(final_output_dir, model_name, method)
    os.makedirs(final_dir, exist_ok=True)

    # Move video and labels
    for item in os.listdir(track_dir):
        src_path = os.path.join(track_dir, item)
        dst_path = os.path.join(final_dir, item)

        if os.path.isdir(src_path):
            if os.path.exists(dst_path):
                shutil.rmtree(dst_path)
            shutil.copytree(src_path, dst_path)
        else:
            shutil.copy2(src_path, dst_path)

    print(f"Results organized in: {final_dir}")


def main():
    # Configuration
    yolo_model_types = ["medium", "mediumaug", "nano"]  # Add your model types
    video_types = ["normal", "adaptive_gaussian_11_opening_KNN", "binary_50_closing_KNN", "binary_250_closing_KNN"]
    frequencies = ["950", "1200", "1400"]
    tracking_methods = ["deepocsort", "strongsort", "ocsort", "bytetrack", "botsort", "boosttrack"]

    conf_level = 0.25  # Confidence threshold

    class_names = ['Aluminum_tube', 'Fish_cage', 'Fish_net', 'Floating_floor_wood', 'Plastic_Deck', 'PVC_cone',
                   'PVC_perforated_deck', 'PVC_Square', 'Vinyl', 'Wooden_deck', 'PVC_Blue_Square', 'rope']

    # Main processing loop
    for yolo_model_type in yolo_model_types:
        print(f"\n=== Processing YOLO model type: {yolo_model_type} ===")

        # Rename model weights
        renamed_models = rename_model_weights(yolo_model_type)

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
            input_video_dir = f"../input_videos/{video_type}"

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

                    # Temporary output directory for BoxMOT
                    temp_output_dir = f"temp_boxmot_output_{model_name}_{method}"

                    # Run BoxMOT
                    success = run_boxmot_tracking(
                        yolo_model_path=model_path,
                        video_path=video_path,
                        method=method,
                        output_dir=temp_output_dir,
                        conf_level=conf_level
                    )

                    if success:
                        # Organize results
                        final_output_dir = "boxmot_results"
                        organize_results(
                            temp_output_dir=temp_output_dir,
                            final_output_dir=final_output_dir,
                            model_name=f"{yolo_model_type}_{video_type}_{frequency}",
                            video_type=video_type,
                            frequency=frequency,
                            method=method
                        )

                    # Clean up temporary directory
                    if os.path.exists(temp_output_dir):
                        shutil.rmtree(temp_output_dir)

    print("\n=== Processing completed ===")


if __name__ == "__main__":
    main()