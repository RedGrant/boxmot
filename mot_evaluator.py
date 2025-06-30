import os
import numpy as np
import json
import argparse
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
import pandas as pd
from scipy.optimize import linear_sum_assignment

class MOTEvaluator:
    def __init__(self, iou_threshold: float = 0.5):
        self.iou_threshold = iou_threshold
        self.reset()

    def reset(self):
        """Reset evaluator state"""
        self.frames = {}
        self.gt_ids = set()
        self.pred_ids = set()
        self.id_switches = 0
        self.fragmentations = 0

    def add_frame(self, frame_id: int, gt_boxes: List[Tuple], pred_boxes: List[Tuple]):
        self.frames[frame_id] = {
            'gt': {int(box[0]): box[1:5] for box in gt_boxes},
            'pred': {int(box[0]): box[1:5] for box in pred_boxes}
        }

        self.gt_ids.update([int(box[0]) for box in gt_boxes])
        self.pred_ids.update([int(box[0]) for box in pred_boxes])

    def compute_iou(self, box1: Tuple, box2: Tuple) -> float:
        """Compute IoU between two boxes in (x, y, w, h) format"""
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2

        # Calculate intersection
        xi1, yi1 = max(x1, x2), max(y1, y2)
        xi2, yi2 = min(x1 + w1, x2 + w2), min(y1 + h1, y2 + h2)

        if xi2 <= xi1 or yi2 <= yi1:
            return 0.0

        intersection = (xi2 - xi1) * (yi2 - yi1)
        union = w1 * h1 + w2 * h2 - intersection

        return intersection / union if union > 0 else 0.0

    def hungarian_assignment(self, iou_matrix: np.ndarray) -> List[Tuple]:
        if iou_matrix.size == 0:
            return []
        cost_matrix = 1 - iou_matrix

        cost_matrix[iou_matrix < self.iou_threshold] = 1.0

        try:
            row_indices, col_indices = linear_sum_assignment(cost_matrix)
            matches = []

            for i, j in zip(row_indices, col_indices):
                if iou_matrix[i, j] >= self.iou_threshold:
                    matches.append((i, j, iou_matrix[i, j]))

            return matches
        except:
            # Fallback to greedy assignment if Hungarian fails
            return self.greedy_assignment(iou_matrix)

    def greedy_assignment(self, iou_matrix: np.ndarray) -> List[Tuple]:
        """Greedy assignment based on IoU"""
        matches = []
        used_gt, used_pred = set(), set()

        # Create list of (iou, gt_idx, pred_idx) and sort by IoU
        candidates = []
        for i in range(iou_matrix.shape[0]):
            for j in range(iou_matrix.shape[1]):
                if iou_matrix[i, j] >= self.iou_threshold:
                    candidates.append((iou_matrix[i, j], i, j))

        candidates.sort(reverse=True)

        for iou, i, j in candidates:
            if i not in used_gt and j not in used_pred:
                matches.append((i, j, iou))
                used_gt.add(i)
                used_pred.add(j)

        return matches

    def compute_metrics(self) -> Dict[str, float]:
        """Compute MOTA, MOTP, IDF1, and other tracking metrics"""
        tp_total = fp_total = fn_total = 0
        motp_sum = motp_count = 0
        id_switches = 0

        # Track ID associations for IDF1 and ID switches
        gt_to_pred_associations = defaultdict(list)
        pred_to_gt_associations = defaultdict(list)
        gt_trajectories = defaultdict(list)  # For tracking ID switches

        frame_ids = sorted(self.frames.keys())

        for frame_id in frame_ids:
            gt_boxes = self.frames[frame_id]['gt']
            pred_boxes = self.frames[frame_id]['pred']

            gt_ids = list(gt_boxes.keys())
            pred_ids = list(pred_boxes.keys())

            if not gt_ids and not pred_ids:
                continue
            elif not gt_ids:
                fp_total += len(pred_ids)
                continue
            elif not pred_ids:
                fn_total += len(gt_ids)
                continue

            # Compute IoU matrix
            iou_matrix = np.zeros((len(gt_ids), len(pred_ids)))
            for i, gt_id in enumerate(gt_ids):
                for j, pred_id in enumerate(pred_ids):
                    iou_matrix[i, j] = self.compute_iou(gt_boxes[gt_id], pred_boxes[pred_id])

            # Find matches
            matches = self.hungarian_assignment(iou_matrix)

            # Update counters
            tp_frame = len(matches)
            fp_frame = len(pred_ids) - tp_frame
            fn_frame = len(gt_ids) - tp_frame

            tp_total += tp_frame
            fp_total += fp_frame
            fn_total += fn_frame

            # Update MOTP
            if matches:
                motp_sum += sum(1 - iou for _, _, iou in matches)
                motp_count += len(matches)

            # Track ID associations for IDF1 and ID switches
            current_gt_to_pred = {}
            for i, j, iou in matches:
                gt_id, pred_id = gt_ids[i], pred_ids[j]
                gt_to_pred_associations[gt_id].append(pred_id)
                pred_to_gt_associations[pred_id].append(gt_id)
                current_gt_to_pred[gt_id] = pred_id
                gt_trajectories[gt_id].append((frame_id, pred_id))

        # Calculate ID switches
        for gt_id, trajectory in gt_trajectories.items():
            if len(trajectory) > 1:
                for i in range(1, len(trajectory)):
                    if trajectory[i][1] != trajectory[i - 1][1]:
                        id_switches += 1

        # Calculate basic metrics
        total_gt = tp_total + fn_total
        mota = 1 - (fn_total + fp_total + id_switches) / total_gt if total_gt > 0 else 0.0
        motp = motp_sum / motp_count if motp_count > 0 else 0.0
        precision = tp_total / (tp_total + fp_total) if (tp_total + fp_total) > 0 else 0.0
        recall = tp_total / (tp_total + fn_total) if (tp_total + fn_total) > 0 else 0.0

        # Calculate IDF1
        idfp = idfn = idtp = 0

        # Count identity matches
        for gt_id, pred_ids in gt_to_pred_associations.items():
            if pred_ids:
                # Find the most frequent prediction ID for this ground truth ID
                most_common_pred = max(set(pred_ids), key=pred_ids.count)
                idtp += pred_ids.count(most_common_pred)
                idfp += len(pred_ids) - pred_ids.count(most_common_pred)
            else:
                idfn += 1

        # Count unmatched predictions
        for pred_id, gt_ids in pred_to_gt_associations.items():
            if not gt_ids:
                idfp += 1

        # Add unmatched ground truth IDs
        for gt_id in self.gt_ids:
            if gt_id not in gt_to_pred_associations:
                idfn += 1

        idf1_precision = idtp / (idtp + idfp) if (idtp + idfp) > 0 else 0.0
        idf1_recall = idtp / (idtp + idfn) if (idtp + idfn) > 0 else 0.0
        idf1 = 2 * idf1_precision * idf1_recall / (idf1_precision + idf1_recall) if (
                                                                                            idf1_precision + idf1_recall) > 0 else 0.0

        return {
            'MOTA': mota,
            'MOTP': motp,
            'IDF1': idf1,
            'Precision': precision,
            'Recall': recall,
            'TP': tp_total,
            'FP': fp_total,
            'FN': fn_total,
            'ID_Switches': id_switches,
            'Fragmentations': len([traj for traj in gt_trajectories.values() if len(traj) > 1]),
            'Mostly_Tracked': len([gt_id for gt_id in self.gt_ids if len(gt_to_pred_associations[gt_id]) > 0]),
            'Partly_Tracked': 0,  # Would need more complex logic
            'Mostly_Lost': len([gt_id for gt_id in self.gt_ids if len(gt_to_pred_associations[gt_id]) == 0])
        }


def load_yolo_annotations(file_path: str) -> List[Tuple]:
    """Load YOLO format annotations: class_id x_center y_center width height track_id"""
    annotations = []
    if not os.path.exists(file_path):
        return annotations

    try:
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) >= 6:
                    class_id, x, y, w, h, track_id = parts[:6]
                    annotations.append((int(track_id), float(x), float(y), float(w), float(h)))
    except Exception as e:
        print(f"Error reading {file_path}: {e}")

    return annotations


def convert_to_absolute(annotations: List[Tuple], img_w: int, img_h: int) -> List[Tuple]:
    """Convert normalized YOLO coordinates to absolute pixel coordinates"""
    converted = []
    for track_id, x_norm, y_norm, w_norm, h_norm in annotations:
        # Convert to absolute coordinates
        x_abs = x_norm * img_w
        y_abs = y_norm * img_h
        w_abs = w_norm * img_w
        h_abs = h_norm * img_h

        # Convert center to top-left corner
        x_tl = x_abs - w_abs / 2
        y_tl = y_abs - h_abs / 2

        converted.append((track_id, x_tl, y_tl, w_abs, h_abs))

    return converted


def extract_frame_number_from_gt(filename: str) -> int:
    """Extract frame number from GT filename like '0_Aluminum_tube_1200.txt'"""
    try:
        # Split by underscore and get the first part before the sequence name
        parts = filename.replace('.txt', '').split('_')
        return int(parts[0])
    except:
        return 0


def extract_frame_number_from_pred(filename: str) -> int:
    """Extract frame number from prediction filename like 'Aluminum_tube_1200_3.txt'"""
    try:
        # Split by underscore and get the last part before .txt
        parts = filename.replace('.txt', '').split('_')
        return int(parts[-1])
    except:
        return 0


def extract_sequence_name_from_filename(filename: str, is_gt: bool = False) -> str:
    try:
        if is_gt:
            # GT: '0_Aluminum_tube_1200.txt' -> 'Aluminum_tube_1200'
            parts = filename.replace('.txt', '').split('_')
            return '_'.join(parts[1:])  # Skip the first part (frame number)
        else:
            # Pred: 'Aluminum_tube_1200_3.txt' -> 'Aluminum_tube_1200'
            parts = filename.replace('.txt', '').split('_')
            return '_'.join(parts[:-1])  # Skip the last part (frame number)
    except:
        return filename.replace('.txt', '')


def create_frame_mapping(gt_files: List[str], pred_files: List[str]) -> Dict[str, List[Tuple[str, str]]]:
    gt_by_sequence = defaultdict(list)
    for gt_file in gt_files:
        if gt_file.endswith('.txt'):
            sequence_name = extract_sequence_name_from_filename(gt_file, is_gt=True)
            frame_id = extract_frame_number_from_gt(gt_file)
            gt_by_sequence[sequence_name].append((gt_file, frame_id))

    # Group prediction files by sequence
    pred_by_sequence = defaultdict(list)
    for pred_file in pred_files:
        if pred_file.endswith('.txt'):
            sequence_name = extract_sequence_name_from_filename(pred_file, is_gt=False)
            frame_id = extract_frame_number_from_pred(pred_file)
            pred_by_sequence[sequence_name].append((pred_file, frame_id))

    # Create mappings for common sequences
    frame_mappings = {}
    for sequence_name in gt_by_sequence:
        if sequence_name in pred_by_sequence:
            # Create frame ID to filename mappings
            gt_frame_map = {fid: fname for fname, fid in gt_by_sequence[sequence_name]}
            pred_frame_map = {fid: fname for fname, fid in pred_by_sequence[sequence_name]}

            # Find common frame IDs
            common_frame_ids = set(gt_frame_map.keys()) & set(pred_frame_map.keys())

            if common_frame_ids:
                frame_mappings[sequence_name] = [
                    (gt_frame_map[fid], pred_frame_map[fid], fid)
                    for fid in sorted(common_frame_ids)
                ]

    return frame_mappings


def evaluate_sequence(pred_sequence_path: str, gt_sequence_path: str, img_size: Tuple[int, int] = (1920, 1080)) -> Dict[
    str, float]:
    """Evaluate a single tracking sequence with improved filename matching"""
    evaluator = MOTEvaluator()

    pred_labels = os.path.join(pred_sequence_path, "labels")

    # Ground truth labels are directly in the gt_sequence_path
    gt_labels = gt_sequence_path
    if os.path.exists(os.path.join(gt_sequence_path, "labels")):
        gt_labels = os.path.join(gt_sequence_path, "labels")

    if not os.path.exists(pred_labels):
        return {"error": f"Prediction labels directory not found: {pred_labels}"}

    if not os.path.exists(gt_labels):
        return {"error": f"Ground truth labels directory not found: {gt_labels}"}

    # Get all files
    try:
        pred_files = [f for f in os.listdir(pred_labels) if f.endswith('.txt')]
        gt_files = [f for f in os.listdir(gt_labels) if f.endswith('.txt')]
    except Exception as e:
        return {"error": f"Error reading directory contents: {e}"}

    if not pred_files or not gt_files:
        return {"error": "No annotation files found"}

    # Create frame mappings
    frame_mappings = create_frame_mapping(gt_files, pred_files)

    if not frame_mappings:
        return {"error": "No matching sequences found between GT and predictions"}

    img_w, img_h = img_size
    total_frames_processed = 0

    # Process all sequences found in the mappings
    for sequence_name, file_pairs in frame_mappings.items():
        print(f"      Processing sequence: {sequence_name} ({len(file_pairs)} frames)")

        for gt_file, pred_file, frame_id in file_pairs:
            gt_path = os.path.join(gt_labels, gt_file)
            pred_path = os.path.join(pred_labels, pred_file)

            # Load and convert annotations
            gt_annotations = load_yolo_annotations(gt_path)
            pred_annotations = load_yolo_annotations(pred_path)

            gt_abs = convert_to_absolute(gt_annotations, img_w, img_h)
            pred_abs = convert_to_absolute(pred_annotations, img_w, img_h)

            evaluator.add_frame(frame_id, gt_abs, pred_abs)
            total_frames_processed += 1

    if total_frames_processed == 0:
        return {"error": "No frames were processed"}

    metrics = evaluator.compute_metrics()
    metrics['frames_processed'] = total_frames_processed

    return metrics


def find_ground_truth_sequence(gt_base_dir: str, sequence_name: str) -> Optional[str]:
    direct_path = os.path.join(gt_base_dir, sequence_name)
    if os.path.exists(direct_path):
        return direct_path

    # Try to find by pattern matching if direct match fails
    if os.path.exists(gt_base_dir):
        for gt_dir in os.listdir(gt_base_dir):
            gt_path = os.path.join(gt_base_dir, gt_dir)
            if os.path.isdir(gt_path) and gt_dir == sequence_name:
                return gt_path

    return None


def run_evaluation(results_dir: str, gt_dir: str, output_file: str = "evaluation_results.json",
                   img_width: int = 1920, img_height: int = 1080):
    """Run complete evaluation on all sequences"""
    all_results = {}
    all_metrics = defaultdict(list)

    # Track statistics
    total_evaluations = 0
    successful_evaluations = 0
    failed_evaluations = 0

    print(f"Evaluating results from: {results_dir}")
    print(f"Ground truth from: {gt_dir}")
    print(f"Image size: {img_width}x{img_height}")

    # Walk through results directory structure: model/method/sequence
    for model_name in sorted(os.listdir(results_dir)):
        model_path = os.path.join(results_dir, model_name)
        if not os.path.isdir(model_path):
            continue

        print(f"\n{'=' * 60}")
        print(f"Model: {model_name}")
        print(f"{'=' * 60}")
        all_results[model_name] = {}

        for method_name in sorted(os.listdir(model_path)):
            method_path = os.path.join(model_path, method_name)
            if not os.path.isdir(method_path):
                continue

            print(f"\n  Method: {method_name}")
            print(f"  {'-' * 50}")
            all_results[model_name][method_name] = {}

            for sequence_name in sorted(os.listdir(method_path)):
                sequence_path = os.path.join(method_path, sequence_name)
                if not os.path.isdir(sequence_path):
                    continue

                # Find matching ground truth sequence
                gt_sequence_path = find_ground_truth_sequence(gt_dir, sequence_name)

                if not gt_sequence_path:
                    print(f"    {sequence_name:<30} - Warning: No GT found")
                    all_results[model_name][method_name][sequence_name] = {"error": "No ground truth found"}
                    failed_evaluations += 1
                    continue

                total_evaluations += 1
                print(f"    {sequence_name:<30} - ", end="")

                try:
                    metrics = evaluate_sequence(sequence_path, gt_sequence_path, (img_width, img_height))

                    if "error" in metrics:
                        print(f"Error: {metrics['error']}")
                        failed_evaluations += 1
                        all_results[model_name][method_name][sequence_name] = metrics
                    else:
                        all_results[model_name][method_name][sequence_name] = metrics
                        successful_evaluations += 1

                        # Collect metrics for summary
                        for metric_name, value in metrics.items():
                            if isinstance(value, (int, float)) and metric_name not in ['error', 'frames_processed']:
                                all_metrics[f"{model_name}_{method_name}_{metric_name}"].append(value)
                                all_metrics[f"overall_{metric_name}"].append(value)

                        print(f"MOTA: {metrics.get('MOTA', 0):.3f}, "
                              f"IDF1: {metrics.get('IDF1', 0):.3f}, "
                              f"MOTP: {metrics.get('MOTP', 0):.3f}, "
                              f"Frames: {metrics.get('frames_processed', 0)}")

                except Exception as e:
                    print(f"Error: {str(e)}")
                    all_results[model_name][method_name][sequence_name] = {"error": str(e)}
                    failed_evaluations += 1

    # Compute summary statistics
    summary_stats = {}
    for metric_name, values in all_metrics.items():
        if values:
            summary_stats[metric_name] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'min': float(np.min(values)),
                'max': float(np.max(values)),
                'median': float(np.median(values)),
                'count': len(values)
            }

    # Create performance summary by model and method
    model_method_summary = {}
    for model_name in all_results:
        model_method_summary[model_name] = {}
        for method_name in all_results[model_name]:
            valid_results = [res for res in all_results[model_name][method_name].values()
                             if isinstance(res, dict) and 'error' not in res]

            if valid_results:
                model_method_summary[model_name][method_name] = {
                    'count': len(valid_results),
                    'total_frames': sum(r.get('frames_processed', 0) for r in valid_results),
                    'avg_MOTA': float(np.mean([r['MOTA'] for r in valid_results])),
                    'avg_IDF1': float(np.mean([r['IDF1'] for r in valid_results])),
                    'avg_MOTP': float(np.mean([r['MOTP'] for r in valid_results])),
                    'avg_Precision': float(np.mean([r['Precision'] for r in valid_results])),
                    'avg_Recall': float(np.mean([r['Recall'] for r in valid_results])),
                }

    # Save results
    final_results = {
        'detailed_results': all_results,
        'summary_statistics': summary_stats,
        'model_method_summary': model_method_summary,
        'evaluation_info': {
            'total_evaluations': total_evaluations,
            'successful_evaluations': successful_evaluations,
            'failed_evaluations': failed_evaluations,
            'success_rate': successful_evaluations / total_evaluations if total_evaluations > 0 else 0,
            'image_size': [img_width, img_height],
            'results_directory': results_dir,
            'ground_truth_directory': gt_dir
        }
    }

    with open(output_file, 'w') as f:
        json.dump(final_results, f, indent=2)

    # Print comprehensive summary
    print(f"\n{'=' * 80}")
    print("EVALUATION SUMMARY")
    print(f"{'=' * 80}")
    print(f"Results saved to: {output_file}")
    print(f"Total evaluations: {total_evaluations}")
    print(f"Successful: {successful_evaluations}")
    print(f"Failed: {failed_evaluations}")
    print(f"Success rate: {successful_evaluations / total_evaluations * 100:.1f}%" if total_evaluations > 0 else "N/A")

    # Print overall metrics
    print(f"\n{'Overall Performance Metrics':<40}")
    print(f"{'-' * 60}")
    key_metrics = ['overall_MOTA', 'overall_IDF1', 'overall_MOTP', 'overall_Precision', 'overall_Recall']
    for metric in key_metrics:
        if metric in summary_stats:
            stats = summary_stats[metric]
            metric_name = metric.replace('overall_', '')
            print(f"{metric_name:<15}: {stats['mean']:.3f} ± {stats['std']:.3f} "
                  f"(min: {stats['min']:.3f}, max: {stats['max']:.3f}, n={stats['count']})")

    # Print top performing combinations
    print(f"\n{'Top Performing Model-Method Combinations (by MOTA)':<60}")
    print(f"{'-' * 80}")
    top_combinations = []
    for model in model_method_summary:
        for method in model_method_summary[model]:
            combo_data = model_method_summary[model][method]
            top_combinations.append((f"{model}_{method}", combo_data['avg_MOTA'],
                                     combo_data['avg_IDF1'], combo_data['count'], combo_data['total_frames']))

    top_combinations.sort(key=lambda x: x[1], reverse=True)

    print(f"{'Model_Method':<50} {'MOTA':<8} {'IDF1':<8} {'Seqs':<6} {'Frames':<8}")
    print(f"{'-' * 80}")
    for combo, mota, idf1, count, frames in top_combinations[:10]:  # Top 10
        print(f"{combo:<50} {mota:.3f}    {idf1:.3f}    {count:<6} {frames:<8}")

    return final_results


def create_performance_report(results_file: str, output_csv: str = "performance_report.csv"):
    """Create a CSV report from evaluation results"""
    with open(results_file, 'r') as f:
        data = json.load(f)

    rows = []
    for model in data['detailed_results']:
        for method in data['detailed_results'][model]:
            for sequence in data['detailed_results'][model][method]:
                result = data['detailed_results'][model][method][sequence]
                if isinstance(result, dict) and 'error' not in result:
                    row = {
                        'Model': model,
                        'Method': method,
                        'Sequence': sequence,
                        **result
                    }
                    rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(output_csv, index=False)
    print(f"Performance report saved to: {output_csv}")
    return df


def main():
    parser = argparse.ArgumentParser(description='Multi-Object Tracking Evaluation for BoxMOT Results')
    parser.add_argument('--results_dir', type=str, default='boxmot_results',
                        help='Directory containing tracking results (default: boxmot_results)')
    parser.add_argument('--gt_dir', type=str, default='video_annotations',
                        help='Directory containing ground truth annotations (default: video_annotations)')
    parser.add_argument('--output', type=str, default='evaluation_results.json',
                        help='Output JSON file for results')
    parser.add_argument('--img_width', type=int, default=450,
                        help='Image width for coordinate conversion')
    parser.add_argument('--img_height', type=int, default=4002,
                        help='Image height for coordinate conversion')
    parser.add_argument('--create_csv', action='store_true',
                        help='Also create a CSV performance report')

    args = parser.parse_args()

    # Validate input directories
    if not os.path.exists(args.results_dir):
        print(f"Error: Results directory not found: {args.results_dir}")
        return

    if not os.path.exists(args.gt_dir):
        print(f"Error: Ground truth directory not found: {args.gt_dir}")
        return

    # Run evaluation
    results = run_evaluation(args.results_dir, args.gt_dir, args.output,
                             args.img_width, args.img_height)

    # Create CSV report if requested
    if args.create_csv:
        create_performance_report(args.output, args.output.replace('.json', '.csv'))

    print(f"\nEvaluation completed successfully!")
    print(f"Detailed results: {args.output}")
    if args.create_csv:
        print(f"CSV report: {args.output.replace('.json', '.csv')}")


if __name__ == "__main__":
    main()