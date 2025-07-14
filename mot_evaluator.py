import os
import numpy as np
import json
import argparse
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
import pandas as pd
from scipy.optimize import linear_sum_assignment
import matplotlib.pyplot as plt
import seaborn as sns
from math import pi
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

        gt_to_pred_associations = defaultdict(list)
        pred_to_gt_associations = defaultdict(list)
        gt_trajectories = defaultdict(list)

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

            # Hungarian assignment
            matches = self.hungarian_assignment(iou_matrix)

            tp_frame = len(matches)
            fp_frame = len(pred_ids) - tp_frame
            fn_frame = len(gt_ids) - tp_frame

            tp_total += tp_frame
            fp_total += fp_frame
            fn_total += fn_frame

            # MOTP: distance-based
            if matches:
                motp_sum += sum(1 - iou for _, _, iou in matches)
                motp_count += len(matches)

            current_gt_to_pred = {}
            for i, j, iou in matches:
                gt_id, pred_id = gt_ids[i], pred_ids[j]
                gt_to_pred_associations[gt_id].append(pred_id)
                pred_to_gt_associations[pred_id].append(gt_id)
                current_gt_to_pred[gt_id] = pred_id
                gt_trajectories[gt_id].append((frame_id, pred_id))

        # Count ID switches
        for gt_id, trajectory in gt_trajectories.items():
            for i in range(1, len(trajectory)):
                if trajectory[i][1] != trajectory[i - 1][1]:
                    id_switches += 1

        total_gt = tp_total + fn_total
        mota = 1 - (fn_total + fp_total + id_switches) / total_gt if total_gt > 0 else 0.0
        motp = motp_sum / motp_count if motp_count > 0 else 0.0
        precision = tp_total / (tp_total + fp_total) if (tp_total + fp_total) > 0 else 0.0
        recall = tp_total / (tp_total + fn_total) if (tp_total + fn_total) > 0 else 0.0

        # IDF1 calculation
        idfp = idfn = idtp = 0

        for gt_id, pred_ids in gt_to_pred_associations.items():
            if pred_ids:
                most_common_pred = max(set(pred_ids), key=pred_ids.count)
                idtp += pred_ids.count(most_common_pred)
                idfp += len(pred_ids) - pred_ids.count(most_common_pred)
            else:
                idfn += 1

        for pred_id, gt_ids in pred_to_gt_associations.items():
            if not gt_ids:
                idfp += 1

        for gt_id in self.gt_ids:
            if gt_id not in gt_to_pred_associations:
                idfn += 1

        idf1_precision = idtp / (idtp + idfp) if (idtp + idfp) > 0 else 0.0
        idf1_recall = idtp / (idtp + idfn) if (idtp + idfn) > 0 else 0.0
        idf1 = 2 * idf1_precision * idf1_recall / (idf1_precision + idf1_recall) if (
                                                                                            idf1_precision + idf1_recall) > 0 else 0.0

        # Mostly Tracked / Partly Tracked / Mostly Lost
        mostly_tracked = partly_tracked = mostly_lost = 0

        for gt_id in self.gt_ids:
            total_frames = sum(1 for frame_id in frame_ids if gt_id in self.frames[frame_id]['gt'])
            matched_frames = sum(1 for frame_id in frame_ids
                                 if gt_id in self.frames[frame_id]['gt']
                                 and any(gt_ids[i] == gt_id for i, _, _ in self.hungarian_assignment(
                np.array([[self.compute_iou(self.frames[frame_id]['gt'][gt_id],
                                            self.frames[frame_id]['pred'][pred_id])]
                          for pred_id in self.frames[frame_id]['pred'].keys()])
            )))

            if total_frames == 0:
                continue

            match_ratio = matched_frames / total_frames
            if match_ratio >= 0.8:
                mostly_tracked += 1
            elif match_ratio >= 0.2:
                partly_tracked += 1
            else:
                mostly_lost += 1

        # Fragmentations
        fragmentations = 0
        for gt_id, traj in gt_trajectories.items():
            if len(traj) < 2:
                continue
            prev_frame = traj[0][0]
            frags = 0
            for i in range(1, len(traj)):
                curr_frame = traj[i][0]
                if curr_frame - prev_frame > 1:
                    frags += 1
                prev_frame = curr_frame
            fragmentations += frags

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
            'Fragmentations': fragmentations,
            'Mostly_Tracked': mostly_tracked,
            'Partly_Tracked': partly_tracked,
            'Mostly_Lost': mostly_lost
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


def evaluate_sequence(pred_sequence_path: str, gt_sequence_path: str, img_size: Tuple[int, int] = (450, 4002)) -> Dict[
    str, float]:
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


def _create_performance_heatmap(df, plots_dir):
    """Create performance heatmaps for each metric."""
    print("Creating performance heatmaps...")

    # Create heatmaps for each metric
    metrics = ['MOTA', 'IDF1', 'MOTP', 'Precision', 'Recall']

    for metric in metrics:
        # Create pivot table for heatmap
        pivot_data = df.pivot_table(
            values=metric,
            index='model',
            columns='method',
            aggfunc='mean'
        )

        # Create the heatmap
        plt.figure(figsize=(12, 8))

        # Special handling for MOTP (lower is better)
        if metric == 'MOTP':
            # Use reverse colormap for MOTP since lower is better
            cmap = 'RdYlGn'  # Red for high (bad), Green for low (good)
            title = f'{metric} Performance Heatmap (Lower is Better)'
        else:
            cmap = 'RdYlGn_r'  # Red for low (bad), Green for high (good)
            title = f'{metric} Performance Heatmap'

        sns.heatmap(pivot_data,
                    annot=True,
                    fmt='.3f',
                    cmap=cmap,
                    center=pivot_data.mean().mean(),
                    linewidths=1.0,
                    annot_kws={"size": 10},
                    cbar_kws={'label': metric})

        plt.title(title, fontsize=14, fontweight='bold', pad=20)
        plt.xlabel('Method', fontsize=12, fontweight='bold')
        plt.ylabel('Model', fontsize=12, fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)

        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f'{metric}_heatmap.eps'),
                    dpi=300, bbox_inches='tight')
        plt.close()

    # Create a combined overview heatmap
    fig, axes = plt.subplots(2, 3, figsize=(26, 20))
    axes = axes.flatten()

    for i, metric in enumerate(metrics):
        ax = axes[i]

        pivot_data = df.pivot_table(
            values=metric,
            index='model',
            columns='method',
            aggfunc='mean'
        )

        # Special handling for MOTP
        if metric == 'MOTP':
            cmap = 'RdYlGn'
            title = f'{metric} (Lower is Better)'
        else:
            cmap = 'RdYlGn_r'
            title = metric

        sns.heatmap(pivot_data,
                    annot=True,
                    fmt='.3f',
                    cmap=cmap,
                    center=pivot_data.mean().mean(),
                    linewidths=1,
                    annot_kws={"size": 6},
                    ax=ax,
                    cbar_kws={'label': metric})

        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xlabel('Method', fontsize=10, fontweight='bold')
        ax.set_ylabel('Model', fontsize=10, fontweight='bold')
        ax.tick_params(axis='x', rotation=45, labelsize=9)
        ax.tick_params(axis='y', rotation=0, labelsize=9)

    # Remove the empty subplot
    axes[-1].remove()

    plt.suptitle('Performance Heatmaps Overview', fontsize=16, fontweight='bold', y=0.95)
    plt.subplots_adjust(wspace=1.5, hspace=1)  # adjust as needed
    plt.savefig(os.path.join(plots_dir, 'performance_heatmaps_overview.eps'),
                dpi=300, bbox_inches='tight')
    plt.close()


def _create_per_model_bar_charts(df, plots_dir):
    """Create individual bar charts for each model showing MOTA, IDF1, and MOTP."""
    print("Creating per-model bar charts...")

    # Set style
    plt.style.use('seaborn-v0_8')

    # Get unique models
    models = df['model'].unique()

    for model in models:
        model_data = df[df['model'] == model]

        # Calculate average metrics per method
        method_metrics = model_data.groupby('method').agg({
            'MOTA': 'mean',
            'IDF1': 'mean',
            'MOTP': 'mean'
        }).reset_index()

        # Create the plot
        fig, ax = plt.subplots(figsize=(12, 8))

        # Set up positions
        methods = method_metrics['method'].tolist()
        x_pos = np.arange(len(methods))
        width = 0.25

        # Create bars
        bars1 = ax.bar(x_pos - width, method_metrics['MOTA'], width,
                       label='MOTA', color='#2E86AB', alpha=0.8)
        bars2 = ax.bar(x_pos, method_metrics['IDF1'], width,
                       label='IDF1', color='#A23B72', alpha=0.8)
        bars3 = ax.bar(x_pos + width, method_metrics['MOTP'], width,
                       label='MOTP (lower is better)', color='#F18F01', alpha=0.8)

        # Customize the plot
        ax.set_xlabel('Methods', fontsize=12, fontweight='bold')
        ax.set_ylabel('Metric Value', fontsize=12, fontweight='bold')
        ax.set_title(f'{model} - Performance Metrics by Method',
                     fontsize=14, fontweight='bold', pad=30)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(methods, rotation=45, ha='right')
        ax.legend(loc='upper right', frameon=True, fancybox=True, shadow=True)
        ax.grid(True, alpha=0.3, axis='y')

        # Add value labels on bars
        def add_value_labels(bars, values):
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax.annotate(f'{value:.3f}',
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3),
                            textcoords="offset points",
                            ha='center', va='bottom',
                            fontsize=9, fontweight='bold')

        add_value_labels(bars1, method_metrics['MOTA'])
        add_value_labels(bars2, method_metrics['IDF1'])
        add_value_labels(bars3, method_metrics['MOTP'])

        # Set y-axis limits
        ax.set_ylim(0, max(method_metrics[['MOTA', 'IDF1', 'MOTP']].max()) * 1.1)

        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f'{model}_bar_chart.eps'),
                    dpi=300, bbox_inches='tight')
        plt.close()


def _create_per_model_radar_charts(df, plots_dir):
    """Create radar charts per model with reduced visual overlap."""
    print("Creating per-model radar charts...")

    # Define metrics
    raw_metrics = ['MOTA', 'IDF1', 'MOTP', 'Precision', 'Recall']
    radar_labels = ['MOTA', 'IDF1', 'MOTP\n(inverted)', 'Precision', 'Recall']

    # Compute normalization ranges
    overall_stats = {
        metric: {'min': df[metric].min(), 'max': df[metric].max()}
        for metric in raw_metrics
    }

    def normalize_metric(value, metric):
        min_val = overall_stats[metric]['min']
        max_val = overall_stats[metric]['max']
        if max_val == min_val:
            return 0.5
        norm = (value - min_val) / (max_val - min_val)
        return 1 - norm if metric == 'MOTP' else norm

    # Loop through models
    for model in df['model'].unique():
        model_data = df[df['model'] == model]

        # Mean per method
        method_metrics = model_data.groupby('method').agg({m: 'mean' for m in raw_metrics}).reset_index()

        # Sort methods so best ones are plotted last (on top)
        method_metrics = method_metrics.sort_values(by='MOTA')

        fig, ax = plt.subplots(figsize=(9, 9), subplot_kw=dict(projection='polar'))

        angles = [n / len(raw_metrics) * 2 * pi for n in range(len(raw_metrics))]
        angles += angles[:1]

        num_methods = len(method_metrics)
        colors = sns.color_palette("pastel", num_methods)

        for idx, row in method_metrics.iterrows():
            # Normalized radar values
            values = [normalize_metric(row[m], m) for m in raw_metrics]
            values += values[:1]

            color = colors[idx]
            label = row['method']

            # Line only, no fill (or very transparent)
            ax.plot(angles, values, color=color, linewidth=0.8, linestyle='-', label=label, marker='o', markersize=3)

        # Axis setup
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(radar_labels, fontsize=11, fontweight='bold')
        ax.set_ylim(0, 1)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=9)
        ax.grid(True, alpha=0.2)

        # Title and legend
        ax.set_title(f'{model} - Normalized Radar Chart', fontsize=14, fontweight='bold', pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.2, 1.0), fontsize=9, frameon=False)

        # Save chart
        os.makedirs(plots_dir, exist_ok=True)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f'{model}_radar_chart.eps'), dpi=300, bbox_inches='tight')
        plt.close()



def _create_metric_distribution_boxplots(df, plots_dir):
    """Create boxplots showing metric distributions by sequence."""
    print("Creating metric distribution boxplots...")

    # Create a comprehensive boxplot visualization
    fig, axes = plt.subplots(2, 3, figsize=(20, 14))
    axes = axes.flatten()

    metrics = ['MOTA', 'IDF1', 'MOTP', 'Precision', 'Recall']

    for i, metric in enumerate(metrics):
        ax = axes[i]

        # Create boxplot
        box_data = []
        labels = []

        for sequence in sorted(df['sequence'].unique()):
            seq_data = df[df['sequence'] == sequence][metric]
            if len(seq_data) > 0:
                box_data.append(seq_data)
                labels.append(sequence)

        if box_data:
            bp = ax.boxplot(box_data, labels=labels, patch_artist=True)

            # Color the boxes
            colors = plt.cm.Set3(np.linspace(0, 1, len(box_data)))
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)

        ax.set_title(f'{metric} Distribution by Sequence',
                     fontsize=12, fontweight='bold')
        ax.set_ylabel(metric, fontsize=11, fontweight='bold')
        ax.tick_params(axis='x', rotation=90)
        ax.grid(True, alpha=0.3)

        # Special handling for MOTP (lower is better)
        if metric == 'MOTP':
            ax.set_title(f'{metric} Distribution by Sequence (lower is better)',
                         fontsize=12, fontweight='bold')

    # Remove the empty subplot
    axes[-1].remove()

    plt.suptitle('Metric Distributions Across Sequences',
                 fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'metric_distributions_by_sequence.eps'),
                dpi=300, bbox_inches='tight')
    plt.close()

def _create_method_comparison_matrix(df, plots_dir):
    """Create a matrix showing method rankings across all models."""
    print("Creating method comparison matrix...")

    # Calculate average performance per method across all models
    method_performance = df.groupby('method').agg({
        'MOTA': 'mean',
        'IDF1': 'mean',
        'MOTP': 'mean',
        'Precision': 'mean',
        'Recall': 'mean'
    }).reset_index()

    # Rank methods for each metric (lower rank = better performance)
    # For MOTP, lower is better, so we rank ascending
    method_performance['MOTA_rank'] = method_performance['MOTA'].rank(ascending=False)
    method_performance['IDF1_rank'] = method_performance['IDF1'].rank(ascending=False)
    method_performance['MOTP_rank'] = method_performance['MOTP'].rank(ascending=True)  # Lower is better
    method_performance['Precision_rank'] = method_performance['Precision'].rank(ascending=False)
    method_performance['Recall_rank'] = method_performance['Recall'].rank(ascending=False)

    # Create ranking matrix
    ranking_data = method_performance[
        ['method', 'MOTA_rank', 'IDF1_rank', 'MOTP_rank', 'Precision_rank', 'Recall_rank']]
    ranking_pivot = ranking_data.set_index('method')

    plt.figure(figsize=(12, 8))
    sns.heatmap(ranking_pivot, annot=True, fmt='.0f', cmap='RdYlGn_r',
                center=ranking_pivot.mean().mean(), square=True, linewidths=0.5,
                cbar_kws={'label': 'Rank (1=Best)'})

    plt.title('Method Ranking Matrix (Lower Numbers = Better Performance)',
              fontsize=14, fontweight='bold', pad=30)
    plt.xlabel('Metrics', fontsize=12, fontweight='bold')
    plt.ylabel('Methods', fontsize=12, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)

    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'method_ranking_matrix.eps'),
                dpi=300, bbox_inches='tight')
    plt.close()

def _create_top_performers_summary(df, plots_dir, save_format='png'):
    """Create a summary visualization of top performing combinations."""
    print("Creating top performers summary...")

    # Calculate overall performance score (weighted average)
    df_summary = df.groupby(['model', 'method']).agg({
        'MOTA': 'mean',
        'IDF1': 'mean',
        'MOTP': 'mean',
        'Precision': 'mean',
        'Recall': 'mean',
        'frames_processed': 'mean'
    }).reset_index()

    # Invert MOTP
    df_summary['MOTP_inverted'] = 1 - df_summary['MOTP']

    # Composite score
    df_summary['composite_score'] = (
            df_summary['MOTA'] * 0.3 +
            df_summary['IDF1'] * 0.25 +
            df_summary['MOTP_inverted'] * 0.2 +
            df_summary['Precision'] * 0.125 +
            df_summary['Recall'] * 0.125
    )

    # Sort by score and get top 10
    top_10 = df_summary.sort_values('composite_score', ascending=False).head(10)
    top_5 = top_10.head(5)

    # --- Plotting ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

    # --- Left: Composite scores ---
    bars = ax1.barh(
        range(len(top_10)),
        top_10['composite_score'],
        color=plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(top_10))),
        edgecolor='black'
    )

    ax1.set_yticks(range(len(top_10)))
    ax1.set_yticklabels([f"{row['model']}\n{row['method']}" for _, row in top_10.iterrows()])
    ax1.set_xlabel('Composite Score', fontsize=12, fontweight='bold')
    ax1.set_title('Top 10 Model-Method Combinations\n(by Composite Score)',
                  fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='x')

    for i, (bar, value) in enumerate(zip(bars, top_10['composite_score'])):
        ax1.text(value + 0.005, bar.get_y() + bar.get_height() / 2,
                 f'{value:.3f}', va='center', fontsize=10, fontweight='bold')

    # --- Right: Grouped Bar Chart ---
    metrics = ['MOTA', 'IDF1', 'MOTP_inverted', 'Precision', 'Recall']
    labels = ['MOTA', 'IDF1', 'MOTP (inv)', 'Precision', 'Recall']
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']

    x = np.arange(len(top_5))  # Number of groups
    bar_width = 0.15

    for i, metric in enumerate(metrics):
        ax2.bar(
            x + i * bar_width,
            top_5[metric],
            width=bar_width,
            label=labels[i],
            color=colors[i],
            edgecolor='black'
        )

    ax2.set_xticks(x + bar_width * (len(metrics) - 1) / 2)
    ax2.set_xticklabels([f"{row['model']}\n{row['method']}" for _, row in top_5.iterrows()],
                        rotation=45, ha='right')

    ax2.set_ylim(0, 1.1)
    ax2.set_ylabel('Normalized Metric Values', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Top 5 Combinations', fontsize=12, fontweight='bold')
    ax2.set_title('Detailed Metrics Breakdown\n(Top 5 Combinations)',
                  fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')

    # Move legend outside
    ax2.legend(loc='upper left', bbox_to_anchor=(1.02, 1))

    plt.tight_layout()

    # Save
    out_path = os.path.join(plots_dir, f'top_performers_summary.{save_format}')
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()


def create_comprehensive_visualizations(df, plots_dir):
    """Create all visualization types including new enhanced ones."""
    print("Creating comprehensive evaluation visualizations...")

    # Ensure plots directory exists
    os.makedirs(plots_dir, exist_ok=True)

     #Original visualizations
    _create_performance_heatmap(df, plots_dir)
    _create_per_model_bar_charts(df, plots_dir)
    _create_per_model_radar_charts(df, plots_dir)
    _create_metric_distribution_boxplots(df, plots_dir)

    # New enhanced visualizations
    _create_method_comparison_matrix(df, plots_dir)
    _create_top_performers_summary(df, plots_dir)

    print(f"All enhanced visualizations saved to: {plots_dir}")

    # Create a summary report
    _create_visualization_summary_report(plots_dir)


def _create_visualization_summary_report(plots_dir):
    """Create a summary report of all generated visualizations."""
    report_path = os.path.join(plots_dir, 'visualization_summary.txt')

    with open(report_path, 'w') as f:
        f.write("TRACKING EVALUATION VISUALIZATION SUMMARY\n")
        f.write("=" * 50 + "\n\n")

        f.write("Generated Visualizations:\n")
        f.write("-" * 30 + "\n")

        visualizations = [
            ("Performance Heatmaps", "MOTA/IDF1/MOTP/Precision/Recall heatmaps and overview"),
            ("Per-Model Bar Charts", "Individual bar charts for each model"),
            ("Per-Model Radar Charts", "Normalized performance radar charts"),
            ("Metric Distribution Boxplots", "Distribution analysis by sequence"),
            ("Sequence Difficulty Analysis", "MOTA-based sequence difficulty ranking"),
            ("Method Comparison Matrix", "Method ranking across all metrics"),
            ("Performance vs Complexity", "Scatter plots showing trade-offs"),
            ("Top Performers Summary", "Composite score ranking and detailed breakdown"),
            ("Sequence Performance Heatmaps", "Performance across sequences for each metric")
        ]

        for name, description in visualizations:
            f.write(f"{name:<35}: {description}\n")

        f.write(f"\nAll visualizations saved to: {plots_dir}\n")
        f.write("\nKey Insights to Look For:\n")
        f.write("-" * 30 + "\n")
        f.write("• MOTP heatmaps: Green = better (lower values)\n")
        f.write("• Other metrics: Green = better (higher values)\n")
        f.write("• Top performers: Composite score balances all metrics\n")


    print(f"Visualization summary report saved to: {report_path}")

# Updated run_evaluation_v2 function with integrated enhanced visualizations
def run_evaluation_v2(results_dir: str, gt_dir: str, output_file: str = "evaluation_results.json",
                      img_width: int = 450, img_height: int = 4002):
    """
    Your complete evaluation function with enhanced visualizations integrated.
    """
    from collections import defaultdict
    import json
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    import os
    import warnings
    warnings.filterwarnings('ignore')

    all_results = {}
    all_metrics = defaultdict(list)

    # For plotting - collect data as we go
    plot_data = []

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

                        # Collect for plotting
                        plot_data.append({
                            'model': model_name,
                            'method': method_name,
                            'sequence': sequence_name,
                            'MOTA': metrics.get('MOTA', 0),
                            'IDF1': metrics.get('IDF1', 0),
                            'MOTP': metrics.get('MOTP', 0),
                            'Precision': metrics.get('Precision', 0),
                            'Recall': metrics.get('Recall', 0),
                            'frames_processed': metrics.get('frames_processed', 0)
                        })

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

    # Print top performing combinations (extended metrics)
    print(f"\n{'Top Performing Model-Method Combinations (by MOTA)':<100}")
    print(f"{'-' * 120}")

    # Collect all combinations
    all_combinations = []
    for model in model_method_summary:
        for method in model_method_summary[model]:
            combo_data = model_method_summary[model][method]
            all_combinations.append((
                f"{model}_{method}",
                combo_data.get('avg_MOTA', 0.0),
                combo_data.get('avg_IDF1', 0.0),
                combo_data.get('avg_MOTP', 0.0),
                combo_data.get('avg_Precision', 0.0),
                combo_data.get('avg_Recall', 0.0),
                combo_data['count'],
                combo_data['total_frames']
            ))

    # Define column widths
    name_width = 65
    metric_width = 10

    # Print header
    print(f"\n{'Model_Method':<{name_width}}"
          f"{'MOTA':<{metric_width}}"
          f"{'IDF1':<{metric_width}}"
          f"{'MOTP':<{metric_width}}"
          f"{'Precision':<{metric_width}}"
          f"{'Recall':<{metric_width}}"
          f"{'Seqs':<6}"
          f"{'Frames':<10}")
    print("-" * (name_width + 5 * metric_width + 6 + 10))

    # Print all combinations
    for combo in all_combinations:
        print(f"{combo[0]:<{name_width}}"
              f"{combo[1]:<{metric_width}.3f}"
              f"{combo[2]:<{metric_width}.3f}"
              f"{combo[3]:<{metric_width}.3f}"
              f"{combo[4]:<{metric_width}.3f}"
              f"{combo[5]:<{metric_width}.3f}"
              f"{combo[6]:<6}"
              f"{combo[7]:<10}")

    report_path = "methods_performance_report.txt"
    with open(report_path, "w") as f:
        f.write(f"{'Model_Method':<{name_width}}"
                f"{'MOTA':<{metric_width}}"
                f"{'IDF1':<{metric_width}}"
                f"{'MOTP':<{metric_width}}"
                f"{'Precision':<{metric_width}}"
                f"{'Recall':<{metric_width}}"
                f"{'Seqs':<6}"
                f"{'Frames':<10}\n")
        f.write("-" * (name_width + 5 * metric_width + 6 + 10) + "\n")

    # CREATE ENHANCED VISUALIZATIONS
    if plot_data:
        print("\nGenerating enhanced visualizations...")
        df = pd.DataFrame(plot_data)

        plots_dir = os.path.join(os.path.dirname(output_file), 'plots')
        os.makedirs(plots_dir, exist_ok=True)

        create_comprehensive_visualizations(df, plots_dir)

        print(f"Enhanced visualizations saved to: {plots_dir}")

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
    results = run_evaluation_v2(args.results_dir, args.gt_dir, args.output,
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