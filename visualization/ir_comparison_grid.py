#!/usr/bin/env python3
"""
Academic Comparison Figure Generator for Event-based Vision Results
Creates publication-quality comparison figures with customizable grid layout
Fixed version with proper spacing and bottom column labels
"""

import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from pathlib import Path
import glob

# ==================== CONFIGURATION SECTION ====================
# Modify these settings according to your needs

# Grid configuration
ROWS = 4
COLS = 9
ROW_LABELS = ["(1)", "(2)", "(3)", "(4)"]

# Column labels
COLUMN_LABELS = [
    "(a)",
    "(b)",
    "(c)",
    "(d)",
    "(e)",
    "(f)",
    "(g)",
    "(h)",
    "(i)",
]

# Dataset base path
DATASET_PATH = r"C:\Users\steve\Dataset\EVSR\IR"

# Dataset sequences to use for each row
ROW_SEQUENCES = ["boxes_6dof", "dynamic_6dof", "office_zigzag", "shapes_6dof"]

# Folder mapping for each column type
FOLDER_MAPPING = {
    "LR_events": "SR_Test/LR/E2VID_Results/events",
    "HR_events_GT": "SR_Test/HR/E2VID_Results/events",
    "HR_events_baseline": "SR_Test/HRPre/E2VID_Results/events",  # UPDATE: Which is Li et al. baseline?
    "HR_events_ours": "SR_Test/HRPre2(Louck-light-p-learn)/E2VID_Results/events",  # UPDATE: Which is your best?
    "frame_GT": "images",
    "LR_frames": "SR_Test/LR/E2VID_Results",
    "HR_GT_frames": "SR_Test/HR/E2VID_Results",
    "baseline_frames": "SR_Test/HRPre/E2VID_Results",  # UPDATE: Which is Li et al. baseline?
    "ours_frames": "SR_Test/HRPre2(Louck-light-p-learn)/E2VID_Results",  # UPDATE: Which is your best?
}

# Image selection for each row - specify exact filenames (without extension)
ROW_EVENT_IMAGES = [
    "events_0000031485",  # Row 1 event image
    "events_0000031485",  # Row 2 event image
    "events_0000031485",  # Row 3 event image
    "events_0000031485",  # Row 4 event image
]

ROW_FRAME_IMAGES = [
    "frame_0000031485",  # Row 1 frame image (for reconstructed frames)
    "frame_0000031485",  # Row 2 frame image
    "frame_0000031485",  # Row 3 frame image
    "frame_0000031485",  # Row 4 frame image
]

# For GT frames from images folder, specify the original frame filename
ROW_GT_FRAME_IMAGES = [
    "frame_00000109",  # Row 1 GT frame from images folder
    "frame_00000106",  # Row 2 GT frame from images folder
    "frame_00000114",  # Row 3 GT frame from images folder
    "frame_00000120",  # Row 4 GT frame from images folder
]

# Figure settings
FIGURE_SIZE = (18, 8)  # Width, Height in inches
DPI = 1000
SAVE_FORMATS = ["png", "pdf"]  # Export both PNG and PDF
OUTPUT_FILENAME_BASE = (
    "ir_academic_comparison"  # Will add extension automatically
)

# Label position settings - adjust these to fine-tune label placement
ROW_LABEL_X = 0.001  # X position for row labels (negative to place outside figure)
ROW_LABEL_FONTSIZE = 14  # Font size for row labels
ROW_LABEL_WEIGHT = "bold"  # Font weight for row labels
ROW_LABEL_HA = "right"  # Horizontal alignment: 'left', 'center', 'right'
ROW_LABEL_VA = "center"  # Vertical alignment: 'top', 'center', 'bottom'
ROW_LABEL_FONT = "Times New Roman"  # Font family for row labels

COL_LABEL_Y = -0.01  # Y position for column labels (negative to place outside figure)
COL_LABEL_FONTSIZE = 14  # Font size for column labels
COL_LABEL_WEIGHT = "bold"  # Font weight for column labels
COL_LABEL_HA = "center"  # Horizontal alignment: 'left', 'center', 'right'
COL_LABEL_VA = "top"  # Vertical alignment: 'top', 'center', 'bottom'
COL_LABEL_FONT = "Times New Roman"  # Font family for column labels

# Grid margins for label positioning (adjust if labels are cut off)
GRID_LEFT = 0.08  # Left margin of the image grid
GRID_RIGHT = 0.98  # Right margin of the image grid

# ==================== HELPER FUNCTIONS ====================


def find_specific_images(
    base_path,
    sequence,
    folder_mapping,
    event_filename,
    frame_filename,
    gt_frame_filename,
):
    """
    Find specific image files across different methods for a given sequence
    Returns a dictionary with method names as keys and file paths as values
    """
    matching_frames = {}

    for method_key, folder_path in folder_mapping.items():
        full_path = os.path.join(base_path, sequence, folder_path)

        if not os.path.exists(full_path):
            print(f"Warning: Path does not exist: {full_path}")
            continue

        # Determine which filename to use based on method type
        if "events" in method_key:
            target_filename = event_filename
        elif method_key == "frame_GT":
            target_filename = gt_frame_filename
        else:
            target_filename = frame_filename

        # Look for the specific image file
        image_extensions = [".png", ".jpg", ".jpeg"]
        found_file = None

        for ext in image_extensions:
            candidate_path = os.path.join(full_path, target_filename + ext)
            if os.path.exists(candidate_path):
                found_file = candidate_path
                break

        if found_file:
            matching_frames[method_key] = found_file
        else:
            print(f"Warning: Image {target_filename} not found in {full_path}")

    return matching_frames


def load_and_resize_image(image_path, target_size=None, is_lr=False):
    """
    Load image and optionally resize to target size
    For LR images, crop to the meaningful 120x90 region (top-left corner)
    """
    try:
        img = mpimg.imread(image_path)

        # Convert to RGB if RGBA
        if len(img.shape) == 3 and img.shape[2] == 4:
            img = img[:, :, :3]

        # Handle LR images: crop to meaningful region (120x90 from top-left)
        if is_lr:
            # LR images are 240x180 but only top-left 120x90 contains meaningful data
            if img.shape[0] >= 90 and img.shape[1] >= 120:
                img = img[:90, :120]  # Crop to top-left 120x90 region
            else:
                print(
                    f"Warning: LR image {image_path} has unexpected size: {img.shape}"
                )

        return img
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None


def create_comparison_figure():
    """
    Create the main comparison figure with proper spacing and bottom labels
    """
    # Create figure with specified size
    fig, axes = plt.subplots(ROWS, COLS, figsize=FIGURE_SIZE, dpi=DPI)

    # Remove spacing between subplots - following DRAW_NMNIST.py approach for no gaps
    plt.subplots_adjust(wspace=0.0, hspace=0.0)

    # Ensure axes is 2D array
    if ROWS == 1:
        axes = axes.reshape(1, -1)
    if COLS == 1:
        axes = axes.reshape(-1, 1)

    # Column mapping order (matches COLUMN_LABELS order)
    column_methods = [
        "LR_events",
        "HR_events_GT",
        "HR_events_baseline",
        "HR_events_ours",
        "frame_GT",
        "LR_frames",
        "HR_GT_frames",
        "baseline_frames",
        "ours_frames",
    ]

    # Process each row
    for row_idx in range(ROWS):
        if row_idx >= len(ROW_SEQUENCES):
            print(f"Warning: Not enough sequences defined for row {row_idx}")
            continue

        sequence = ROW_SEQUENCES[row_idx]
        print(f"Processing row {row_idx + 1}: {sequence}")

        # Get specific image filenames for this row
        event_filename = (
            ROW_EVENT_IMAGES[row_idx]
            if row_idx < len(ROW_EVENT_IMAGES)
            else "events_0000031485"
        )
        frame_filename = (
            ROW_FRAME_IMAGES[row_idx]
            if row_idx < len(ROW_FRAME_IMAGES)
            else "frame_0000031485"
        )
        gt_frame_filename = (
            ROW_GT_FRAME_IMAGES[row_idx]
            if row_idx < len(ROW_GT_FRAME_IMAGES)
            else "frame_00000100"
        )

        print(f"  Event image: {event_filename}")
        print(f"  Frame image: {frame_filename}")
        print(f"  GT frame image: {gt_frame_filename}")

        # Find specific images for this sequence
        matching_frames = find_specific_images(
            DATASET_PATH,
            sequence,
            FOLDER_MAPPING,
            event_filename,
            frame_filename,
            gt_frame_filename,
        )

        # Process each column
        for col_idx in range(COLS):
            ax = axes[row_idx, col_idx]

            if col_idx < len(column_methods):
                method_key = column_methods[col_idx]

                if method_key in matching_frames:
                    # Check if this is an LR column
                    is_lr_column = "LR" in method_key

                    # Load and display image
                    img = load_and_resize_image(
                        matching_frames[method_key], is_lr=is_lr_column
                    )
                    if img is not None:
                        ax.imshow(img, interpolation="nearest", extent=[0, 1, 0, 1])
                        print(f"  Loaded {method_key}: {matching_frames[method_key]}")
                    else:
                        ax.text(
                            0.5,
                            0.5,
                            "Image\nNot Found",
                            ha="center",
                            va="center",
                            transform=ax.transAxes,
                            fontsize=8,
                        )
                else:
                    ax.text(
                        0.5,
                        0.5,
                        "Path\nNot Found",
                        ha="center",
                        va="center",
                        transform=ax.transAxes,
                        fontsize=8,
                    )

            # Remove axis ticks and labels - following DRAW_NMNIST.py approach
            ax.axis("off")

            # No labels on individual axes - will add them outside the grid

    # Add row labels AFTER layout adjustment - get actual subplot positions
    for row_idx, row_label in enumerate(ROW_LABELS):
        if row_idx < ROWS:
            # Get the actual position of the first subplot in this row
            ax_pos = axes[row_idx, 0].get_position()
            # Calculate the vertical center of this subplot
            row_center_y = ax_pos.y0 + (ax_pos.height / 2)

            # Add row label on the left, always outside the plot area
            label_x = ax_pos.x0 - ROW_LABEL_X  # Use configurable offset

            fig.text(
                label_x,
                row_center_y,  # Actual subplot center Y position
                row_label,
                fontsize=ROW_LABEL_FONTSIZE,
                fontweight=ROW_LABEL_WEIGHT,
                color="black",
                ha="right",  # Right-align so text doesn't overlap with images
                va="center",
                transform=fig.transFigure,
            )

    # Add column labels AFTER layout adjustment - place them outside the plot area
    for col_idx, col_label in enumerate(COLUMN_LABELS):
        if col_idx < COLS:
            # Get the actual position of the bottom subplot in this column
            ax_pos = axes[ROWS - 1, col_idx].get_position()

            # Calculate label position (below the subplot, outside plot area)
            label_x = ax_pos.x0 + ax_pos.width / 2  # Center horizontally
            label_y = (
                ax_pos.y0 + COL_LABEL_Y
            )  # Use configurable offset (COL_LABEL_Y is negative)

            # Add column label as figure text (outside the subplot area)
            fig.text(
                label_x,
                label_y,
                col_label,
                fontsize=COL_LABEL_FONTSIZE,
                fontweight=COL_LABEL_WEIGHT,
                color="black",
                ha="center",
                va="top",  # Top-align so text doesn't overlap with images
                transform=fig.transFigure,
            )

    # Save figure in multiple formats
    saved_files = []
    for save_format in SAVE_FORMATS:
        output_filename = f"{OUTPUT_FILENAME_BASE}.{save_format}"
        if save_format == "pdf":
            plt.savefig(
                output_filename,
                format="pdf",
                dpi=1000,
                bbox_inches="tight",
                pad_inches=0,
            )
        else:
            plt.savefig(output_filename, dpi=600, bbox_inches="tight", pad_inches=0)
        saved_files.append(output_filename)
        print(f"Figure saved as: {output_filename}")

    print(f"\nAll figures saved: {', '.join(saved_files)}")

    # Show figure
    plt.show()


# ==================== MAIN EXECUTION ====================

if __name__ == "__main__":
    print("Academic Comparison Figure Generator (Fixed Version)")
    print("=" * 60)
    print(f"Grid size: {ROWS} rows × {COLS} columns")
    print(f"Dataset path: {DATASET_PATH}")
    print(f"Sequences: {ROW_SEQUENCES}")
    print("\nGenerating comparison figure...")

    create_comparison_figure()
