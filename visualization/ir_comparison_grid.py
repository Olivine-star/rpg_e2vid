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
ROWS = 3
COLS = 9
ROW_LABELS = ["(1)", "(2)", "(3)"]

# Column labels
COLUMN_LABELS = [
    "LR",
    "HR-GT",
    "Baseline",
    "Ours",
    "GT Frame",
    "LR",
    "HR-GT",
    "Baseline",
    "Ours",
]

# Dataset base path
DATASET_PATH = r"C:\Users\steve\Dataset\EVSR\IR"

# Dataset sequences to use for each row
ROW_SEQUENCES = ["office_zigzag", "shapes_6dof", "dynamic_6dof"]

# Folder mapping for each column type
FOLDER_MAPPING = {
    "LR_events": "SR_Test/LR/E2VID_Results/events",
    "HR_events_GT": "SR_Test/HR/E2VID_Results/events",
    "HR_events_baseline": "SR_Test/HRPre/E2VID_Results/events",  # UPDATE: Which is Li et al. baseline?
    "HR_events_ours": "SR_Test/HRPre-Louck-light-p/E2VID_Results/events",  # UPDATE: Which is your best?
    "frame_GT": "images",
    "LR_frames": "SR_Test/LR/E2VID_Results",
    "HR_GT_frames": "SR_Test/HR/E2VID_Results",
    "baseline_frames": "SR_Test/HRPre/E2VID_Results",  # UPDATE: Which is Li et al. baseline?
    "ours_frames": "SR_Test/HRPre-Louck-light-p/E2VID_Results",  # UPDATE: Which is your best?
}

# Image selection for each row - specify exact filenames (without extension)
ROW_EVENT_IMAGES = [
    "events_0000094455",  # Row 1 event image
    "events_0000188910",  # Row 2 event image
    "events_0000125940",  # Row 3 event image
]

ROW_FRAME_IMAGES = [
    "frame_0000094455",  # Row 1 frame image (for reconstructed frames)
    "frame_0000188910",  # Row 2 frame image
    "frame_0000125940",  # Row 3 frame image
]

# For GT frames from images folder, specify the original frame filename
ROW_GT_FRAME_IMAGES = [
    "frame_00000117",  # Row 1 GT frame from images folder
    "frame_00000139",  # Row 2 GT frame from images folder
    "frame_00000112",  # Row 3 GT frame from images folder
]

# Manual override for specific grid positions (HIGHEST PRIORITY)
# Format: {(row, col): "absolute_path_to_image"}
# Row and column indices start from 0
# Grid layout: 3 rows × 9 columns
#   Columns: 0=LR_events, 1=HR_events_GT, 2=HR_events_baseline, 3=HR_events_ours,
#           4=frame_GT, 5=LR_frames, 6=HR_GT_frames, 7=baseline_frames, 8=ours_frames
#   Rows: 0=office_zigzag, 1=shapes_6dof, 2=dynamic_6dof
#
# Usage examples:
# - Replace position (0,0) with a specific image: (0, 0): r"C:\path\to\image.png"
# - Replace position (1,4) with another image: (1, 4): r"C:\another\path\image.png"
MANUAL_IMAGE_OVERRIDE = {
    # Uncomment and modify these examples as needed:
    # (0, 0): r"C:\Users\steve\Dataset\EVSR\IR\office_zigzag\SR_Test\HR\E2VID_Results\frame_0000094455.png",
    # (1, 4): r"C:\Users\steve\Dataset\EVSR\IR\shapes_6dof\images\frame_00000139.png",
    # (2, 8): r"C:\Users\steve\Dataset\EVSR\IR\dynamic_6dof\SR_Test\HRPre-Louck-light-p\E2VID_Results\frame_0000125940.png",
}

# Figure settings
# Calculate figure size to ensure 4:3 aspect ratio for each subplot
SUBPLOT_WIDTH = 2.0  # inches per subplot
SUBPLOT_HEIGHT = SUBPLOT_WIDTH * 3 / 4  # 4:3 aspect ratio means height = width * 3/4
FIGURE_SIZE = (COLS * SUBPLOT_WIDTH, ROWS * SUBPLOT_HEIGHT)  # (18, 4.5) for 9x3 grid
DPI = 1000
SAVE_FORMATS = ["png", "pdf"]  # Export both PNG and PDF
OUTPUT_FILENAME_BASE = "ir_academic_comparison"  # Will add extension automatically

# Label position settings - adjust these to fine-tune label placement
ROW_LABEL_X = 0.001  # X position for row labels (negative to place outside figure)
ROW_LABEL_FONTSIZE = 12  # Font size for row labels
ROW_LABEL_WEIGHT = "bold"  # Font weight for row labels
ROW_LABEL_HA = "right"  # Horizontal alignment: 'left', 'center', 'right'
ROW_LABEL_VA = "center"  # Vertical alignment: 'top', 'center', 'bottom'
ROW_LABEL_FONT = "Times New Roman"  # Font family for row labels

COL_LABEL_Y = -0.01  # Y position for column labels (negative to place outside figure)
COL_LABEL_FONTSIZE = 12  # Font size for column labels
COL_LABEL_WEIGHT = "bold"  # Font weight for column labels
COL_LABEL_HA = "center"  # Horizontal alignment: 'left', 'center', 'right'
COL_LABEL_VA = "top"  # Vertical alignment: 'top', 'center', 'bottom'
COL_LABEL_FONT = "Times New Roman"  # Font family for column labels

# Top section labels (Event Stream and Image Reconstruction)
TOP_LABEL_Y = 1.0  # Y position for top section labels (positive to place above figure)
TOP_LABEL_FONTSIZE = 14  # Font size for top section labels
TOP_LABEL_WEIGHT = "bold"  # Font weight for top section labels
TOP_LABEL_COLOR = "black"  # Color for top section labels (black)
TOP_LABEL_FONT = "Times New Roman"  # Font family for top section labels

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


def load_and_resize_image(image_path, is_lr=False):
    """
    Load image and optionally crop LR images to meaningful region
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

            # Check for manual override first (HIGHEST PRIORITY)
            manual_key = (row_idx, col_idx)
            if manual_key in MANUAL_IMAGE_OVERRIDE:
                manual_path = MANUAL_IMAGE_OVERRIDE[manual_key]
                print(
                    f"  Using manual override for position ({row_idx}, {col_idx}): {manual_path}"
                )

                # Load manual image
                img = load_and_resize_image(manual_path, is_lr=False)
                if img is not None:
                    ax.imshow(img, interpolation="nearest", extent=[0, 4, 0, 3])
                    print(f"  ✅ Manual image loaded: {manual_path}")
                else:
                    ax.text(
                        0.5,
                        0.5,
                        "Manual Image\nNot Found",
                        ha="center",
                        va="center",
                        transform=ax.transAxes,
                        fontsize=8,
                        color="red",
                    )
                    print(f"  ❌ Manual image failed to load: {manual_path}")

            # Use default logic if no manual override
            elif col_idx < len(column_methods):
                method_key = column_methods[col_idx]

                if method_key in matching_frames:
                    # Check if this is an LR column
                    is_lr_column = "LR" in method_key

                    # Load and display image
                    img = load_and_resize_image(
                        matching_frames[method_key], is_lr=is_lr_column
                    )
                    if img is not None:
                        # Display image with 4:3 aspect ratio extent
                        ax.imshow(img, interpolation="nearest", extent=[0, 4, 0, 3])
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

            # Empty cell if no method defined and no manual override
            else:
                ax.text(
                    0.5,
                    0.5,
                    "Empty",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                    fontsize=8,
                    color="gray",
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

    # Add top section labels (Event Stream and Image Reconstruction)
    # Event Stream label - covers columns 0-3 (left 4 columns)
    if COLS >= 4:
        # Get positions of first and fourth columns
        left_ax_pos = axes[0, 0].get_position()
        right_ax_pos = axes[0, 3].get_position()

        # Calculate center position for Event Stream label
        event_label_x = (left_ax_pos.x0 + right_ax_pos.x0 + right_ax_pos.width) / 2
        event_label_y = (
            left_ax_pos.y1 + TOP_LABEL_Y - 1.0
        )  # Adjust for proper positioning

        fig.text(
            event_label_x,
            event_label_y,
            "Event Stream",
            fontsize=TOP_LABEL_FONTSIZE,
            fontweight=TOP_LABEL_WEIGHT,
            color=TOP_LABEL_COLOR,
            ha="center",
            va="bottom",
            transform=fig.transFigure,
        )

    # Image Reconstruction label - covers columns 5-8 (right 4 columns)
    if COLS >= 9:
        # Get positions of sixth and ninth columns
        left_ax_pos = axes[0, 5].get_position()
        right_ax_pos = axes[0, 8].get_position()

        # Calculate center position for Image Reconstruction label
        recon_label_x = (left_ax_pos.x0 + right_ax_pos.x0 + right_ax_pos.width) / 2
        recon_label_y = (
            left_ax_pos.y1 + TOP_LABEL_Y - 1.0
        )  # Adjust for proper positioning

        fig.text(
            recon_label_x,
            recon_label_y,
            "Image Reconstruction",
            fontsize=TOP_LABEL_FONTSIZE,
            fontweight=TOP_LABEL_WEIGHT,
            color=TOP_LABEL_COLOR,
            ha="center",
            va="bottom",
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
