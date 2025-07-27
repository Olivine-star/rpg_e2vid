#!/usr/bin/env python3
"""
Automatic reconstruction script for all event data files in EVSR dataset.
Processes all IR.txt and result.txt files and saves results in the same directories.
Auto-crops generated images to original sensor size.
"""

import os
import glob
import subprocess
import sys
import cv2
import numpy as np
import pandas as pd
from pathlib import Path


def find_all_event_files(dataset_root):
    """Find all event data files (IR.txt and result.txt) in the dataset."""
    event_files = []

    # Search for IR.txt files
    ir_files = glob.glob(os.path.join(dataset_root, "**", "IR.txt"), recursive=True)
    event_files.extend(ir_files)

    # Search for result.txt files
    result_files = glob.glob(
        os.path.join(dataset_root, "**", "result.txt"), recursive=True
    )
    event_files.extend(result_files)

    return sorted(event_files)


def get_sensor_size(event_file):
    """Read sensor size from the first line of the event file."""
    try:
        header = pd.read_csv(
            event_file,
            sep=r"\s+",
            header=None,
            names=["width", "height"],
            dtype={"width": int, "height": int},
            nrows=1,
        )
        width, height = header.values[0]
        return width, height
    except Exception as e:
        print(f"Error reading sensor size from {event_file}: {e}")
        return None, None


def crop_generated_images(event_file, output_dir, target_width=240, target_height=180):
    """Crop all generated images to target size (default: 240x180)."""
    # Get current sensor size from file for reference
    file_width, file_height = get_sensor_size(event_file)
    if file_width is None or file_height is None:
        print(f"❌ Could not read sensor size from {event_file}")
        return False

    print(f"📏 File sensor size: {file_width} x {file_height}")
    print(f"🎯 Target crop size: {target_width} x {target_height}")

    # Use target dimensions for cropping
    width, height = target_width, target_height

    # Find all generated images (both frames and events)
    frame_pattern = os.path.join(output_dir, "frame_*.png")
    frame_files = glob.glob(frame_pattern)

    events_pattern = os.path.join(output_dir, "events", "events_*.png")
    event_files = glob.glob(events_pattern)

    all_files = frame_files + event_files

    if not all_files:
        print(f"⚠️  No images found in {output_dir}")
        return True

    print(
        f"🖼️  Found {len(frame_files)} frame images and {len(event_files)} event images to crop"
    )

    cropped_count = 0
    skipped_count = 0
    original_width, original_height = None, None

    for image_file in all_files:
        try:
            # Read image
            img = cv2.imread(image_file)
            if img is None:
                print(f"❌ Could not read image: {image_file}")
                continue

            current_height, current_width = img.shape[:2]

            # Record original dimensions for reporting
            if original_width is None:
                original_width, original_height = current_width, current_height

            # Check if cropping is needed
            if current_width == width and current_height == height:
                skipped_count += 1
                continue

            # Crop from top-left corner to original sensor size
            if current_width >= width and current_height >= height:
                cropped_img = img[:height, :width]

                # Save cropped image (overwrite original)
                cv2.imwrite(image_file, cropped_img)
                cropped_count += 1
            else:
                print(
                    f"⚠️  Image too small to crop: {image_file} ({current_width}x{current_height})"
                )

        except Exception as e:
            print(f"❌ Error processing {image_file}: {e}")

    if cropped_count > 0 and original_width is not None:
        print(
            f"✂️  Cropped {cropped_count} images from {original_width}x{original_height} to {width}x{height}"
        )
    if skipped_count > 0:
        print(f"✅ {skipped_count} images already correct size")

    print(f"✅ Auto-crop complete: {cropped_count} cropped, {skipped_count} skipped")
    return True


def run_reconstruction(event_file, model_path):
    """Run E2VID reconstruction for a single event file."""
    # Get the directory of the event file to use as output folder
    event_dir = os.path.dirname(event_file)

    cmd = [
        sys.executable,
        "run_reconstruction.py",
        "-c",
        model_path,
        "-i",
        event_file,
        "--output_folder",
        event_dir,
        "--dataset_name",
        "E2VID_Results",
        "--use_event_path_for_output",
        "--auto_hdr",
        "--show_events",
    ]

    print(f"Processing: {event_file}")

    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"✅ Reconstruction success: {event_file}")

        # Auto-crop generated images to 240x180 (actual sensor size)
        output_dir = os.path.join(event_dir, "E2VID_Results")
        print(f"🔄 Auto-cropping images in: {output_dir}")

        if crop_generated_images(
            event_file, output_dir, target_width=240, target_height=180
        ):
            print(f"✅ Complete with auto-crop: {event_file}")
            return True
        else:
            print(f"⚠️  Reconstruction succeeded but cropping failed: {event_file}")
            return True  # Still consider it a success since reconstruction worked

    except subprocess.CalledProcessError as e:
        print(f"❌ Failed: {event_file}")
        print(f"Error: {e.stderr}")
        return False


def main():
    # Configuration
    dataset_root = r"C:\Users\steve\Dataset\EVSR\IR"
    model_path = "pretrained/E2VID_lightweight.pth.tar"

    print("=== Auto Reconstruction for EVSR Dataset ===")
    print(f"Dataset: {dataset_root}")
    print(f"Model: {model_path}")

    # Validate model exists
    if not os.path.exists(model_path):
        print(f"Error: Model file not found: {model_path}")
        return 1

    # Find all event files
    event_files = find_all_event_files(dataset_root)

    if not event_files:
        print("No event files found!")
        return 1

    print(f"\nFound {len(event_files)} event files:")
    for f in event_files:
        print(f"  {f}")

    # Process all files
    print(f"\n🚀 Starting reconstruction...")
    successful = 0
    failed = 0

    for i, event_file in enumerate(event_files, 1):
        print(f"\n--- {i}/{len(event_files)} ---")
        if run_reconstruction(event_file, model_path):
            successful += 1
        else:
            failed += 1

    # Summary
    print(f"\n=== Summary ===")
    print(f"✅ Successful: {successful}")
    print(f"❌ Failed: {failed}")
    print(f"📊 Total: {len(event_files)}")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    exit(main())
