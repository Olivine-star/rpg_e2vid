#!/usr/bin/env python3
"""
Standalone script to crop existing generated images to original sensor size.
Useful for post-processing already generated images.
"""

import os
import glob
import cv2
import pandas as pd
import argparse
from pathlib import Path


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


def crop_images_in_directory(
    event_file, output_dir, backup=False, target_width=240, target_height=180
):
    """Crop all generated images in a directory to target size (default: 240x180)."""
    # Get file sensor size for reference
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
        f"🖼️  Found {len(frame_files)} frame images and {len(event_files)} event images to process"
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
                # Create backup if requested
                if backup:
                    backup_file = image_file.replace(".png", "_original.png")
                    if not os.path.exists(backup_file):
                        cv2.imwrite(backup_file, img)

                cropped_img = img[:height, :width]

                # Save cropped image
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

    print(f"✅ Processing complete: {cropped_count} cropped, {skipped_count} skipped")
    return True


def find_all_result_directories(dataset_root):
    """Find all E2VID_Results directories in the dataset."""
    result_dirs = []

    # Search for E2VID_Results directories
    for root, dirs, files in os.walk(dataset_root):
        if "E2VID_Results" in dirs:
            result_dir = os.path.join(root, "E2VID_Results")
            # Check if there's a corresponding event file
            event_files = []
            for event_name in ["IR.txt", "result.txt"]:
                event_path = os.path.join(root, event_name)
                if os.path.exists(event_path):
                    event_files.append(event_path)

            if event_files:
                result_dirs.append((result_dir, event_files[0]))

    return result_dirs


def main():
    parser = argparse.ArgumentParser(
        description="Crop existing generated images to original sensor size"
    )
    parser.add_argument(
        "-d",
        "--dataset_root",
        type=str,
        help="Root directory of the dataset",
    )
    parser.add_argument(
        "--backup",
        action="store_true",
        help="Create backup copies of original images before cropping",
    )
    parser.add_argument(
        "--single_dir",
        type=str,
        help="Process only a single E2VID_Results directory (provide path to the directory containing both the event file and E2VID_Results folder)",
    )
    parser.add_argument(
        "--target_width",
        type=int,
        default=240,
        help="Target width for cropping (default: 240)",
    )
    parser.add_argument(
        "--target_height",
        type=int,
        default=180,
        help="Target height for cropping (default: 180)",
    )

    args = parser.parse_args()

    # Validate arguments
    if not args.single_dir and not args.dataset_root:
        parser.error("Either --single_dir or -d/--dataset_root must be provided")

    print("=== Image Cropping Tool ===")

    if args.single_dir:
        # Process single directory
        event_files = []
        for event_name in ["IR.txt", "result.txt"]:
            event_path = os.path.join(args.single_dir, event_name)
            if os.path.exists(event_path):
                event_files.append(event_path)

        if not event_files:
            print(f"❌ No event file found in {args.single_dir}")
            return 1

        result_dir = os.path.join(args.single_dir, "E2VID_Results")
        if not os.path.exists(result_dir):
            print(f"❌ E2VID_Results directory not found in {args.single_dir}")
            return 1

        print(f"Processing: {result_dir}")
        crop_images_in_directory(
            event_files[0],
            result_dir,
            args.backup,
            args.target_width,
            args.target_height,
        )

    else:
        # Process all directories
        result_dirs = find_all_result_directories(args.dataset_root)

        if not result_dirs:
            print("❌ No E2VID_Results directories found!")
            return 1

        print(f"Found {len(result_dirs)} result directories:")
        for result_dir, event_file in result_dirs:
            print(f"  {result_dir}")

        print(f"\n🚀 Starting batch cropping...")
        successful = 0
        failed = 0

        for i, (result_dir, event_file) in enumerate(result_dirs, 1):
            print(f"\n--- {i}/{len(result_dirs)} ---")
            print(f"Processing: {result_dir}")

            if crop_images_in_directory(
                event_file,
                result_dir,
                args.backup,
                args.target_width,
                args.target_height,
            ):
                successful += 1
            else:
                failed += 1

        # Summary
        print(f"\n=== Summary ===")
        print(f"✅ Successful: {successful}")
        print(f"❌ Failed: {failed}")
        print(f"📊 Total: {len(result_dirs)}")

    return 0


if __name__ == "__main__":
    exit(main())
