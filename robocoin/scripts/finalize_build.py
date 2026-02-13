"""
Finalize a tfds Beam-based build when the client process (e.g. SLURM job)
was killed after the Dataflow pipeline completed but before tfds could
write metadata and rename the incomplete directory.

Steps replicated from tensorflow_datasets internals:
  1. Read {split}.split_info.json files from the incomplete directory
  2. Instantiate the builder to get the feature spec
  3. Populate SplitInfo objects and set them on DatasetInfo
  4. Write features.json and dataset_info.json
  5. Rename incomplete.XXXXXX_<version>/ → <version>/

Usage:
    python finalize_build.py \
        --data_dir gs://saksham-euw4/robocoin_bimanual/ \
        --incomplete_dir_name incomplete.JA8J6S_1.0.0
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'robocoin_bimanual'))

import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow_datasets.core import naming, splits as splits_lib, file_adapters
from etils import epath


def main():
    parser = argparse.ArgumentParser(description = "Finalize a tfds build from an incomplete directory.")
    parser.add_argument("--data_dir", required = True,
                        help = "Root data dir (e.g. gs://saksham-euw4/robocoin_bimanual/)")
    parser.add_argument("--incomplete_dir_name", required = True,
                        help = "Name of the incomplete directory (e.g. incomplete.JA8J6S_1.0.0)")
    parser.add_argument("--dry_run", action = "store_true",
                        help = "Print what would be done without making changes")
    args = parser.parse_args()

    data_dir = args.data_dir.rstrip('/')
    dataset_name = "robocoin_bimanual"
    version = "1.0.0"

    incomplete_dir = os.path.join(data_dir, dataset_name, args.incomplete_dir_name)
    final_dir = os.path.join(data_dir, dataset_name, version)

    print(f"Incomplete dir: {incomplete_dir}")
    print(f"Final dir:      {final_dir}")

    # Verify incomplete dir exists
    if not tf.io.gfile.exists(incomplete_dir):
        print(f"ERROR: Incomplete directory does not exist: {incomplete_dir}")
        sys.exit(1)

    # Check if final dir already exists
    if tf.io.gfile.exists(final_dir):
        print(f"ERROR: Final directory already exists: {final_dir}")
        print("The build may have already been finalized.")
        sys.exit(1)

    # --- Step 1: Read split_info.json files ---
    splits = ["train", "val"]
    split_data = {}
    for split_name in splits:
        split_info_path = os.path.join(
            incomplete_dir, f"{dataset_name}-{split_name}.split_info.json"
        )
        print(f"Reading {split_info_path} ...")
        if not tf.io.gfile.exists(split_info_path):
            print(f"ERROR: Split info file not found: {split_info_path}")
            print("The Beam pipeline may not have finished for this split.")
            sys.exit(1)
        with tf.io.gfile.GFile(split_info_path, 'r') as f:
            split_data[split_name] = json.loads(f.read())
        info = split_data[split_name]
        print(f"  {split_name}: {len(info['shard_lengths'])} shards, "
              f"{sum(info['shard_lengths'])} examples, "
              f"{info['total_size']:,} bytes")

    if args.dry_run:
        print("\n[DRY RUN] Would write features.json, dataset_info.json, then rename directory.")
        return

    # --- Step 2: Instantiate the builder to get _info() ---
    from robocoin_bimanual_dataset_builder import RobocoinBimanual
    builder = RobocoinBimanual(data_dir = data_dir)

    # --- Step 3: Create SplitInfo objects ---
    file_format = file_adapters.DEFAULT_FILE_FORMAT.value
    split_infos = []
    for split_name in splits:
        info = split_data[split_name]
        filename_template = naming.ShardedFileTemplate(
            data_dir = epath.Path(incomplete_dir),
            dataset_name = dataset_name,
            split = split_name,
            filetype_suffix = file_format,
        )
        split_info = splits_lib.SplitInfo(
            name = split_name,
            shard_lengths = info["shard_lengths"],
            num_bytes = info["total_size"],
            filename_template = filename_template,
        )
        split_infos.append(split_info)

    split_dict = splits_lib.SplitDict(split_infos)
    builder.info.set_splits(split_dict)

    # --- Step 4: Write features.json and dataset_info.json ---
    print(f"\nWriting metadata to {incomplete_dir} ...")
    builder.info.write_to_directory(incomplete_dir)
    print("  Wrote features.json and dataset_info.json")

    # --- Step 5: Delete split_info.json files (tfds does this in finalize()) ---
    for split_name in splits:
        split_info_path = os.path.join(
            incomplete_dir, f"{dataset_name}-{split_name}.split_info.json"
        )
        print(f"  Deleting {split_info_path}")
        tf.io.gfile.remove(split_info_path)

    # --- Step 6: Rename incomplete dir to final dir ---
    print(f"\nRenaming {incomplete_dir} → {final_dir} ...")
    tf.io.gfile.rename(incomplete_dir, final_dir, overwrite = False)
    print("Done! Dataset finalized successfully.")

    # Print summary
    print(f"\nDataset available at: {final_dir}")
    for si in split_infos:
        print(f"  {si.name}: {sum(si.shard_lengths)} examples, "
              f"{len(si.shard_lengths)} shards, {si.num_bytes:,} bytes")


if __name__ == "__main__":
    main()
