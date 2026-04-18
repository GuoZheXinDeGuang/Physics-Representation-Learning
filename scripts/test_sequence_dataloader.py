#!/usr/bin/env python3
import os
import tempfile
from pathlib import Path

import h5py
import numpy as np
import torch

from physics_jepa.data import get_train_sequence_dataloader


def create_synthetic_active_matter_dataset(base_dir: Path):
    dataset_root = base_dir / "active_matter" / "data"
    for split in ("train", "valid", "test"):
        split_dir = dataset_root / split
        split_dir.mkdir(parents=True, exist_ok=True)
        file_path = split_dir / "synthetic_active_matter.h5"

        num_objects = 2
        num_frames = 20
        height = 8
        width = 8
        num_channels = 11

        with h5py.File(file_path, "w") as f:
            t0_fields = f.create_group("t0_fields")
            scalars = f.create_group("scalars")

            for channel_idx in range(num_channels):
                data = np.zeros((num_objects, num_frames, height, width), dtype=np.float32)
                for obj_idx in range(num_objects):
                    for time_idx in range(num_frames):
                        # Encode object, channel, and time directly into the field values so
                        # the expected temporal window is easy to verify after loading.
                        data[obj_idx, time_idx, :, :] = (
                            obj_idx * 1000.0 + channel_idx * 100.0 + time_idx
                        )
                t0_fields.create_dataset(f"field_{channel_idx}", data=data)

            scalars.create_dataset("alpha", data=np.array(-3.0, dtype=np.float32))
            scalars.create_dataset("zeta", data=np.array(9.0, dtype=np.float32))
            scalars.create_dataset("L", data=np.array(1.0, dtype=np.float32))


def assert_exact_invocation_smoke_test():
    train_loader = get_train_sequence_dataloader(
        dataset_name="active_matter",
        num_frames=16,
        num_examples=None,
        batch_size=8,
        resolution=224,
        offset=1,
        noise_std=0.0,
    )

    batch = next(iter(train_loader))
    assert set(batch.keys()) == {"sequence", "physical_params"}
    assert batch["sequence"].shape == (8, 11, 16, 224, 224), batch["sequence"].shape
    assert batch["physical_params"].shape == (8, 2), batch["physical_params"].shape
    print("Smoke test passed for the exact invocation.")
    print("sequence shape:", tuple(batch["sequence"].shape))
    print("physical_params shape:", tuple(batch["physical_params"].shape))


def assert_window_contents():
    # Disable shuffle so we can verify the first sample exactly.
    train_loader = get_train_sequence_dataloader(
        dataset_name="active_matter",
        num_frames=16,
        num_examples=None,
        batch_size=2,
        resolution=224,
        offset=1,
        noise_std=0.0,
        shuffle=False,
        num_workers=0,
        persistent_workers=False,
    )

    batch = next(iter(train_loader))
    sequence = batch["sequence"]
    physical_params = batch["physical_params"]

    # Sample 0 should correspond to object 0, t0 = 0 because shuffle=False and offset=1.
    first_sample = sequence[0]
    assert first_sample.shape == (11, 16, 224, 224)

    for channel_idx in range(11):
        observed = first_sample[channel_idx, :, 0, 0].numpy()
        expected = np.array([channel_idx * 100.0 + t for t in range(16)], dtype=np.float32)
        if not np.allclose(observed, expected, atol=1e-4):
            raise AssertionError(
                f"Unexpected values for channel {channel_idx}. "
                f"observed={observed[:4]} expected={expected[:4]}"
            )

    expected_params = np.array([-3.0, 9.0], dtype=np.float32)
    if not np.allclose(physical_params[0].numpy(), expected_params):
        raise AssertionError(
            f"Unexpected physical params. observed={physical_params[0].numpy()} expected={expected_params}"
        )

    print("Deterministic window-content test passed.")
    print("first sample channel 0 timeline:", first_sample[0, :, 0, 0].tolist())


def main():
    with tempfile.TemporaryDirectory() as tmpdir:
        base_dir = Path(tmpdir)
        create_synthetic_active_matter_dataset(base_dir)
        os.environ["THE_WELL_DATA_DIR"] = str(base_dir)

        assert_exact_invocation_smoke_test()
        assert_window_contents()
        print("All sequence dataloader checks passed.")


if __name__ == "__main__":
    main()
