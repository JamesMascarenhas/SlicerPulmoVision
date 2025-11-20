# PulmoVision checkpoints

This folder stores a lightweight synthetic checkpoint used by the PulmoVision demo UNet3D backend. The default file, `unet3d_synthetic.pt`, is created automatically on first use so the loader can validate and run with deterministic parameters even when no trained weights are available. The runtime logic will fall back to percentile segmentation if the checkpoint cannot be loaded.
