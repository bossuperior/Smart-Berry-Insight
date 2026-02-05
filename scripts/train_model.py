"""
Faster CPU training tweaks for Intel onboard (no CUDA):
- Limit/align thread pools (MKL/OMP/Torch) to reduce oversubscription
- Use smaller imgsz and cache dataset in RAM
- Tune dataloader workers + persistent workers for Windows
- Disable heavy plots during training; keep metrics
"""

# Imports
import os, numpy as np, cv2, ultralytics, torch
print("numpy:", np.__version__)
print("opencv:", cv2.__version__)
print("ultralytics:", ultralytics.__version__)

# Paths / data config
DATA_YAML = "../data/Straw_DS3_Twoclasses/data.yaml"

# CPU threading and perf tuning (helps on Intel CPUs)
def _configure_cpu_threads():
    ncpu = os.cpu_count() or 4
    main_threads = max(1, ncpu - 1)
    # Limit BLAS/OpenMP threads to avoid oversubscription
    os.environ.setdefault("OMP_NUM_THREADS", str(main_threads))
    os.environ.setdefault("MKL_NUM_THREADS", str(main_threads))
    try:
        # Avoid OpenCV creating its own large thread pool
        cv2.setNumThreads(0)
    except Exception:
        pass
    try:
        torch.set_num_threads(main_threads)
        torch.set_num_interop_threads(max(1, main_threads // 2))
        # Prefer high matmul precision optimizations (PyTorch 2.x)
        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision("high")
    except Exception:
        pass
    return main_threads

_threads = _configure_cpu_threads()

from ultralytics import YOLO

# Build model (nano-seg is already the fastest variant)
model = YOLO('yolov8n-seg.pt')

# Dataloader workers: Windows spawn is costly; keep modest
workers = max(0, min(4, _threads // 2))
print(f"Configured CPU threads={_threads}, dataloader workers={workers}")

# Train with CPU-friendly settings
results = model.train(
    data=DATA_YAML,
    device='cpu',           # Force CPU (Intel iGPU training via DirectML is unsupported here)
    epochs=80,              # Slightly fewer epochs; use patience for early-stop
    imgsz=512,              # Smaller image size speeds up CPU training
    batch=4,                # Adjust if you have plenty of system RAM
    cache='ram',            # Cache images in RAM for faster epochs
    workers=workers,        # Modest worker count for Windows/CPU
    plots=False,            # Skip heavy plots during training
    patience=20,            # Early-stop if no improvement
    name="strawberry_seg_twoclasses",
)
