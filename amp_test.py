import torch

# This test won't work without a CUDA-enabled GPU.
if not torch.cuda.is_available():
    print("CUDA is not available. This test requires a GPU.")
    raise SystemExit

print(f"Using device: {torch.cuda.get_device_name(0)}")
maj, min = torch.cuda.get_device_capability()
print(f"CUDA Capability: {maj}.{min} " + ("(Tensor Cores supported)" if maj >= 7 else "(no Tensor Cores)"))

# --- Settings ---
SIZE = 8192          # A large matrix to make sure the GPU is busy.
ITERATIONS = 100
WARMUP_ITER = 10
AMP_DTYPE = torch.float16

# This is important for a fair test. By default, PyTorch uses TF32 for FP32 matmuls
# on modern GPUs, which is already a form of mixed precision. Turn it off
# to see the true speedup from pure FP32 to AMP.
torch.backends.cuda.matmul.allow_tf32 = False
torch.set_float32_matmul_precision("highest")

device = "cuda"
a = torch.randn(SIZE, SIZE, device=device, dtype=torch.float32)
b = torch.randn(SIZE, SIZE, device=device, dtype=torch.float32)

# Run a few iterations of both modes first to warm up the GPU and JIT compilers.
# This prevents one-time setup costs from affecting the benchmark.
for _ in range(WARMUP_ITER):
    _ = a @ b
    with torch.amp.autocast(device_type="cuda", dtype=AMP_DTYPE, enabled=True):
        _ = a @ b
torch.cuda.synchronize()

# A helper for accurate timing on the GPU.
# CPU-based timers can be misleading due to asynchronous CUDA calls.
def time_kernel(fn):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / 1000.0  # return seconds

# --- FP32 baseline ---
def run_fp32():
    for _ in range(ITERATIONS):
        _ = a @ b

t_fp32 = time_kernel(run_fp32)
print(f"\nFP32 (no TF32) duration: {t_fp32:.4f} s")

# --- AMP (mixed precision) ---
def run_amp():
    # The 'autocast' context manager automatically casts operations
    # to the lower-precision dtype where it's safe and beneficial.
    with torch.amp.autocast(device_type="cuda", dtype=AMP_DTYPE, enabled=True):
        for _ in range(ITERATIONS):
            _ = a @ b

t_amp = time_kernel(run_amp)
print(f"AMP (dtype={AMP_DTYPE}) duration: {t_amp:.4f} s")

# --- Results ---
print("\n" + "="*34)
if t_amp < t_fp32:
    print(f"SUCCESS: AMP speedup {t_fp32 / t_amp:.2f}x")
else:
    print("NOTE: No speedup observed. Check GPU/driver/build;")
    print("      also remember disabling TF32 makes FP32 slower (fairer to AMP).")
print("="*34)