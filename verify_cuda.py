"""验证 PyTorch CUDA 环境。"""

from __future__ import annotations

import sys
import time

import torch

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def main() -> int:
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA build: {torch.version.cuda}")
    print(f"cuDNN: {torch.backends.cudnn.version()}")
    print(f"torch.cuda.is_available(): {torch.cuda.is_available()}")

    if not torch.cuda.is_available():
        print("❌ CUDA 不可用，请检查驱动与 CUDA 版 PyTorch 安装。")
        return 1

    device = torch.device("cuda")
    properties = torch.cuda.get_device_properties(0)
    print(f"设备数量: {torch.cuda.device_count()}")
    print(f"设备名称: {torch.cuda.get_device_name(0)}")
    print(f"计算能力: {properties.major}.{properties.minor}")
    print(f"显存: {properties.total_memory / 1024 ** 3:.1f} GB")

    # 实际执行一次 GPU 计算，确认 CUDA 运行时和 cuBLAS 正常
    size = 4096
    a = torch.randn(size, size, device=device)
    b = torch.randn(size, size, device=device)
    torch.cuda.synchronize()

    iterations = 5
    started = time.perf_counter()
    for _ in range(iterations):
        c = a @ b
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started

    tflops = (2 * size ** 3 * iterations) / elapsed / 1e12
    print(f"{size}x{size} 矩阵乘 x{iterations}: {elapsed * 1000:.1f} ms ({tflops:.1f} TFLOPS)")
    print(f"结果校验 (有限值): {bool(torch.isfinite(c).all())}")
    print("✅ CUDA 验证通过")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
