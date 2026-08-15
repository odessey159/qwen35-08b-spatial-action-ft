from __future__ import annotations

import torch


def main() -> None:
    print(f"torch={torch.__version__}")
    print(f"cuda_available={torch.cuda.is_available()}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available to PyTorch")
    print(f"device={torch.cuda.get_device_name(0)}")
    result = torch.ones(4, device="cuda") @ torch.ones(4, device="cuda")
    print(f"cuda_dot={result.item()}")


if __name__ == "__main__":
    main()
