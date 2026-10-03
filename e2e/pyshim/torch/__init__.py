# E2E guard: the e2e server puts this directory first on PYTHONPATH so `import torch` fails fast.
# The app then reports gpu=false and Laya stays in cached mode. Nothing in the e2e run may touch the GPU.
raise ImportError("torch is disabled in the FraudShield e2e environment (no GPU use allowed)")
