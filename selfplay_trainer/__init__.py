# SPDX-License-Identifier: GPL-3.0-only
"""Native Showdown actors and the JAX policy/PPO learner.

Import this package before JAX to inherit the original launcher's numerical defaults.
An embedding application may set its own explicit environment before importing.
"""
import os
os.environ.setdefault('JAX_ENABLE_X64','true')
os.environ.setdefault('JAX_DEFAULT_MATMUL_PRECISION','highest')
os.environ.setdefault('NVIDIA_TF32_OVERRIDE','0')
__version__ = "0.1.0"
