# SPDX-License-Identifier: GPL-3.0-only
"""PPO source-operand branch encoding; includes FP32 ties."""
import jax.numpy as j


def branches(a, r):
    clipped = j.clip(r, .8, 1.2)
    raw, capped = r*a, clipped*a
    threshold_tie = (r == j.float32(.8)) | (r == j.float32(1.2))
    outside = (r < j.float32(.8)) | (r > j.float32(1.2))
    # 0 raw/unclipped (including interior min ties); 1 clipped;
    # 2 outside rounding tie (half gradient); 3 clip threshold tie.
    branch = j.where(raw < capped, 0, j.where(raw > capped, 1,
                     j.where(threshold_tie, 3, j.where(outside, 2, 0))))
    return j.where(a == 0, 0, branch).astype(j.int32)
