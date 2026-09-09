"""Match each ORIGINAL inference script's physical gripper postprocessing."""
import numpy as np


def clip_physical_gripper(physical, kind):
    if physical.shape != (50, 7):
        raise ValueError('Expected physical action [50, 7]')
    if kind not in ('pi05', 'pap_moe'):
        raise ValueError(kind)
    # Pi0.5 clips the entire physical chunk BEFORE copying its execution prefix.
    # PAP clips only the prefix view. Neither changes the normalized RTC leftover.
    count = 50 if kind == 'pi05' else 10
    physical[:count, 6] = np.clip(physical[:count, 6], 0., .8)
    return physical
