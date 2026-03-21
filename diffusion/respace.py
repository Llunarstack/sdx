# Timestep respacing (DiT/OpenAI-style): ddimN and section-based steps.
# Ported from external/DiT/diffusion/respace.py; adapted for our GaussianDiffusion.
import numpy as np


def space_timesteps(num_timesteps: int, section_counts) -> np.ndarray:
    """
    Create a sorted array of timestep indices for diffusion.
    :param num_timesteps: Total steps in the original process (e.g. 1000).
    :param section_counts: Either a list of ints (step count per section) or a string:
        - "ddimN": use DDIM-style striding to get exactly N steps (e.g. "ddim50").
        - "10,15,20": comma-separated counts per section (sections are equal-sized portions of [0, num_timesteps)).
    :return: Sorted 1D numpy array of timestep indices in [0, num_timesteps).
    """
    if isinstance(section_counts, str):
        s = section_counts.strip().lower()
        if s.startswith("ddim"):
            n = int(s[4:].strip())
            for stride in range(1, num_timesteps):
                steps = np.arange(0, num_timesteps, stride)
                if len(steps) == n:
                    return steps
            # Fallback: linear spacing to get ~n steps
            return np.linspace(0, num_timesteps - 1, n, dtype=np.int64)
        section_counts = [int(x.strip()) for x in section_counts.split(",") if x.strip()]
    if not section_counts:
        return np.arange(num_timesteps, dtype=np.int64)
    size_per = num_timesteps // len(section_counts)
    extra = num_timesteps % len(section_counts)
    start_idx = 0
    all_steps = []
    for i, section_count in enumerate(section_counts):
        size = size_per + (1 if i < extra else 0)
        if size < section_count:
            raise ValueError(f"Section has {size} steps but requested {section_count}; cannot divide.")
        if section_count <= 1:
            frac_stride = 1.0
        else:
            frac_stride = (size - 1) / (section_count - 1)
        cur_idx = 0.0
        for _ in range(section_count):
            all_steps.append(start_idx + int(round(cur_idx)))
            cur_idx += frac_stride
        start_idx += size
    return np.unique(np.array(all_steps, dtype=np.int64))
