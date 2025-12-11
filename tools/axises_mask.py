"""AxisesMask

Torch-based Python implementation of the AxisesMask utility (ported from
placo C++). This module intentionally does not reference placo or any
bindings; it provides a small class that can be used by tasks to mask
position/orientation axes.

API:
  mask = AxisesMask()
  mask.set_axises("xy", "local")
  mask.R_local_world = ... (3x3 torch tensor)
  M_masked = mask.apply(M)  # M shape (3, n) or (3,) -> returns rows selected

All vectors/matrices are torch tensors.
"""
from typing import List, Sequence, Union
import torch


class AxisesMask:
    TASK_FRAME = 0
    LOCAL_FRAME = 1
    CUSTOM_FRAME = 2

    def __init__(self, device: Union[torch.device, str, None] = None, dtype=torch.float64):
        self.device = torch.device(device) if device is not None else torch.device('cpu')
        self.dtype = dtype
        # default rotations are identity
        self.R_local_world = torch.eye(3, device=self.device, dtype=self.dtype)
        self.R_custom_world = torch.eye(3, device=self.device, dtype=self.dtype)
        # default keep all axes
        self.indices: List[int] = [0, 1, 2]
        self.frame = AxisesMask.TASK_FRAME

    def set_axises(self, axises: str, frame: Union[int, str] = TASK_FRAME):
        """Set the axes to keep. `axises` is a string like "xy" or "z".

        `frame` may be an integer constant (TASK_FRAME/LOCAL_FRAME/CUSTOM_FRAME)
        or one of the strings: 'task'|'world', 'local', 'custom'.
        """
        if not isinstance(axises, str):
            raise TypeError("axises must be a string containing 'x','y','z'")

        # parse frame
        if isinstance(frame, str):
            frame_l = frame.lower()
            if frame_l in ("task", "world"):
                self.frame = AxisesMask.TASK_FRAME
            elif frame_l == "local":
                self.frame = AxisesMask.LOCAL_FRAME
            elif frame_l == "custom":
                self.frame = AxisesMask.CUSTOM_FRAME
            else:
                raise ValueError(f"Invalid frame string: {frame}")
        elif isinstance(frame, int):
            if frame in (AxisesMask.TASK_FRAME, AxisesMask.LOCAL_FRAME, AxisesMask.CUSTOM_FRAME):
                self.frame = frame
            else:
                raise ValueError(f"Invalid frame value: {frame}")
        else:
            raise TypeError("frame must be int or str")

        indices: List[int] = []
        for c in axises:
            cl = c.lower()
            if cl == 'x':
                indices.append(0)
            elif cl == 'y':
                indices.append(1)
            elif cl == 'z':
                indices.append(2)
            else:
                raise ValueError(f"Invalid axis character: {c}")

        # remove duplicates while preserving order
        seen = set()
        ordered = []
        for i in indices:
            if i not in seen:
                ordered.append(i)
                seen.add(i)

        self.indices = ordered

    def apply(self, M: Union[torch.Tensor, Sequence[float]]) -> torch.Tensor:
        """Apply the mask to matrix M.

        M is expected to be shape (3, n) or (3,) (a single 3-vector). The result
        will be the selected rows after optionally rotating M into the chosen
        reference frame.
        """
        # accept many reasonable torch shapes: 1-D length 3, 1-D flattened len=3*n,
        # 2-D with shape (3,n), (6,n) (take top3), (1,3*n) (reshape), (3*n,1) (reshape),
        # or (n,3) (transpose)
        if not isinstance(M, torch.Tensor):
            M = torch.as_tensor(M, device=self.device, dtype=self.dtype)
        else:
            M = M.to(device=self.device, dtype=self.dtype)

        was_1d = False
        # normalize to shape (3, n)
        if M.ndim == 0:
            raise ValueError("Scalar is not a valid input for AxisesMask.apply")
        elif M.ndim == 1:
            if M.shape[0] == 3:
                M = M.unsqueeze(1)
                was_1d = True
            elif M.shape[0] % 3 == 0:
                M = M.view(3, -1)
            else:
                raise ValueError("1-D input length must be 3 or a multiple of 3")
        elif M.ndim == 2:
            r, c = M.shape
            # already (3, n)
            if r == 3:
                pass
            # sometimes the jacobian comes as 6 x n -> take position rows (top 3)
            elif r == 6:
                M = M[0:3, :]
            # flattened row (1, 3*n)
            elif r == 1 and c % 3 == 0:
                M = M.view(3, -1)
            # flattened column (3*n, 1)
            elif c == 1 and r % 3 == 0:
                M = M.view(3, -1)
            # transposed (n,3) -> convert to (3,n)
            elif c == 3 and r % 3 == 0:
                M = M.transpose(0, 1)
            else:
                raise ValueError("Input must be shape (3,n), (6,n), (1,3*n), (3*n,1) or (n,3)")
        else:
            raise ValueError("Input must be 1-D or 2-D tensor")

        # choose rotation based on frame
        if self.frame == AxisesMask.CUSTOM_FRAME:
            M_rot = self.R_custom_world.to(dtype=M.dtype) @ M
        elif self.frame == AxisesMask.LOCAL_FRAME:
            M_rot = self.R_local_world.to(dtype=M.dtype) @ M
        else:
            M_rot = M

        # select rows
        if len(self.indices) == 0:
            # nothing selected -> return empty (0,n)
            out = torch.zeros((0, M_rot.shape[1]), device=self.device, dtype=self.dtype)
        else:
            out = M_rot[self.indices, :]

        if was_1d:
            # if original input was length-3, return 1D vector; otherwise keep (3,) -> squeeze
            if out.shape[1] == 1:
                return out.squeeze(1)
            else:
                return out.squeeze(1)
        return out


__all__ = ["AxisesMask"]
