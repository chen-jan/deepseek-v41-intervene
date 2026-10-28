"""Graph-safe capture: device rings written by the edit kernel, host row map, dump.

Format: INTERFACE.md section 6. Rank 0 only; other ranks allocate empty rings and never assign rows.

The host assigns ring rows in ``prepare_inputs`` (before the step, also before graph replays) and writes them into
``tok[:, CROW / VROW]``; the kernel writes the rows. Each ring has one extra trash row at its end (``coef_trash``,
``vec_trash``) that takes the torch path's writes for uncaptured tokens; ``coef`` / ``vec`` are views without it.
The kernel also writes each coef row's device position into ``cpos`` (dumped as ``coef_pos_dev``), so a reader can
check that the host row map (``coef_pos``) lines up with the tokens the kernel saw. Rows are assigned when a token is computed, so a preempted and recomputed request gets fresh rows for the same
positions (keep the last row per ``(req, pos)`` when reading a dump).
"""

import json
import os
import time
from dataclasses import dataclass

import numpy as np
import torch

from . import HC, HIDDEN


@dataclass
class CapSpec:
    """One request's capture request, parsed from ``vllm_xargs``."""

    kind: str  # "coef" or "vec"
    lo: int  # cap_from (inclusive)
    hi: int  # cap_to (exclusive)
    pos: np.ndarray | None  # explicit positions (sorted int64), overrides lo/hi
    tag: str
    req_id: str


_MAP_FIELDS = ("pos", "arm", "req", "tag")


class CaptureRings:
    """Coefficient ring ``coef`` [COEF_ROWS, I, 3, 4] fp32, its device positions ``cpos`` [COEF_ROWS] int32, and
    vector ring ``vec`` [VEC_ROWS, N_CAP, 5120] bf16."""

    def __init__(self, n_inst: int, capture_layers: list[int], coef_rows: int, vec_rows: int,
                 device: torch.device, enabled: bool):
        """Allocate both rings (zero-size when ``enabled`` is False) and an empty host row map."""
        self.enabled = enabled
        self.n_inst = n_inst
        self.capture_layers = list(capture_layers)
        self.n_cap = len(self.capture_layers)
        self.coef_rows = coef_rows if enabled else 0
        self.vec_rows = vec_rows if enabled and self.n_cap else 0
        self.coef_trash = self.coef_rows
        self.vec_trash = self.vec_rows
        self.coef_buf = torch.zeros(self.coef_rows + 1, n_inst, 3, HC, dtype=torch.float32, device=device)
        self.cpos_buf = torch.full((self.coef_rows + 1,), -1, dtype=torch.int32, device=device)
        self.vec_buf = torch.zeros(self.vec_rows + 1, max(self.n_cap, 1), HIDDEN, dtype=torch.bfloat16,
                                   device=device)
        self.coef = self.coef_buf[: self.coef_rows]
        self.cpos = self.cpos_buf[: self.coef_rows]
        self.vec = self.vec_buf[: self.vec_rows]
        self._map = {"coef": {f: np.full(self.coef_rows, -1, np.int32) for f in _MAP_FIELDS},
                     "vec": {f: np.full(self.vec_rows, -1, np.int32) for f in _MAP_FIELDS}}
        self.reset()

    def reset(self) -> None:
        """Clear the host map and bump pointers (device rows are overwritten later, not zeroed)."""
        self.coef_used = 0
        self.vec_used = 0
        self.coef_overflow = 0
        self.vec_overflow = 0
        self.req_ids: list[str] = []
        self.tags: list[str] = []
        self._req_index: dict[str, int] = {}
        self._tag_index: dict[str, int] = {}

    @staticmethod
    def _intern(index: dict[str, int], names: list[str], name: str) -> int:
        if name not in index:
            index[name] = len(names)
            names.append(name)
        return index[name]

    def _take(self, ring: str, n: int) -> np.ndarray:
        """Bump-allocate ``n`` rows of ``ring``; rows that do not fit are -1 and counted as overflow."""
        used = getattr(self, f"{ring}_used")
        k = max(0, min(n, getattr(self, f"{ring}_rows") - used))
        rows = np.full(n, -1, np.int32)
        rows[:k] = np.arange(used, used + k, dtype=np.int32)
        setattr(self, f"{ring}_used", used + k)
        setattr(self, f"{ring}_overflow", getattr(self, f"{ring}_overflow") + n - k)
        return rows

    def _record(self, ring: str, rows: np.ndarray, pos: np.ndarray, arm: int, req: int, tag: int) -> None:
        ok = rows >= 0
        m = self._map[ring]
        m["pos"][rows[ok]] = pos[ok]
        m["arm"][rows[ok]] = arm
        m["req"][rows[ok]] = req
        m["tag"][rows[ok]] = tag

    def assign(self, specs: list["CapSpec | None"], start: np.ndarray, cnt: np.ndarray, arm: np.ndarray,
               out_crow: np.ndarray, out_vrow: np.ndarray) -> None:
        """Assign ring rows to this step's tokens.

        ``specs[i]``, ``start[i]``, ``cnt[i]``, ``arm[i]`` describe batch row i (first position, scheduled tokens,
        arm). Fills ``out_crow`` / ``out_vrow`` (flattened tokens, length ``cnt.sum()``) with row ids or -1 and
        appends to the host map. A full ring assigns -1 and bumps ``coef_overflow`` / ``vec_overflow``.
        """
        out_crow.fill(-1)
        out_vrow.fill(-1)
        if not self.enabled:
            return
        off = 0
        for i, spec in enumerate(specs):
            n = int(cnt[i])
            if spec is not None and n > 0:
                pos = int(start[i]) + np.arange(n, dtype=np.int64)
                if spec.pos is not None:
                    sel = np.flatnonzero(np.isin(pos, spec.pos))
                else:
                    sel = np.flatnonzero((pos >= spec.lo) & (pos < spec.hi))
                if sel.size:
                    req = self._intern(self._req_index, self.req_ids, spec.req_id)
                    tag = self._intern(self._tag_index, self.tags, spec.tag)
                    rows = self._take("coef", sel.size)
                    out_crow[off + sel] = rows
                    self._record("coef", rows, pos[sel], int(arm[i]), req, tag)
                    if spec.kind == "vec" and self.n_cap:
                        rows = self._take("vec", sel.size)
                        out_vrow[off + sel] = rows
                        self._record("vec", rows, pos[sel], int(arm[i]), req, tag)
            off += n

    def dump(self, path: str, meta: dict) -> dict:
        """Write used rows plus the host map to ``path`` (.safetensors) and ``path + '.json'``, then ``reset()``.
        Returns ``{"path", "coef_rows", "vec_rows"}``."""
        from safetensors.torch import save_file

        if self.coef_buf.is_cuda:
            torch.cuda.synchronize(self.coef_buf.device)
        n, m = self.coef_used, self.vec_used
        tensors = {"coef": self.coef[:n].cpu().contiguous(), "coef_pos_dev": self.cpos[:n].cpu().contiguous(),
                   "vec": self.vec[:m].cpu().contiguous()}
        for ring, used in (("coef", n), ("vec", m)):
            for f in _MAP_FIELDS:
                tensors[f"{ring}_{f}"] = torch.from_numpy(self._map[ring][f][:used].copy())
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        tmp = path + ".tmp"
        save_file(tensors, tmp)
        os.replace(tmp, path)
        side = {**meta, "capture_layers": self.capture_layers, "req_ids": self.req_ids, "tags": self.tags,
                "coef_rows": n, "vec_rows": m, "coef_overflow": self.coef_overflow,
                "vec_overflow": self.vec_overflow, "coef_layout": "[row, slot, (c, c_post, pre_mix), stream]",
                "time": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
        with open(path + ".json", "w") as f:
            json.dump(side, f, indent=1)
        self.reset()
        return {"path": path, "coef_rows": n, "vec_rows": m}

    def stats(self) -> dict:
        """``{"coef_rows_used", "vec_rows_used", "coef_overflow", "vec_overflow"}``."""
        return {"coef_rows_used": self.coef_used, "vec_rows_used": self.vec_used,
                "coef_overflow": self.coef_overflow, "vec_overflow": self.vec_overflow}
