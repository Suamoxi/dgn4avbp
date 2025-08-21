# dgn4avbp/transforms_local.py
import torch
import math

class EnsureEdgeAttrFromPos:
    """Ensure data.edge_attr = pos[j]-pos[i] if missing."""
    def __call__(self, data):
        if getattr(data, 'edge_attr', None) is None and hasattr(data, 'pos') and hasattr(data, 'edge_index'):
            row, col = data.edge_index
            data.edge_attr = data.pos[col] - data.pos[row]
        return data

class ScaleEdgeAttr:
    """Multiply edge_attr (i.e., relative positions) by a factor like 0.015 or 0.02."""
    def __init__(self, factor: float):
        self.factor = float(factor)
    def __call__(self, data):
        if hasattr(data, 'edge_attr') and data.edge_attr is not None:
            data.edge_attr = data.edge_attr * self.factor
        return data

class ScaleAttr:
    """
    Min–max scale a tensor attribute to [-1, 1].
    vmin/vmax can be scalars OR length-C tensors to scale per-channel.
    """
    def __init__(self, name: str, vmin, vmax):
        self.name = name
        self.vmin = torch.as_tensor(vmin, dtype=torch.float32)
        self.vmax = torch.as_tensor(vmax, dtype=torch.float32)

    def __call__(self, data):
        if not hasattr(data, self.name):
            return data
        x = getattr(data, self.name).float()
        vmin = self.vmin.to(x.device)
        vmax = self.vmax.to(x.device)
        # broadcast per-channel if needed
        if vmin.ndim == 0: vmin = vmin.view(1)
        if vmax.ndim == 0: vmax = vmax.view(1)
        if x.dim() >= 2 and vmin.numel() == 1 and vmax.numel() == 1:
            # single range for all channels
            y = 2.0 * (x - vmin) / (vmax - vmin + 1e-12) - 1.0
        else:
            # per-channel (assume channels on last dim)
            # expand vmin/vmax to match channel count
            C = x.shape[-1]
            if vmin.numel() != C or vmax.numel() != C:
                raise ValueError(f"{self.name}: channel count mismatch (got C={C}, vmin={vmin.numel()}, vmax={vmax.numel()})")
            while vmin.dim() < x.dim():
                vmin = vmin.unsqueeze(0)
                vmax = vmax.unsqueeze(0)
            y = 2.0 * (x - vmin) / (vmax - vmin + 1e-12) - 1.0
        setattr(data, self.name, y)
        return data

class EdgeCondFreeStreamLocalAxes:
    """
    Compute per-edge conditional features: projection of free-stream U_inf onto an
    edge-local orthonormal frame (t, n, b). Produces edge_cond with 3 scalars per edge.
    """
    def __init__(self, U_inf):
        self.U = torch.as_tensor(U_inf, dtype=torch.float32).view(1, 3)

    def __call__(self, data):
        if not (hasattr(data, 'edge_index') and hasattr(data, 'pos')):
            return data
        row, col = data.edge_index
        e = data.pos[col] - data.pos[row]             # [E,3]
        eps = 1e-9
        t = e / (e.norm(dim=1, keepdim=True) + eps)   # unit tangent, [E,3]

        # Build an orthonormal frame (t, n, b)
        ref_x = torch.tensor([1.0, 0.0, 0.0], device=t.device).view(1, 3).expand_as(t)
        use_y = (t.abs().argmax(dim=1) == 0).view(-1, 1)          # if t ~ x, use y as ref
        ref = torch.where(use_y, torch.tensor([0.,1.,0.], device=t.device).view(1,3).expand_as(t), ref_x)
        n = torch.nn.functional.normalize(torch.cross(t, ref, dim=1), dim=1)  # [E,3]
        b = torch.nn.functional.normalize(torch.cross(t, n,   dim=1), dim=1)  # [E,3]

        U = self.U.to(t.device).expand_as(t)                       # [E,3]
        proj_t = (U * t).sum(dim=1)
        proj_n = (U * n).sum(dim=1)
        proj_b = (U * b).sum(dim=1)
        data.edge_cond = torch.stack([proj_t, proj_n, proj_b], dim=1)  # [E,3]
        return data

class CenterPos:
    """Optionally center positions to zero mean like pOnEllipse."""
    def __call__(self, data):
        if hasattr(data, 'pos'):
            data.pos = data.pos - data.pos.mean(dim=0, keepdim=True)
        return data

class MeshCoarseningNOP:
    """
    Placeholder for multi-scale pyramid. Returns data unchanged.
    (If you later add a real coarsener, replace this with your implementation.)
    """
    def __call__(self, data):
        return data
