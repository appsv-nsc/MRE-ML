import math, torch
import torch.nn.functional as F
import numpy as np  # only used for type checks; not for RNG
from torchvision.transforms import functional as TF


# ---------- helpers ----------

def gaussian_kernel2d(ks, sigma, device, dtype):
    ax = torch.arange(ks, device=device, dtype=dtype) - (ks - 1) / 2
    try:
        yy, xx = torch.meshgrid(ax, ax, indexing="ij")
    except TypeError:
        yy, xx = torch.meshgrid(ax, ax)  # defaults to "ij"
    k = torch.exp(-(xx**2 + yy**2) / (2 * sigma * sigma))
    k = k / k.sum().clamp_min(torch.finfo(dtype).eps)   # safety guard
    return k.view(1, 1, ks, ks)

def smooth_noise(noise, ks=11, sigma=3.0, pad_mode="reflect"):
    # noise: (N=1, C=1, H, W)
    k = gaussian_kernel2d(ks, sigma, noise.device, noise.dtype)
    pad = ks // 2
    if pad_mode is not None:
        noise = F.pad(noise, (pad, pad, pad, pad), mode=pad_mode)
        return F.conv2d(noise, k, padding=0)
    else:
        return F.conv2d(noise, k, padding=pad)


def _to_bchw(arr):
    t = torch.as_tensor(arr)
    if t.ndim == 2:  # mask HxW -> (1,H,W)
        t = t.unsqueeze(0)
    elif t.ndim == 3:
        if t.shape[0] <= 16 and t.shape[1] == t.shape[2]:
            pass  # (C,H,W)
        else:
            t = t.permute(2, 0, 1)  # (H,W,C)->(C,H,W)
    else:
        raise ValueError(f"Expected 2D/3D array, got {t.shape}")
    return t.unsqueeze(0).float()  # (1,C,H,W)


def _from_bchw(t, channels_last=True, is_mask=False):
    """
    Convert from BCHW (or CHW) tensor to numpy, optionally channels_last (HWC).
    - t: (B,C,H,W), (C,H,W), or None
    - channels_last: if True → (B,H,W,C) / (H,W,C)
    - is_mask: if True, cast to uint8 before numpy
    """
    if t is None:
        return None

    if not torch.is_tensor(t):
        raise TypeError(f"_from_bchw expected tensor or None, got {type(t)}")

    # detach + cpu
    t = t.detach().to("cpu")

    if t.ndim == 4:  # (B,C,H,W)
        if channels_last:
            t = t.permute(0, 2, 3, 1)  # → (B,H,W,C)
    elif t.ndim == 3:  # (C,H,W)
        if channels_last:
            t = t.permute(1, 2, 0)     # → (H,W,C)
    elif t.ndim == 2:  # (H,W) mask case
        pass
    else:
        raise ValueError(f"Unexpected tensor shape {tuple(t.shape)} for _from_bchw")

    arr = t.numpy()
    if is_mask:
        arr = (arr > 0.5).astype("uint8")
    return arr


# ---------- new range + probability helpers (🔧 NEW) ----------

def _uniform(a, b, g, device):
    return (torch.rand((), generator=g, device=device) * (b - a) + a).item()


def _angle_range(rot_degrees):
    # float -> (-d, d); tuple/list -> (min,max)
    if isinstance(rot_degrees, (tuple, list)) and len(rot_degrees) == 2:
        return float(rot_degrees[0]), float(rot_degrees[1])
    d = float(rot_degrees)
    return -d, d


def _translate_range(translate_frac):
    """
    Returns ((minx,maxx),(miny,maxy)) in FRACTIONS of (W,H).
    - float f                  -> ((-f, +f), (-f, +f))
    - (fx, fy)                 -> ((-fx, +fx), (-fy, +fy))
    - ((minx,maxx),(miny,maxy))-> as is
    """
    if isinstance(translate_frac, (tuple, list)):
        # (fx, fy)
        if len(translate_frac) == 2 and all(isinstance(v, (int, float)) for v in translate_frac):
            fx, fy = map(float, translate_frac)
            return (-fx, fx), (-fy, fy)
        # ((minx,maxx),(miny,maxy))
        if (len(translate_frac) == 2 and
                all(isinstance(v, (tuple, list)) and len(v) == 2 for v in translate_frac)):
            (minx, maxx), (miny, maxy) = translate_frac
            return (float(minx), float(maxx)), (float(miny), float(maxy))
        raise ValueError("translate_frac must be float, (fx, fy), or ((minx,maxx),(miny,maxy))")
    f = float(translate_frac)
    return (-f, f), (-f, f)


def _sample_affine(
    rot_degrees,
    translate_frac,
    H,
    W,
    g,
    device,
    rot_prob = 1.0,
    trans_prob = 1.0,
    batch_size = None,
    dtype= None,
):
    """
    Vectorized sampler for affine params.

    Returns:
      if batch_size is not None:
        angle:     (B,)     degrees, dtype
        translate: (B, 2)   pixels,  dtype
      else (back-compat single-sample mode):
        angle:     scalar tensor (0-dim)
        translate: (2,) tensor

    Notes:
    - rot_prob / trans_prob apply per-example when batch_size is provided.
    - rot_degrees: float or (min,max) in degrees
    - translate_frac: float, (fx,fy), or ((minx,maxx),(miny,maxy)) in *fractions* of W/H, converted to pixels.
    """
    device = torch.device(device)
    if dtype is None:
        dtype = torch.float32

    # helpers that return (lo, hi) as Python floats
    def _angle_range(a):
        # a: float or (min,max)
        if isinstance(a, (tuple, list)):
            return float(a[0]), float(a[1])
        a = float(a)
        return -a, a

    def _translate_range(t):
        # t: float, (fx, fy), or ((minx,maxx),(miny,maxy))
        if isinstance(t, (tuple, list)):
            if len(t) == 2 and all(isinstance(v, (int, float)) for v in t):
                # (fx, fy)
                fx_lo, fx_hi = -float(t[0]), float(t[0])
                fy_lo, fy_hi = -float(t[1]), float(t[1])
            else:
                # ((minx,maxx),(miny,maxy))
                (fx_lo, fx_hi), (fy_lo, fy_hi) = t
                fx_lo, fx_hi = float(fx_lo), float(fx_hi)
                fy_lo, fy_hi = float(fy_lo), float(fy_hi)
        else:
            f = float(t)
            fx_lo, fx_hi = -f, f
            fy_lo, fy_hi = -f, f
        return (fx_lo, fx_hi), (fy_lo, fy_hi)

    # vectorized uniform between lo and hi (scalars) -> shape
    def _uniform_vec(lo, hi, shape):
        # lo, hi are Python floats; returns tensor on device/dtype
        return torch.rand(shape, generator=g, device=device, dtype=dtype) * (hi - lo) + lo

    # --- parse ranges once
    a_lo, a_hi = _angle_range(rot_degrees)
    (fx_lo, fx_hi), (fy_lo, fy_hi) = _translate_range(translate_frac)

    if batch_size is None:
        # ----- single-sample (back-compat) -----
        # rotation
        do_rot = torch.rand((), generator=g, device=device) < float(rot_prob)
        if do_rot:
            angle = _uniform_vec(a_lo, a_hi, ())
        else:
            angle = torch.zeros((), device=device, dtype=dtype)

        # translation (in *pixels*)
        do_trans = torch.rand((), generator=g, device=device) < float(trans_prob)
        if do_trans:
            tx = _uniform_vec(fx_lo * W, fx_hi * W, ())
            ty = _uniform_vec(fy_lo * H, fy_hi * H, ())
        else:
            tx = torch.zeros((), device=device, dtype=dtype)
            ty = torch.zeros((), device=device, dtype=dtype)

        translate = torch.stack([tx, ty])  # (2,)
        return angle, translate

    # ----- batched path -----
    B = int(batch_size)

    # per-example Bernoulli for whether we apply rotation/translation
    if rot_prob >= 1.0:
        rot_mask = torch.ones((B,), device=device, dtype=torch.bool)
    elif rot_prob <= 0.0:
        rot_mask = torch.zeros((B,), device=device, dtype=torch.bool)
    else:
        rot_mask = (torch.rand((B,), generator=g, device=device) < float(rot_prob))

    if trans_prob >= 1.0:
        trans_mask = torch.ones((B,), device=device, dtype=torch.bool)
    elif trans_prob <= 0.0:
        trans_mask = torch.zeros((B,), device=device, dtype=torch.bool)
    else:
        trans_mask = (torch.rand((B,), generator=g, device=device) < float(trans_prob))

    # sample raw values
    angle_raw = _uniform_vec(a_lo, a_hi, (B,))
    tx_raw = _uniform_vec(fx_lo * W, fx_hi * W, (B,))
    ty_raw = _uniform_vec(fy_lo * H, fy_hi * H, (B,))

    # apply masks (0 if not applied)
    angle = torch.where(rot_mask, angle_raw, torch.zeros_like(angle_raw))               # (B,)
    tx     = torch.where(trans_mask, tx_raw, torch.zeros_like(tx_raw))                  # (B,)
    ty     = torch.where(trans_mask, ty_raw, torch.zeros_like(ty_raw))                  # (B,)
    translate = torch.stack([tx, ty], dim=1)                                            # (B,2)

    return angle.to(dtype), translate.to(dtype)


# ---------- robust affine wrapper (kept, modernized) ----------

def _affine_try(imgCHW, angle, translate, mode="bilinear", fill=None):
    """
    imgCHW: (C,H,W) tensor
    Tries torchvision TF.affine with different signatures; falls back to grid_sample if needed.
    mode: 'bilinear' or 'nearest'
    """
    # 2) Older signature: resample=..., fillcolor=... (fallback)
    try:
        from PIL import Image  # only for old APIs
        return TF.affine(
            imgCHW,
            angle=angle, translate=translate, scale=1.0, shear=[0.0, 0.0],
            resample=(Image.BILINEAR if mode == "bilinear" else Image.NEAREST),
            fillcolor=fill if fill is not None else 0
        )
    except TypeError:
        pass

    # 3) Minimal older signature: resample only
    try:
        from PIL import Image
        return TF.affine(
            imgCHW,
            angle=angle, translate=translate, scale=1.0, shear=[0.0, 0.0],
            resample=(Image.BILINEAR if mode == "bilinear" else Image.NEAREST)
        )
    except TypeError:
        pass

    # 4) LAST RESORT: pure torch implementation
    C, H, W = imgCHW.shape
    img = imgCHW.unsqueeze(0)  # (1,C,H,W)
    rad = torch.deg2rad(angle)
    c, s = torch.cos(rad), torch.sin(rad)
    # translate (pixels) -> normalized offsets (align_corners=False): 2*dx/W, 2*dy/H
    tx_n = 2.0 * translate[0] / W
    ty_n = 2.0 * translate[1] / H
    theta = torch.tensor([[c, -s, tx_n],
                          [s, c, ty_n]], dtype=img.dtype, device=img.device).unsqueeze(0)  # (1,2,3)
    grid = F.affine_grid(theta, size=img.shape, align_corners=False)
    out = F.grid_sample(
        img, grid,
        mode=("bilinear" if mode == "bilinear" else "nearest"),
        padding_mode="zeros", align_corners=False
    )
    return out.squeeze(0)


def _apply_affine_batch(x, angle, translate, mode="bilinear", fill=0.0):
    """
    Apply rotation + translation to a batch of images.
    
    Args:
        x         : (B,C,H,W) tensor
        angle     : (B,) in degrees OR scalar tensor
        translate : (B,2) in pixels OR (2,) tensor
        mode      : "bilinear" or "nearest"
        fill      : fill value outside boundaries
    Returns:
        x_aug     : (B,C,H,W) transformed
    """
    B, C, H, W = x.shape
    device, dtype = x.device, x.dtype

    # make angle, translate batched
    if angle.ndim == 0:
        angle = angle.view(1).expand(B)
    if translate.ndim == 1:
        translate = translate.view(1, 2).expand(B, 2)

    # convert to radians
    rad = torch.deg2rad(angle)

    c, s = torch.cos(rad), torch.sin(rad)   # (B,)
    tx, ty = translate[:, 0], translate[:, 1]  # (B,)

    # normalize translations from pixels → [-1,1] coords
    tx_n = tx * 2.0 / W
    ty_n = ty * 2.0 / H

    # build theta: (B,2,3)
    theta = torch.zeros((B, 2, 3), device=device, dtype=dtype)
    theta[:, 0, 0] = c
    theta[:, 0, 1] = -s
    theta[:, 0, 2] = tx_n
    theta[:, 1, 0] = s
    theta[:, 1, 1] = c
    theta[:, 1, 2] = ty_n

    # build grid and sample
    grid = F.affine_grid(theta, x.size(), align_corners=False)
    x_warp = F.grid_sample(x, grid, mode=mode, padding_mode="zeros", align_corners=False)

    if fill != 0.0:
        # grid_sample always pads with 0 → if you want nonzero fill, replace outside
        mask = (grid.abs() <= 1).all(dim=-1, keepdim=True)  # (B,H,W,1)
        mask = mask.permute(0, 3, 1, 2).to(dtype)          # (B,1,H,W)
        x_warp = x_warp * mask + fill * (1 - mask)

    return x_warp


# ---------- main augmenter ----------

import torch
import torch.nn.functional as F

class TVPairedAffineNoiseAugmentor:
    def __init__(
        self,
        rot_degrees=10.0,              # float or (min,max) in degrees
        translate_frac=0.07,           # float, (fx,fy), or ((minx,maxx),(miny,maxy))
        apply_noise_prob=1.0,
        noise_pct=(0.01, 0.05),
        fill_value=0.0,
        seed=123,
        device="cpu",
        rot_prob=1.0,                  # prob to apply rotation
        trans_prob=1.0,                # prob to apply translation
        use_autocast=True,             # mixed precision for speed (esp. CUDA)
    ):
        """
        Batched/optimized augmentor:
        - Does K augs per input sample in one call (repeat_interleave along batch)
        - Stays on device (no .item / Python scalars from tensors)
        - Batched Gaussian smoothing and per-example noise magnitudes
        """
        self.rot = rot_degrees
        self.tfrac = translate_frac
        self.noise_pct = tuple(noise_pct)
        self.fill_value = float(fill_value)
        self.apply_noise_prob = float(apply_noise_prob)
        self.device = torch.device(device)
        self.rot_prob = float(rot_prob)
        self.trans_prob = float(trans_prob)
        self.use_autocast = bool(use_autocast)

        # constants/buffers
        self.FIRST4_MIN = -3.14
        self.FIRST4_MAX =  3.14
        self.FIRST4_RANGE = self.FIRST4_MAX - self.FIRST4_MIN
        self.SMALL = 1e-12

        # reproducible per-device RNGs
        self._base_seed = int(seed) if seed is not None else int(torch.seed())
        self._generators = {}  # device -> torch.Generator

        # small perf hint for grid_sample/conv-like ops
        try:
            torch.backends.cudnn.benchmark = True
        except Exception:
            pass

    def _get_gen(self, dev):
        if dev not in self._generators:
            g = torch.Generator(device=dev)
            g.manual_seed(self._base_seed)
            self._generators[dev] = g
        return self._generators[dev]

    @torch.no_grad()
    def __call__(self, x_in, y_in=None, mask_in=None, banana_mask_in=None, channels_last=True, n_augmentations=6):
        """
        Inputs can be BCHW/CHW/HW; returns K augmentations per input.
        Output batch dimension = B*K (if B=1, it’s K).
        """
        # ---- shape/placement ----
        x = _to_bchw(x_in).to(self.device)          # (B,C,H,W)
        B, C, H, W = x.shape
        dev, dt = x.device, x.dtype
        g = self._get_gen(dev)

        if y_in is not None:
            y = _to_bchw(y_in).to(self.device)
        else:
            y = None

        m = None
        if mask_in is not None:
            m = _to_bchw(mask_in).to(self.device)

        bmask = None
        if banana_mask_in is not None:
            bmask = _to_bchw(banana_mask_in).to(self.device)

        # ---- repeat each sample K times (no Python loops) ----
        K = n_augmentations
        x = x.repeat_interleave(K, dim=0)           # (B*K,C,H,W)
        if y is not None:
            y = y.repeat_interleave(K, dim=0)
        if m is not None:
            m = m.repeat_interleave(K, dim=0)
        if bmask is not None:
            bmask = bmask.repeat_interleave(K, dim=0)

        BK = x.shape[0]

        # autocast helps grid_sample + conv-like smoothing a lot on GPU
        from contextlib import nullcontext

        if self.use_autocast and dev.type == "cuda":
            autocast_ctx = torch.cuda.amp.autocast
        else:
            autocast_ctx = nullcontext

        with autocast_ctx():
            # ---- 1) Affine (per-example params) ----
            # REQUIREMENT: make _sample_affine accept batch_size and return:
            #   - angle:     (BK,) in degrees (float tensor on device)
            #   - translate: (BK, 2) in pixels
            angle, translate = _sample_affine(
                self.rot, self.tfrac, H, W, g, x.device,
                rot_prob=self.rot_prob, trans_prob=self.trans_prob,
                batch_size=BK,            # BK = B*K
                dtype=x.dtype,
            )

            x = _apply_affine_batch(x, angle, translate, mode="bilinear", fill=self.fill_value)
            if y is not None:
                y = _apply_affine_batch(y, angle, translate, mode="bilinear", fill=self.fill_value)

            if m is not None:
                m = _apply_affine_batch(m, angle, translate, mode="nearest", fill=0.0)
                m = (m > 0.5).to(dtype=dt)  # keep 0/1 (as float)
            if bmask is not None:
                bmask = _apply_affine_batch(bmask, angle, translate, mode="nearest", fill=0.0)
                bmask = (bmask > 0.5).to(dtype=dt)  # keep 0/1 (as float)

            if y is None:
                # paired case: y tracks x after affine
                y = x.detach().clone()
            
            # ---- 2) Shared Gaussian noise (ROI-aware) ----
            # Probabilistic switch per-example (vectorized)
            if self.apply_noise_prob >= 1.0:
                apply_noise = torch.ones((BK, 1, 1, 1), device=dev, dtype=torch.bool)
            elif self.apply_noise_prob <= 0.0:
                apply_noise = torch.zeros((BK, 1, 1, 1), device=dev, dtype=torch.bool)
            else:
                apply_noise = (torch.rand((BK, 1, 1, 1), generator=g, device=dev) < self.apply_noise_prob)

            # ROI mask (BK,1,H,W) -> broadcast to channels
            if m is not None:
                mx_bool = (m > 0.5).to(torch.bool)
            else:
                mx_bool = torch.ones((BK, 1, H, W), dtype=torch.bool, device=dev)
            mxC = mx_bool.expand(-1, C, -1, -1)  # (BK,C,H,W)

            # One std percentage per example
            low_pct, high_pct = self.noise_pct
            u = torch.rand((BK, 1, 1, 1), generator=g, device=dev, dtype=dt)
            std_pct = u * (high_pct - low_pct) + low_pct  # (BK,1,1,1)

            # Per-channel stds
            # First 4 channels are bounded: same std across those 4
            FIRST4_RANGE = torch.tensor(self.FIRST4_RANGE, device=dev, dtype=dt)
            std_first4 = std_pct * FIRST4_RANGE  # (BK,1,1,1)

            # Last channel (unbounded) std based on dynamic range in ROI (fallback to global)
            ch_last = x[:, -1:, :, :]  # (BK,1,H,W)
            m_last = mx_bool  # ROI per-example (BK,1,H,W)

            # has ROI?
            has_roi = m_last.view(m_last.size(0), -1).any(dim=1, keepdim=True).view(-1,1,1,1)

            # masked min/max with fallback
            pos_inf = torch.tensor(float("inf"), device=dev, dtype=dt)
            neg_inf = torch.tensor(float("-inf"), device=dev, dtype=dt)

            vmin_roi = ch_last.masked_fill(~m_last, pos_inf).amin(dim=(1, 2, 3), keepdim=True)
            vmax_roi = ch_last.masked_fill(~m_last, neg_inf).amax(dim=(1, 2, 3), keepdim=True)
            vmin_all = ch_last.amin(dim=(1, 2, 3), keepdim=True)
            vmax_all = ch_last.amax(dim=(1, 2, 3), keepdim=True)

            vmin_last = torch.where(has_roi, vmin_roi, vmin_all)
            vmax_last = torch.where(has_roi, vmax_roi, vmax_all)

            last_range = (vmax_last - vmin_last).clamp_min(1e-8)  # (BK,1,1,1)
            std_last = std_pct * last_range

            # Build per-channel std tensor
            if C == 5:
                per = torch.cat([std_first4.repeat(1, 4, 1, 1), std_last], dim=1)  # (BK,5,1,1)
            elif C == 9:
                per = torch.cat([std_first4.repeat(1, 8, 1, 1), std_last], dim=1)  # (BK,9,1,1)
            elif C == 2:
                per = torch.cat([std_first4, std_last], dim=1)                     # (BK,2,1,1)
            else:
                # Default: last channel is "unbounded", first (C-1) like bounded
                per = torch.cat([std_first4.repeat(1, C - 1, 1, 1), std_last], dim=1)

            # Make one smooth unit-std noise field per example (BK,1,H,W), then scale per channel
            base = torch.randn((BK, 1, H, W), generator=g, device=dev, dtype=dt)
            base = smooth_noise(base, ks=11, sigma=3.0)                 # should support batched (BK,1,H,W)
            base = base / base.std(dim=(1, 2, 3), keepdim=True).clamp_min(self.SMALL)
            base = base * mx_bool.to(dt)                                # ROI-only
            eps = base.expand(-1, C, -1, -1) * per                      # (BK,C,H,W)

            # tiny tolerance as tensor (dtype-aware), capped vs noise size
            if dt == torch.float64:
                base_tol = torch.tensor(6.28e-8, device=dev, dtype=dt)
            elif dt in (torch.float16, torch.bfloat16):
                base_tol = torch.tensor(6.28e-4, device=dev, dtype=dt)
            else:
                base_tol = torch.tensor(6.28e-6, device=dev, dtype=dt)
            tol_first4 = torch.minimum(base_tol.view(1, 1, 1, 1), 0.1 * std_first4)  # (BK,1,1,1)

            # channels 0..C-2 bounded; channel C-1 unbounded
            if C >= 2:
                x_bounded = x[:, :-1, :, :]
                eps_bounded = eps[:, :-1, :, :]

                hi = (self.FIRST4_MAX - x_bounded) - tol_first4
                lo = (x_bounded - self.FIRST4_MIN) - tol_first4
                headroom = torch.minimum(hi, lo).clamp_min(0.0)

                step_bounded = headroom * torch.tanh(eps_bounded / (headroom + self.SMALL))
                step_bounded = torch.where(mxC[:, :-1, :, :], step_bounded, torch.zeros_like(step_bounded))
                x_bounded_new = (x_bounded + step_bounded).clamp(self.FIRST4_MIN, self.FIRST4_MAX)

                eps_last = eps[:, -1:, :, :]
                step_last = torch.where(mxC[:, -1:, :, :], eps_last, torch.zeros_like(eps_last))
                x_last_new = x[:, -1:, :, :] + step_last

                x = torch.cat([x_bounded_new, x_last_new], dim=1)
            else:
                # Edge case C==1 (treat as unbounded channel)
                step = torch.where(mxC, eps, torch.zeros_like(eps))
                x = x + step

            # optional: zero outside ROI (keep if desired)
            x = torch.where(mxC, x, torch.zeros_like(x))

            # Only apply noise on examples selected by apply_noise;
            # Otherwise keep the affine-only result. (blend per-example)
            if apply_noise.logical_not().any():
                # build mask (BK,1,1,1) -> (BK,C,H,W)
                keep_affine = apply_noise.logical_not().expand(-1, C, H, W)
                # restore the pre-noise state for those (we saved affine-only in y_temp)
                # Trick: we can reconstruct affine-only as y + (x_affine - y) when y==x_affine initially.
                # But simpler: re-run a light path by caching affine output before noise.
                # For performance, we cached nothing; so handle via selective overwrite from y if you rely on paired init.
                # If y was set to x after affine (paired case), then y == affine(x).
                x = torch.where(keep_affine, y, x)

        # ---- Back to original layout ----
        x_out = _from_bchw(x, channels_last)
        y_out = _from_bchw(y, channels_last) if y is not None else None

        m_out = None
        if m is not None:
            m_out = _from_bchw(m.to(dt), channels_last=False, is_mask=True)
        b_out = None
        if banana_mask_in is not None:
            b_out = _from_bchw(bmask.to(dt) if bmask is not None else None, channels_last=False, is_mask=True)

        return x_out, y_out, m_out, b_out

