import math, torch
import torch.nn.functional as F
import numpy as np  # only used for type checks; not for RNG
from torchvision.transforms import functional as TF


# ---------- helpers ----------

def gaussian_kernel2d(ks, sigma, device, dtype):
    ax = torch.arange(ks, device=device, dtype=dtype) - (ks - 1) / 2
    yy, xx = torch.meshgrid(ax, ax, indexing="ij")
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
    if t is None:
        return None
    t = t.squeeze(0)
    if is_mask:
        return (t.squeeze(0) > 0.5).cpu().numpy()  # boolean mask
    return (t.permute(1, 2, 0) if channels_last else t).cpu().numpy()


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


def _sample_affine(rot_degrees, translate_frac, H, W, g, device, rot_prob=1.0, trans_prob=1.0):
    # angle
    if torch.rand((), generator=g, device=device).item() < float(rot_prob):
        a_lo, a_hi = _angle_range(rot_degrees)
        angle = _uniform(a_lo, a_hi, g, device)
    else:
        angle = 0.0

    # translation (in *pixels*)
    if torch.rand((), generator=g, device=device).item() < float(trans_prob):
        (fx_lo, fx_hi), (fy_lo, fy_hi) = _translate_range(translate_frac)
        tx = _uniform(fx_lo * W, fx_hi * W, g, device)
        ty = _uniform(fy_lo * H, fy_hi * H, g, device)
    else:
        tx, ty = 0.0, 0.0

    return float(angle), (float(tx), float(ty))


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
    rad = math.radians(angle)
    c, s = math.cos(rad), math.sin(rad)
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


def _apply_affine_batch(x, angle, translate, mode="bilinear", fill=None):
    # x: (1,C,H,W)
    out = []
    for b in range(x.shape[0]):
        out.append(_affine_try(x[b], angle, translate, mode=mode, fill=fill))
    return torch.stack(out, 0)


# ---------- main augmenter ----------

class TVPairedAffineNoiseAugmentor:
    def __init__(
            self,
            rot_degrees=10.0,  # accepts float or (min_deg, max_deg)
            translate_frac=0.07,  # accepts float, (fx,fy), or ((minx,maxx),(miny,maxy))
            apply_noise_prob=1.0,
            noise_pct=(0.01, 0.05),
            fill_value=0.0,
            seed=123,
            device="cpu",
            rot_prob=1.0,  # probability to apply rotation
            trans_prob=1.0  # probability to apply translation
    ):
        """
        rot_degrees: float (±rot) or (min,max) in degrees
        translate_frac: float for symmetric frac or (fx,fy) or ((minx,maxx),(miny,maxy))
        noise_pct: (min,max) for Gaussian σ as % of joint (x,y) value range
        """
        # keep as given (not forced to float) to allow ranges
        self.rot = rot_degrees
        self.tfrac = translate_frac

        self.noise_pct = tuple(noise_pct)
        self.fill_value = float(fill_value)
        self.apply_noise_prob = float(apply_noise_prob)
        self.device = device
        self.rot_prob = float(rot_prob)
        self.trans_prob = float(trans_prob)

        # persistent base seed and per-device generators (reproducible sequence across runs)
        self._base_seed = int(seed) if seed is not None else int(torch.seed())
        self._generators = {}  # device -> torch.Generator

    def _get_gen(self, dev):
        # ensure generator's device matches tensors' device
        if dev not in self._generators:
            g = torch.Generator(device=dev)
            g.manual_seed(self._base_seed)
            self._generators[dev] = g
        return self._generators[dev]

    @torch.no_grad()
    def __call__(self, x_in, y_in=None, mask_in=None, banana_mask_in=None, channels_last=True):
        # Move inputs to configured device
        x = _to_bchw(x_in).to(self.device)  # (1,Cx,H,W)
        y = _to_bchw(y_in).to(self.device) if y_in is not None else None
        m = _to_bchw(mask_in).to(self.device) if mask_in is not None else None  # (1,1,H,W)
        b = _to_bchw(banana_mask_in).to(self.device) if banana_mask_in is not None else None  # (1,1,H,W)

        _, _, H, W = x.shape
        dev = x.device
        g = self._get_gen(dev)

        # 1) Affine: sample once, apply to x/y/m/b (bilinear for x,y; nearest for masks)
        angle, translate = _sample_affine(
            self.rot, self.tfrac, H, W, g, dev,
            rot_prob=self.rot_prob, trans_prob=self.trans_prob
        )
        x = _apply_affine_batch(x, angle, translate, mode="bilinear", fill=self.fill_value)
        if y is not None:
            y = _apply_affine_batch(y, angle, translate, mode="bilinear", fill=self.fill_value)

        if m is not None:
            m = _apply_affine_batch(m, angle, translate, mode="nearest", fill=0.0)
            m = (m > 0.5).to(dtype=x.dtype)  # keep strictly 0/1

        if b is not None:
            b = _apply_affine_batch(b, angle, translate, mode="nearest", fill=0.0)
            b = (b > 0.5).to(dtype=x.dtype)  # keep strictly 0/1

        # 2) Shared Gaussian noise (σ as % of joint (x,y) range), added to x and y
        # ---- per-channel limits (first 4 bounded, last unbounded) ----
        FIRST4_MIN = -3.14
        FIRST4_MAX = 3.14
        FIRST4_RANGE = FIRST4_MAX - FIRST4_MIN  # 6.28
        SMALL = 0.000000000001  # 1e-12 as decimal, for safe divides

        #assert x.shape[1] == 5, "Expected C=5 (first 4 bounded, last unbounded)."

        if torch.rand((), generator=g, device=dev).item() < self.apply_noise_prob:
            # --- ROI mask (1,1,H,W) -> broadcast to channels ---
            if m is not None:
                mx_bool = (m > 0.5)
            else:
                mx_bool = torch.ones((1, 1, H, W), dtype=torch.bool, device=x.device)
            mxC = mx_bool.expand(-1, x.shape[1], -1, -1)  # (1,5,H,W)

            # --- one random noise percent from external range (FIXED: use torch.rand) ---
            low_pct, high_pct = self.noise_pct  # e.g., (0.010, 0.030)
            u = torch.rand((), generator=g, device=dev)  # uniform in [0,1)
            std_pct = float(u * (high_pct - low_pct) + low_pct)  # scalar percent

            # --- per-channel stds ---
            # First 4 channels: based on fixed allowed range
            std_first4 = std_pct * FIRST4_RANGE

            # Last channel (unbounded): use its own ROI dynamic range for proportional noise
            ch_last = x[:, -1:]  # (1,1,H,W)
            if mxC[:, -1:].any():
                vmin_last = ch_last[mxC[:, -1:]].amin().item()
                vmax_last = ch_last[mxC[:, -1:]].amax().item()
            else:
                vmin_last = ch_last.amin().item()
                vmax_last = ch_last.amax().item()
            last_range = max(0.00000001, float(vmax_last - vmin_last))  # avoid zero std
            std_last = std_pct * last_range

            if x.shape[1] == 5:
                std_per_ch = torch.tensor(
                    [std_first4, std_first4, std_first4, std_first4, std_last],
                    device=x.device, dtype=x.dtype
                ).view(1, 5, 1, 1)  # (1,5,1,1)
            elif x.shape[1] == 9:
                std_per_ch = torch.tensor(
                    [std_first4, std_first4, std_first4, std_first4,std_first4, std_first4, std_first4, std_first4, std_last],
                    device=x.device, dtype=x.dtype
                ).view(1, 9, 1, 1)  # (1,5,1,1)
            elif x.shape[1] == 2:
                std_per_ch = torch.tensor(
                    [std_first4, std_last],
                    device=x.device, dtype=x.dtype
                ).view(1, 2, 1, 1)  # (1,5,1,1)

            # --- make one smooth unit-std noise field and scale per channel (ROI-only) ---
            def make_smooth_unit(dtype):
                n = torch.randn((1, 1, H, W), generator=g, device=dev, dtype=dtype)
                n = smooth_noise(n, ks=11, sigma=3.0)  # your gaussian smoother
                n = n / n.std().clamp_min(SMALL)  # unit std
                return n * mx_bool.to(dtype)  # ROI-only

            base = make_smooth_unit(x.dtype)  # (1,1,H,W)
            eps = base.expand(-1, x.shape[1], -1, -1) * std_per_ch  # (1,5,H,W), signed noise

            # --- channels 0..3: soft limiter to avoid edge pile-ups, then clamp ---
            # tiny safety buffer (no scientific notation), capped vs noise size
            if x.dtype == torch.float64:
                base_tol = 0.0000000628
            elif x.dtype in (torch.float16, torch.bfloat16):
                base_tol = 0.0006280000
            else:  # torch.float32
                base_tol = 0.0000062800
            tol_first4 = min(base_tol, 0.1 * float(std_first4))

            x4 = x[:, :-1]  # (1,4,H,W)
            eps4 = eps[:, :-1]

            hi = (FIRST4_MAX - x4) - tol_first4  # room up
            lo = (x4 - FIRST4_MIN) - tol_first4  # room down
            headroom = torch.minimum(hi, lo).clamp_min(0.0)

            step4 = headroom * torch.tanh(eps4 / (headroom + SMALL))

            step4 = torch.where(mxC[:, :-1], step4, torch.zeros_like(step4))

            x4_new = (x4 + step4).clamp(FIRST4_MIN, FIRST4_MAX)

            # --- channel 4 (unbounded): add noise inside ROI, no limiter/clamp ---
            eps5 = eps[:, -1:]
            step5 = torch.where(mxC[:, -1:], eps5, torch.zeros_like(eps5))
            x5_new = ch_last + step5

            # --- merge back ---
            x = torch.cat([x4_new, x5_new], dim=1)

            # --- optional: zero outside ROI (keep if you want that behavior) ---
            x = x * mxC.to(x.dtype)

        x_out = _from_bchw(x, channels_last)
        y_out = _from_bchw(y, channels_last) if y is not None else None
        m_out = _from_bchw(m.to(x.dtype) if m is not None else None, channels_last=False, is_mask=True)
        b_out = _from_bchw(b.to(x.dtype) if b is not None else None, channels_last=False, is_mask=True)
        return x_out, y_out, m_out, b_out
