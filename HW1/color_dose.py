"""
Photonic Shirt Simulator  —  spectral Fabry-Pérot reflection model
======================================================================
Pipeline:
  1. Origin image  — shirt at W=0 (dry); each pixel's dry thickness d_dry is chosen
                     so that its spectral colour matches the origin
  2. EBL dose map  — each pixel's dose is chosen so that its wet colour matches the target
  3. Add water     — shirt at W=1 → shows the target image

Physics (planar cavity, equal mirrors of reflectance R_MIRROR):
  h(dose)      = A·exp(−B·dose) + C        measured wet thickness, 700–4000 µC/cm²
  d(dose, W)   = d_dry + W · (h(dose) − d_dry)
  δ/2          = 2π · n · d / λ
  F            = 4 R / (1 − R)²
  R(λ)         = F sin²(δ/2) / (1 + F sin²(δ/2))          (spectral reflectance)
  colour       = CIE 1931 integral of R(λ), converted to linear sRGB, normalised
                 to equal-energy white, then gamma-encoded
"""

from PIL import Image
import numpy as np
from scipy.spatial import KDTree
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit


#target_path = "/home/linux/MEFT/M1/S1/Nano/HW1/IST.png"
#origin_path = "/home/linux/MEFT/M1/S1/Nano/HW1/NOVA.jpeg"

origin_path = "/home/linux/MEFT/M1/S1/Nano/HW1/test2.jpeg"
target_path = "/home/linux/MEFT/M1/S1/Nano/HW1/test1.jpeg"


# ── Parameters ──────────────────────────────────────────────────────
SIZE     = 256     # px per side after crop + resize
n        = 1.50    # PEDOT:PSS refractive index
R_MIRROR = 0.1   # PLACEHOLDER: Ag mirror reflectance — replace with the paper's value

D_MIN, D_MAX = 700.0, 4000.0   # measured dose range (µC/cm²)
CHUNK        = 512             # pixels processed at once when matching the dose


def fit_swelling(path="dose_data.csv", plot=False):
    df = pd.read_csv(path, sep=None, engine="python", skipinitialspace=True)
    df.columns = df.columns.str.strip()
    df["solvent"] = df["solvent"].str.strip()

    w = df[df["solvent"] == "water"]
    D = w["dose_uC_per_cm2"].to_numpy(dtype=float)
    h = w["height_nm"].to_numpy(dtype=float)

    def f(D, a, b, c):
        return a * np.exp(-b * D) + c

    (a, b, c), _ = curve_fit(f, D, h, p0=[1e3, 1e-3, 150], maxfev=20000)

    resid  = h - f(D, a, b, c)
    r2     = 1 - np.sum(resid**2) / np.sum((h - h.mean())**2)
    rmse   = np.sqrt(np.mean(resid**2))
    print(f"A = {a:.4f}, B = {b:.6g}, C = {c:.4f}  R² = {r2:.4f}  RMSE = {rmse:.1f} nm")

    if plot:
        x = np.linspace(D.min() * 0.95, D.max() * 1.05, 500)
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.scatter(D, h, color="firebrick", s=40, zorder=3, label="water data")
        ax.plot(x, f(x, a, b, c), lw=2, label=f"fit (R² = {r2:.3f})")
        ax.set_xlabel("Dose (µC cm⁻²)")
        ax.set_ylabel("Height (nm)")
        ax.set_title("Water swelling vs. dose: data and fit")
        ax.legend()
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig("fit_plot.png", dpi=150)
        plt.show()

    return a, b, c


A_FIT, B_FIT, C_FIT = fit_swelling()


# ── Dose ↔ wet thickness (measured range only) ───────────────────────
def height(D):
    """Dose (µC/cm²) → wet thickness (nm), clipped to the measured range."""
    return A_FIT * np.exp(-B_FIT * np.clip(D, D_MIN, D_MAX)) + C_FIT


def swelling(D, d_dry):
    """Dose (µC/cm²) and this pixel's dry thickness → swelling (nm) above it."""
    return height(D) - d_dry


# ── CIE 1931 colour (Wyman, Sloan & Shirley 2013 analytic fit) ───────
WL = np.arange(380.0, 700.1, 2.0)   # wavelength grid for the spectrum (nm)


def _g(x, mu, s1, s2):
    t = (x - mu) / np.where(x < mu, s1, s2)
    return np.exp(-0.5 * t * t)


def _cmf(l):
    x = 1.056*_g(l,599.8,37.9,31.0) + 0.362*_g(l,442.0,16.0,26.7) - 0.065*_g(l,501.1,20.4,26.2)
    y = 0.821*_g(l,568.8,46.9,40.5) + 0.286*_g(l,530.9,16.3,31.1)
    z = 1.217*_g(l,437.0,11.8,36.0) + 0.681*_g(l,459.0,26.0,13.8)
    return np.stack([x, y, z], axis=-1)


CMF = _cmf(WL)                                   # (N, 3)
M_XYZ_RGB = np.array([[ 3.2406, -1.5372, -0.4986],
                      [-0.9689,  1.8758,  0.0415],
                      [ 0.0557, -0.2040,  1.0570]])
WHITE = M_XYZ_RGB @ CMF.sum(axis=0)              # equal-energy white, (3,)


def reflectance_to_rgb_lin(R):
    """Spectrum (..., N) → linear RGB (..., 3), normalised to white = 1."""
    xyz = R @ CMF
    return (xyz @ M_XYZ_RGB.T) / WHITE


def _enc(v):
    v = np.clip(v, 0, 1)
    return np.where(v <= 0.0031308, 12.92 * v, 1.055 * v**(1/2.4) - 0.055)


def _dec(v):
    return np.where(v <= 0.04045, v / 12.92, ((v + 0.055) / 1.055)**2.4)


# ── Airy reflectance of the cavity ───────────────────────────────────
def reflectance(d):
    """Cavity thickness d (nm), any shape → spectral reflectance (..., N)."""
    F  = 4 * R_MIRROR / (1 - R_MIRROR)**2
    s2 = np.sin(2 * np.pi * n * d[..., None] / WL)**2
    return F * s2 / (1 + F * s2)


def colour_of(d):
    """Cavity thickness (nm) → linear RGB, shape (..., 3)."""
    return reflectance_to_rgb_lin(reflectance(d))


# ── Load image ───────────────────────────────────────────────────────
def load_square(path, size=SIZE):
    """Load an image, centre-crop to square, resize, return float32 in [0,1] (sRGB)."""
    img  = Image.open(path).convert("RGB")
    w, h = img.size
    side = min(w, h)
    img  = img.crop(((w - side) // 2, (h - side) // 2,
                     (w + side) // 2, (h + side) // 2))
    img  = img.resize((size, size), Image.LANCZOS)
    return np.array(img, dtype=np.float32) / 255.0


# ── Step 1: dry thickness per pixel from the origin colour ───────────
#DRY_GRID = np.arange(1.0, height(D_MIN) + 0.25, 0.5)   # dry thicknesses up to the thickest wet film (nm)
DRY_GRID = np.arange(1.0, 1000 + 0.25, 0.5)   # dry thicknesses up to the thickest wet film (nm)


def match_dry(origin_lin):
    table = colour_of(DRY_GRID)                  # (G, 3)
    _, idx = KDTree(table).query(origin_lin.reshape(-1, 3), workers=-1)
    return DRY_GRID[idx].reshape(origin_lin.shape[:2]).astype(np.float32)


# ── Step 2: dose per pixel from the target colour (W = 1) ────────────
DOSE_GRID = np.linspace(D_MIN, D_MAX, 2001)     # candidate doses (µC/cm²)
WET_GRID  = height(DOSE_GRID)                   # wet thickness of each dose (nm)


def match_dose(d_dry, target_lin):
    table = colour_of(WET_GRID)                  # (J, 3)
    H, W = d_dry.shape
    d_flat = d_dry.reshape(-1)
    T_flat = target_lin.reshape(-1, 3)
    out    = np.empty(d_flat.size, dtype=np.float32)

    for s in range(0, d_flat.size, CHUNK):
        sl  = slice(s, s + CHUNK)
        err = ((T_flat[sl, None, :] - table[None, :, :])**2).sum(axis=-1)
        err = np.where(WET_GRID[None, :] < d_flat[sl, None], np.inf, err)  # cannot shrink when wet
        out[sl] = DOSE_GRID[err.argmin(axis=1)]

    return out.reshape(H, W)


# ── Forward simulate → sRGB colour at any W ──────────────────────────
def simulate(d_dry, dose, W):
    d = d_dry + W * swelling(dose, d_dry)
    return _enc(colour_of(d))


# ── Main ─────────────────────────────────────────────────────────────
if __name__ == "__main__":

    origin = load_square(origin_path)
    target = load_square(target_path)

    d_dry = match_dry(_dec(origin))
    dose  = match_dose(d_dry, _dec(target))

    result_dry  = simulate(d_dry, dose, W=0.0)
    result_half = simulate(d_dry, dose, W=0.5)
    result_full = simulate(d_dry, dose, W=1.0)

    print(f"dose  : {dose.min():.0f} – {dose.max():.0f} µC/cm²")
    print(f"d_dry : {d_dry.min():.1f} – {d_dry.max():.1f} nm")

    fig, axes = plt.subplots(2, 3, figsize=(13, 8))

    top = [(origin,  "origin\n(W = 0 design)"),
           (target,  "target\n(desired at W = 1)"),
           (dose,    "EBL dose map\n(permanent)")]
    bottom = [(result_dry,  "simulated\n(W = 0)"),
              (result_half, "simulated\n(W = 0.5)"),
              (result_full, "simulated\n(W = 1)")]

    for ax, (img, title) in zip(axes[0], top):
        if img.ndim == 2:
            im = ax.imshow(img, cmap="gray", vmin=D_MIN, vmax=D_MAX)
            fig.colorbar(im, ax=ax, fraction=0.046, label="dose (µC/cm²)")
        else:
            ax.imshow(np.clip(img, 0, 1))
        ax.set_title(title)
        ax.axis("off")

    for ax, (img, title) in zip(axes[1], bottom):
        ax.imshow(np.clip(img, 0, 1))
        ax.set_title(title)
        ax.axis("off")

    fig.tight_layout()
    fig.savefig("dose_map.png", dpi=140, bbox_inches="tight")
    print("saved → dose_map.png")
    plt.show()

    wet = d_dry + swelling(dose, d_dry)

    fig2, axes2 = plt.subplots(1, 2, figsize=(10, 4.5))
    for ax, img, title in zip(axes2,
            [d_dry, wet],
            ["dry height (W = 0)", "wet height (W = 1)"]):
        im = ax.imshow(img, cmap="viridis")
        fig2.colorbar(im, ax=ax, fraction=0.046, label="height (nm)")
        ax.set_title(title)
        ax.axis("off")
    fig2.tight_layout()
    fig2.savefig("height_map.png", dpi=140, bbox_inches="tight")
    print("saved → height_map.png")
