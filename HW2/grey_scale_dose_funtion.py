import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d


d_0 = 850  # nm

# Modelo sigmoidal
def h(D, A, B, D_50, k):
    return A + (B - A) / (
        1 + np.exp(-k * (np.log10(D) - np.log10(D_50)))
    )

# # where:
# Parameter	Meaning
# \(A\)	Minimum normalised thickness
# \(B\)	Maximum normalised thickness
# \(D_{50}\)	Dose at which thickness is halfway between \(A\) and \(B\)
# \(k\)	Steepness of the transition

def d(h, A, B, D_50, k):   #Inverse function to calculate dose from thickness
    return D_50 * (h / (B - h)) ** (1 / (k * np.log(10)))


h_rel = np.array([0.09, 0.42, 0.84, 0.94, 0.955, 0.99, 0.998, 1.0])

dose = np.array([125, 150, 180, 200, 210, 240, 270, 300])

# Ajustar espessura relativa
p0 = [0, 1, 160, 10]

fit_params, covariance = curve_fit(h, dose, h_rel*d_0, p0=p0)

A, B, D_50, k = fit_params

print(f"A    = {A:.4f}")
print(f"B    = {B:.4f}")
print(f"D50  = {D_50:.2f}")
print(f"k    = {k:.4f}")

grid_d = np.linspace(100, 350, 500)
grid_h = h(grid_d, *fit_params)

plt.figure(figsize=(6,4))
plt.scatter(dose, h_rel*d_0, label='Data')
plt.title('Dose vs Thickness (Data + Fit)')
plt.plot(grid_d, grid_h, 'r', label='Fit')
plt.grid(True)
plt.xlabel(r'Dose ($\mu C/cm^2$)')
plt.ylabel(r'Thickness (nm)')
plt.legend()
plt.show()


inverse_h = interp1d(grid_h, grid_d, bounds_error=False, fill_value='extrapolate')

D = inverse_h(0.8)
print(D)

grid_inv_h = np.linspace(100, 800, 500)
grid_inv_d = inverse_h(grid_inv_h)

plt.figure(figsize=(6,4))
plt.title('Necesary Dose vs Desired Thickness')
plt.plot(grid_inv_h, grid_inv_d, 'r', label='Fit')
plt.grid(True)
plt.ylabel(r'Dose ($\mu C/cm^2$)')
plt.xlabel(r'Thickness (nm)')
plt.legend()
plt.show()

r_in = 244/2 #nm
r_out = 176/2 #nm
d_h = 200 #nm
p = 420 #nm
h_margin = 20 + r_in #nm

def height(x, r_in, r_out, dh, p):
    local_x = (x % p) - r_out
    circ_out = np.sqrt(abs(r_out**2 - local_x**2))
    circ_in = -np.sqrt(abs(r_in**2 - (local_x-r_out-r_in)**2))
    if local_x < r_out:
        y =  circ_out + (dh + h_margin)
    else:
        y =  circ_in +  h_margin

    return y

grid_xtest = np.linspace(0, 3*p, 1000)
grid_height = np.array([height(x, r_in, r_out, d_h, p) for x in grid_xtest])


plt.figure(figsize=(10, 5)) # Optional: gives the wider plot plenty of room
plt.plot(grid_xtest, grid_height, 'r', label='Height profile')
plt.fill_between(grid_xtest, grid_height, 0, color='r', alpha=0.2)

# Fix: Call set_aspect on the Axes object via plt.gca()
plt.gca().set_aspect('equal', adjustable='box')

plt.ylim(0, 550)

plt.title('PDMS Height profile')
plt.grid(True)
plt.ylabel(r'Height (nm)')
plt.xlabel(r'Position (nm)')
plt.show()


grid_dose = np.array([inverse_h(850-height(x, r_in, r_out, d_h, p)) for x in grid_xtest])

plt.figure(figsize=(10, 5)) # Optional: gives the wider plot plenty of room
plt.plot(grid_xtest, grid_dose, 'r', label='Dose profile')

plt.title('Medusa 82 Dose Profile [height_to_dose(850 - height_profile(x))]')
plt.grid(True)
plt.ylabel(r'Dose ($\mu C/cm^2$)')
plt.xlabel(r'Position (nm)')
plt.show()

n_per = grid_xtest[-1] / p
dose_per_period = np.trapezoid(grid_dose, grid_xtest) / n_per * 1e-7   # uC/cm: charge per cm of y-length, one period
print(f"Dose per period: {dose_per_period:.3e} uC/cm")
print(f"Mean dose over a period: {dose_per_period / (p * 1e-7):.2f} uC/cm^2")

# Pattern size: periods along x times length along y
L_x_cm = 1.0
L_y_cm = 1.0
N_x_periods = L_x_cm * 1e7 / p
Q_total = N_x_periods * L_y_cm * dose_per_period      # uC
print(f"Periods in x: {N_x_periods:.0f}, total charge: {Q_total:.3e} uC")


#--------AI Aided--------#
#--------vvvvvvvv--------#

from scipy.ndimage import gaussian_filter

n_periods = 3                 # number of grating periods simulated

# ==========================================
# Realistic e-beam exposure: ELS-BODEN 50, 50 kV
# ==========================================



I_beam = 1e-9                # A (1 nA)
FWHM_beam = 2.8                 # nm, Gaussian beam FWHM
sigma_beam = FWHM_beam / (2 * np.sqrt(2 * np.log(2)))
step_x = 2                    # nm, shot pitch along the scan direction
step_y = 2                    # nm, line spacing
t_clk = 5e-9                 # s, minimum dwell time (ELS-BODEN 50)
t_step = 0.1e-9               # s, dwell time resolution (ELS-BODEN 50)
USE_CLK_LIMIT = True          # True: quantise dwell times to t_step (min = t_clk)
n_lines = 24                  # number of y-lines shown in the 2D window
fine = 0.5                    # nm, grid used to compute the delivered dose

# ---- Dwell time per shot ----
n_px = int(round(n_periods * p / step_x))
x_px = (np.arange(n_px) + 0.5) * step_x
dose_px = np.array([inverse_h(850 - height(x, r_in, r_out, d_h, p)) for x in x_px])  # uC/cm^2

t_dwell = dose_px * 1e-6 * (step_x * step_y * 1e-14) / I_beam   # s, t = D * A_pixel / I

if USE_CLK_LIMIT:
    t_used = np.maximum(np.round(t_dwell / t_step) * t_step, t_clk)
else:
    t_used = t_dwell

print(f"Dwell time per shot: min = {t_used.min()*1e9:.4f} ns, max = {t_used.max()*1e9:.4f} ns")
print(f"Pattern generator clock (assumed): {t_clk*1e9:.1f} ns")

# ---- Time for 1 cm^2 ----
area_cm2 = 1.0
N_pix = area_cm2 / (step_x * step_y * 1e-14)
T_ideal = np.mean(dose_px) * 1e-6 * area_cm2 / I_beam
T_clk_limited = N_pix * np.mean(np.maximum(t_dwell, t_clk))
T_used = N_pix * np.mean(t_used)

print(f"Mean dose: {dose_px.mean():.2f} uC/cm^2")
print(f"Number of shots in 1 cm^2: {N_pix:.3e}")
print(f"Exposure time, ideal (Q/I): {T_ideal:.1f} s = {T_ideal/3600:.3f} h")
print(f"Exposure time, clock-limited: {T_clk_limited:.1f} s = {T_clk_limited/3600:.3f} h")
print(f"Exposure time, as simulated: {T_used:.1f} s = {T_used/3600:.3f} h")

# ---- Delivered dose: shots (impulses) convolved with the Gaussian beam ----
r_x = int(round(step_x / fine))
r_y = int(round(step_y / fine))
nx = n_px * r_x
ny = n_lines * r_y

impulses = np.zeros((ny, nx))                       # charge (C) per fine pixel
impulses[r_y // 2::r_y, r_x // 2::r_x] = np.tile(t_used * I_beam, (n_lines, 1))

q_delivered = gaussian_filter(impulses, sigma=sigma_beam / fine, mode=('constant', 'wrap'))
dose_delivered = q_delivered / ((fine * 1e-7) ** 2) * 1e6   # uC/cm^2

x_fine = (np.arange(nx) + 0.5) * fine
dose_applied = dose_delivered[2 * r_y:ny - 2 * r_y].mean(axis=0)   # y-average over whole line pitches, edge rows excluded
# ==========================================
# Plot 1: dwell time per shot + delivered 1D dose
# ==========================================

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7), sharex=True)


ax1.step(x_px - step_x / 2, dose_px, where='post', color='gray', label='Target dose (per shot bin)')
ax1.plot(x_fine, dose_applied, 'r', label='Applied dose')
ax1.set_xlabel('Position (nm)')
ax1.set_ylabel(r'Dose ($\mu C/cm^2$)')
ax1.grid(True)
ax1.legend()

ax2.step(x_px - step_x / 2, t_used * 1e9, where='post', color='k')
ax2.set_ylabel('Dwell time per shot (ns)')
ax1.set_title(f'I = {I_beam*1e12:.0f} pA, {step_x} nm pitch, beam FWHM = {FWHM_beam} nm')
ax2.grid(True)

time_per_period = t_used.sum() / n_periods / step_y * 1e7   # s/cm: time per cm of y-length, one period
print(f"Time per period: {time_per_period:.3e} s/cm")
print(f"Cross-check, ideal Q/I: {dose_per_period * 1e-6 / I_beam:.3e} s/cm")

T_total = N_x_periods * L_y_cm * time_per_period            # s
print(f"Total time for {L_x_cm} x {L_y_cm} cm^2: {T_total:.3e} s = {T_total/3600:.1f} h")


plt.tight_layout()
plt.show()

# ==========================================
# Plot 2: 2D serpentine scan and delivered dose
# ==========================================

extent = [0, n_px * step_x, 0, n_lines * step_y]
dwell_map = np.tile(t_used * 1e9, (n_lines, 1))

path_x, path_y = [], []
for j in range(n_lines):
    xs = x_px if j % 2 == 0 else x_px[::-1]          # serpentine: reverse every other line
    path_x.append(xs)
    path_y.append(np.full_like(xs, (j + 0.5) * step_y))
path_x = np.concatenate(path_x)
path_y = np.concatenate(path_y)

fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True)

im1 = axes[0].imshow(dwell_map, extent=extent, aspect='auto', origin='lower', interpolation='nearest')
axes[0].plot(path_x, path_y, color='w', lw=0.5, alpha=0.7)
axes[0].plot(path_x[0], path_y[0], 'o', color='lime', label='Start')
axes[0].plot(path_x[-1], path_y[-1], 'o', color='red', label='End')
axes[0].set_title('Dwell time per shot (ns) with serpentine path')
axes[0].set_ylabel('Y (nm)')
axes[0].legend(loc='upper right')
fig.colorbar(im1, ax=axes[0], label='Dwell time (ns)')

im2 = axes[1].imshow(dose_delivered, extent=extent, aspect='auto', origin='lower', interpolation='nearest')
axes[1].set_title('Delivered dose (shots convolved with Gaussian beam)')
axes[1].set_xlabel('X (nm)')
axes[1].set_ylabel('Y (nm)')
fig.colorbar(im2, ax=axes[1], label=r'Dose ($\mu C/cm^2$)')

plt.tight_layout()
plt.show()
