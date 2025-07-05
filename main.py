import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# --- 1. Grid and Physical Parameters ---
# Wavelength is implicitly 1, so k0 = 2*pi.
k0 = 2 * np.pi

# Aperture parameters
radius = 5.0  # Aperture radius in units of wavelength

# Simulation grid parameters
N = 512  # Increased grid size for better resolution

# IMPORTANT FIX: Added Zero-Padding.
# The grid size should be larger than the aperture to avoid aliasing and
# to get a higher resolution (interpolated) far-field pattern.
# We make the grid 2x the size of the aperture diameter.
grid_width = radius * 8

x = np.linspace(-grid_width / 2, grid_width / 2, N)
y = x.copy()
X, Y = np.meshgrid(x, y)
dx = x[1] - x[0]

# --- 2. Phase Generators ---
phi_r = np.pi / 4

# IMPORTANT FIX: Reduced phase gradient 'ks'.
# The original value ks = 10*k0 was far too high. For a wave to radiate,
# the phase gradient |∇Φ| must be <= k0. The original value steered the
# beam to non-physical angles (u,v >> 1). A value like k0/2 is reasonable.
ks = k0 / 2.0

def gb_linear(X, Y):
    return ks * (np.cos(phi_r) * X + np.sin(phi_r) * Y)

a_spiral, b_spiral = 1.0, 2.0
def gb_archimedean(X, Y):
    r = np.sqrt(X**2 + Y**2)
    theta = np.arctan2(Y, X)
    return a_spiral * r + b_spiral * theta

# IMPORTANT FIX: Reduced quadratic coefficient 'beta'.
# The original value beta=0.5 also created too steep a phase ramp when translated.
def gb_quadratic(X, Y, beta=0.05):
    return beta * k0 * (X**2 + Y**2) # Simplified k0 dependency for clarity

def gb_spherical(X, Y, f=20.0): # Increased focal length
    return k0 * np.sqrt(X**2 + Y**2 + f**2)

# --- 3. Transforms + Parameter Sweep ---

# 3) Transforms + param sweep
frames = 100
transforms = [
    ("Rotate",   lambda X,Y,a: (np.cos(a)*X - np.sin(a)*Y, np.sin(a)*X + np.cos(a)*Y),
                 {'a': np.linspace(0,2*np.pi,frames)}, gb_linear),

    ("Translate (spherical)",   # move the feed laterally under a spherical gb
        lambda X,Y,dx,dy: (X - dx, Y - dy),
        {
            'dx': 2.0 * np.cos(np.linspace(0, 2*np.pi, frames)), # Exaggerated range
            'dy': 2.0 * np.sin(np.linspace(0, 2*np.pi, frames))  # Exaggerated range
        },
        gb_spherical
    ),

    ("Translate",lambda X,Y,dx,dy: (X-dx, Y-dy),
                 {'dx':2.0*np.cos(np.linspace(0,2*np.pi,frames)), # Exaggerated range
                  'dy':2.0*np.sin(np.linspace(0,2*np.pi,frames))}, # Exaggerated range
                 gb_quadratic),
    ("Shear",    lambda X,Y,s: (X+s*Y, Y),
                 {'s': np.linspace(-3,3,frames)}, gb_linear), # Exaggerated range
    ("Scale",    lambda X,Y,a,b: (a*X, b*Y),
                 {'a':1+0.5*np.cos(np.linspace(0,2*np.pi,frames)),
                  'b':1+0.5*np.sin(np.linspace(0,2*np.pi,frames))},
                 gb_linear),
    ("Bend",     lambda X,Y,R: (X, R*np.arcsin(np.clip(Y/R,-1,1))),
                 {'R':np.linspace(1.2,3,frames)}, gb_linear),
    ("Twist",    lambda X,Y,tau: (X*np.cos(tau*Y)-Y*np.sin(tau*Y),
                                  X*np.sin(tau*Y)+Y*np.cos(tau*Y)),
                 {'tau':np.linspace(-2,2,frames)}, gb_archimedean) # Reduced range
]

# --- 4. Circular Aperture Window ---
window = (X**2 + Y**2 <= radius**2).astype(float)

# --- 5. Angular (Direction Cosine) Axes ---
# u = fx * lambda. Since k0 = 2*pi/lambda, this is u = fx * 2*pi/k0
fx = np.fft.fftshift(np.fft.fftfreq(N, d=dx))
u = fx * (2 * np.pi / k0)
v = u.copy()

# --- 6. Compute Pattern Function ---
def compute_pattern_db(T, params, gb):
    gs = gb(X, Y)
    Xm, Ym = T(X, Y, **params)
    gm = gb(Xm, Ym)
    # The aperture field has a phase difference that steers/shapes the beam
    A = window * np.exp(1j * (gs - gm))
    # The far-field is the Fourier Transform of the aperture field
    F = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(A)))
    P = np.abs(F)**2
    # Normalize and convert to dB, adding a small number to avoid log(0)
    P_max = P.max()
    if P_max > 0:
        P /= P_max
    return 10 * np.log10(P + 1e-12)

# --- 7. Setup Figure ---
fig, axes = plt.subplots(len(transforms), 2, figsize=(8, 3 * len(transforms)))
moire_ims, beam_ims = [], []
for (ax_m, ax_b), (title, _, _, gb) in zip(axes, transforms):
    ax_m.set_title(f"{title} moiré"); ax_m.axis('off')
    im_m = ax_m.imshow(np.zeros((N, N, 3)),
                       extent=(-2, 2, -2, 2),
                       origin='lower')
    moire_ims.append((im_m, gb))

    ax_b.set_title(f"{title} beam"); ax_b.set_xlabel('u'); ax_b.set_ylabel('v')
    im_b = ax_b.imshow(np.zeros((N, N)),
                       extent=(u.min(), u.max(), v.min(), v.max()),
                       origin='lower', vmin=-50, vmax=0, cmap='viridis')
    # IMPORTANT FIX: Limit the view to the visible region u^2+v^2 <= 1
    ax_b.set_xlim(-1, 1)
    ax_b.set_ylim(-1, 1)
    beam_ims.append(im_b)

plt.tight_layout()

# --- 8. Animation Update Function ---
def update(fr):
    artists = []
    for (im_m, gb_m), im_b, (title, T, params, gb_b) in zip(moire_ims, beam_ims, transforms):
        p = {k: v[fr] for k, v in params.items()}

        # Moiré pattern: overlay red=gs, blue=gm
        # FIX: Correctly render both phase profiles for the moire effect
        gs = gb_m(X, Y)
        Xm, Ym = T(X, Y, **p)
        gm = gb_m(Xm, Ym)
        
        # Normalize phases to [0, 1] for color mapping
        gs_norm = (gs % (2 * np.pi)) / (2 * np.pi)
        gm_norm = (gm % (2 * np.pi)) / (2 * np.pi)
        
        rgb = np.zeros((N, N, 3))
        rgb[..., 0] = gs_norm  # Red channel for original phase
        rgb[..., 2] = gm_norm  # Blue channel for transformed phase
        rgb *= window[..., np.newaxis] # Apply window to visualization
        im_m.set_data(rgb)
        artists.append(im_m)

        # Beam pattern
        Pdb = compute_pattern_db(T, p, gb_b)
        im_b.set_data(Pdb)
        artists.append(im_b)
        
    return artists

# Create and save the animation
ani = FuncAnimation(fig, update, frames=frames, interval=50, blit=True)
# You may need to have 'imagemagick' or 'ffmpeg' installed for saving.
ani.save('moire_scan_all.gif', writer='imagemagick', fps=20)
plt.close(fig)

print("Animation 'moire_scan_fixed.gif' saved successfully.")