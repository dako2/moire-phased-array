import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.colors as mcolors
import os

# --- 1. Simulation Setup & Physical Parameters ---
# Wavelength is implicitly 1, so k0 = 2*pi.
k0 = 2 * np.pi

# Aperture parameters
radius = 5.0  # Aperture radius in units of wavelength

# Simulation grid parameters
N = 256  # Grid size (256 is good for speed, 512 for higher quality)
grid_width = radius * 4  # Grid width. Must be > 2*radius for zero-padding.

x = np.linspace(-grid_width / 2, grid_width / 2, N)
y = x.copy()
X, Y = np.meshgrid(x, y)
dx = x[1] - x[0]

# Angular (Direction Cosine) Axes for the far-field pattern
fx = np.fft.fftshift(np.fft.fftfreq(N, d=dx))
u = fx * (2 * np.pi / k0)
v = u.copy()


# --- 2. Amplitude Windows (Taper Functions) ---
# These define the amplitude distribution of the field across the aperture.

def uniform_window(X, Y, radius):
    """Uniform illumination within a circular aperture."""
    return (X**2 + Y**2 <= radius**2).astype(float)

def parabolic_taper(X, Y, radius):
    """Parabolic taper (common in reflector antennas). SLL ~ -25 dB."""
    r = np.sqrt(X**2 + Y**2)
    # Taper is 0 at the edge (r=radius) and 1 at the center (r=0)
    taper = 1 - (r / radius)**2
    taper[r > radius] = 0  # Set to zero outside the aperture
    return taper

def gaussian_taper(X, Y, radius, edge_db=-10):
    """Gaussian taper. Low sidelobes, but inefficient."""
    # Calculate sigma so the taper is 'edge_db' down at r=radius
    # P(r) ~ exp(-r^2/sigma^2), so dB = 10*log10(P) => sigma = r/sqrt(-dB/10*ln(10))
    sigma = radius / np.sqrt(-edge_db / (10 / np.log(10)))
    taper = np.exp(-(X**2 + Y**2) / sigma**2)
    # Apply the circular aperture mask
    taper *= (X**2 + Y**2 <= radius**2).astype(float)
    return taper


# --- 3. Phase Distribution Functions ---
# These define the phase of the field across the aperture.

def phase_steering(X, Y, u_steer, v_steer):
    """Linear phase for beam steering to direction (u, v)."""
    return k0 * (u_steer * X + v_steer * Y)

def phase_defocus(X, Y, beta):
    """Quadratic phase for defocusing the beam."""
    return beta * (X**2 + Y**2)

def phase_oam(X, Y, m):
    """Spiral phase for an Orbital Angular Momentum (OAM) beam."""
    # 'm' is the topological charge (integer for perfect donuts)
    return m * np.arctan2(Y, X)


# --- 4. Simulation Cases ---
# Each case combines a window, a phase function, and animated parameters.
frames = 120
cases = [
    {
        "name": "Uniform - Beam Steering",
        "window_func": uniform_window,
        "phase_func": phase_steering,
        "params": {
            'u_steer': 0.4 * np.cos(np.linspace(0, 2 * np.pi, frames)),
            'v_steer': 0.4 * np.sin(np.linspace(0, 2 * np.pi, frames))
        },
        "plot_params": {"vmin": -40} # Sidelobes are high
    },
    {
        "name": "Parabolic Taper - Beam Steering",
        "window_func": parabolic_taper,
        "phase_func": phase_steering,
        "params": {
            'u_steer': 0.4 * np.cos(np.linspace(0, 2 * np.pi, frames)),
            'v_steer': 0.4 * np.sin(np.linspace(0, 2 * np.pi, frames))
        },
        "plot_params": {"vmin": -60} # Sidelobes are lower
    },
    {
        "name": "Gaussian Taper - Defocusing",
        "window_func": lambda X, Y, r: gaussian_taper(X, Y, r, edge_db=-15),
        "phase_func": phase_defocus,
        "params": {
            # Sweep beta from 0 (focused) to a value and back
            'beta': 0.2 * (1 - np.cos(np.linspace(0, 2 * np.pi, frames)))
        },
        "plot_params": {"vmin": -60}
    },
    {
        "name": "Parabolic Taper - OAM Beam",
        "window_func": parabolic_taper,
        "phase_func": phase_oam,
        "params": {
            # Sweep the OAM mode number 'm'
            'm': np.linspace(-3, 3, frames)
        },
        "plot_params": {"vmin": -60}
    }
]


# --- 5. Main Animation Loop ---
output_dir = './results/tapering'
os.makedirs(output_dir, exist_ok=True)

for case in cases:
    name = case['name']
    window_func = case['window_func']
    phase_func = case['phase_func']
    params = case['params']
    plot_params = case['plot_params']

    # Setup the figure for this case
    fig, (ax_ap, ax_ff) = plt.subplots(1, 2, figsize=(11, 5))
    fig.suptitle(f'Antenna Simulation: {name}', fontsize=16)

    # --- Aperture Field Plot ---
    ax_ap.set_title("Aperture Field (Amp/Phase)")
    # We will use an HSV colormap: Hue=Phase, Value=Amplitude
    hsv_array = np.zeros((N, N, 3))
    hsv_array[..., 1] = 1.0 # Full saturation
    im_ap = ax_ap.imshow(hsv_array, extent=(-grid_width/2, grid_width/2, grid_width/2, -grid_width/2))
    ax_ap.set_xlabel('x / λ')
    ax_ap.set_ylabel('y / λ')
    ax_ap.grid(True, linestyle=':', alpha=0.5)

    # --- Far-Field Pattern Plot ---
    ax_ff.set_title("Far-Field Pattern (dB)")
    im_ff = ax_ff.imshow(np.zeros((N, N)),
                         extent=(u.min(), u.max(), u.max(), u.min()),
                         vmin=plot_params['vmin'], vmax=0, cmap='viridis')
    ax_ff.set_xlabel('u (sinθcosφ)')
    ax_ff.set_ylabel('v (sinθsinφ)')
    # Draw a circle for the visible region boundary (u^2 + v^2 = 1)
    visible_circle = plt.Circle((0, 0), 1, color='white', fill=False, linestyle='--', alpha=0.7)
    ax_ff.add_artist(visible_circle)
    ax_ff.set_xlim(-1.2, 1.2)
    ax_ff.set_ylim(-1.2, 1.2)
    fig.colorbar(im_ff, ax=ax_ff, label='dB')

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    def update(fr):
        # 1. Get current animation parameters
        p = {key: val[fr] for key, val in params.items()}

        # 2. Calculate aperture amplitude and phase
        amplitude = window_func(X, Y, radius)
        phase = phase_func(X, Y, **p)

        # 3. Form the complex aperture field
        aperture_field = amplitude * np.exp(1j * phase)

        # 4. Update the aperture plot
        # Phase (0 to 2pi) -> Hue (0 to 1)
        hsv_array[..., 0] = (phase % (2 * np.pi)) / (2 * np.pi)
        # Amplitude (0 to max) -> Value (0 to 1)
        amp_max = amplitude.max()
        hsv_array[..., 2] = amplitude / amp_max if amp_max > 0 else amplitude
        im_ap.set_data(mcolors.hsv_to_rgb(hsv_array))

        # 5. Calculate and update the far-field pattern
        F = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(aperture_field)))
        P = np.abs(F)**2
        P_max = P.max()
        if P_max > 0:
            P /= P_max
        P_db = 10 * np.log10(P + 1e-12)
        im_ff.set_data(P_db)

        return [im_ap, im_ff]

    # Create and save the animation
    ani = FuncAnimation(fig, update, frames=frames, interval=50, blit=True)
    safe_name = name.replace(' ', '_').replace('-', '').lower()
    output_path = os.path.join(output_dir, f'{safe_name}.gif')

    print(f"Generating animation for '{name}'...")
    ani.save(output_path, writer='imagemagick', fps=20)
    plt.close(fig)

print(f"\nAll animations saved successfully in '{output_dir}'.")