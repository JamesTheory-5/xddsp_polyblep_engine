# xddsp_polyblep_engine
```python
"""
polyblep_osc.py

Band-limited PolyBLEP oscillator core in an XDDSP-style functional layout,
implemented with NumPy + Numba.

Features
--------
- Pure functional DSP: state is a tuple, no classes or dicts.
- PolyBLEP for band-limited saw, ramp, square, rect.
- PolyBLAMP for band-limited triangle.
- Parameter smoothing (frequency, pulse width) via one-pole filters.
- Numba JIT (@njit(cache=True, fastmath=True)) on tick() and process().
- Public API:
    polyblep_osc_init(...)
    polyblep_osc_update_state(...)
    polyblep_osc_tick(x, state, params)
    polyblep_osc_process(x, state, params)

State and params are tuples only, suitable for use in a larger functional DSP
graph or XDDSP-style core system.
"""

from math import sin, cos, floor, pi, exp
import numpy as np
from numba import njit

# --------------------------------------
# Waveform IDs (must match external usage)
# --------------------------------------
SINE = 0
COSINE = 1
TRIANGLE = 2
SQUARE = 3
RECTANGLE = 4
SAWTOOTH = 5
RAMP = 6

# polyBLEP / BLAMP orders
L2 = 2
L3 = 3
L4 = 4

TWO_PI = 2.0 * pi


# --------------------------------------
# Helpers (non-jitted)
# --------------------------------------

def midi2freq(note: float) -> float:
    """Convert MIDI note to frequency in Hz."""
    return 440.0 * (2.0 ** ((note - 69.0) / 12.0))


def one_pole_smoothing_alpha(time_ms: float, sample_rate: float) -> float:
    """
    Compute one-pole smoothing coefficient alpha for a given time constant.

    y[n] = y[n-1] + alpha * (x - y[n-1])

    time_ms = 0 disables smoothing (alpha = 1).
    """
    if time_ms <= 0.0:
        return 1.0
    tau = time_ms * 0.001  # seconds
    return 1.0 - exp(-1.0 / (tau * sample_rate))


# --------------------------------------
# PolyBLEP kernels (scalar, jitted)
# --------------------------------------

@njit(cache=True, fastmath=True)
def poly_blep2(t: float, dt: float) -> float:
    """
    Quadratic (order-2) polyBLEP kernel.
    """
    if t < dt:
        x = t / dt
        return 0.5 * x * x - x + 0.5
    elif t > 1.0 - dt:
        x = (t - 1.0) / dt + 1.0
        return -0.5 * x * x + 0.5 * x
    return 0.0


@njit(cache=True, fastmath=True)
def poly_blep3(t: float, dt: float) -> float:
    """
    Cubic (order-3) polyBLEP kernel.
    """
    if t < dt:
        x = t / dt
        return x * x * (x / 3.0 - 0.5) + 1.0 / 6.0
    elif t > 1.0 - dt:
        x = (t - 1.0) / dt + 1.0
        return -x * x * (x / 3.0 + 0.5) + 1.0 / 6.0
    return 0.0


@njit(cache=True, fastmath=True)
def poly_blep4(t: float, dt: float) -> float:
    """
    Quartic (order-4) polyBLEP kernel, 2-sample-wide.
    """
    if t < 2.0 * dt:
        x = t / dt
        return (-0.0416667 * x**4 + 0.333333 * x**3
                - 0.833333 * x**2 + 0.5 * x)
    elif t > 1.0 - 2.0 * dt:
        x = (t - 1.0) / dt + 2.0
        return (0.0416667 * x**4 - 0.333333 * x**3
                + 0.833333 * x**2 - 0.5 * x)
    return 0.0


@njit(cache=True, fastmath=True)
def poly_blamp2(t: float, dt: float) -> float:
    """
    Quadratic-integral polyBLAMP kernel (integral of order-2 polyBLEP).
    """
    if t < dt:
        x = t / dt
        return (x**3) / 6.0 - (x**2) / 2.0 + x / 3.0
    elif t > 1.0 - dt:
        x = (t - 1.0) / dt + 1.0
        return (x**3) / 6.0 + (x**2) / 2.0 + x / 3.0
    return 0.0


@njit(cache=True, fastmath=True)
def poly_blamp3(t: float, dt: float) -> float:
    """
    Cubic-integral polyBLAMP kernel (integral of order-3 polyBLEP).
    """
    if t < dt:
        x = t / dt
        return (x**4) / 12.0 - (x**3) / 6.0 + (x**2) / 12.0
    elif t > 1.0 - dt:
        x = (t - 1.0) / dt + 1.0
        return -(x**4) / 12.0 - (x**3) / 6.0 - (x**2) / 12.0
    return 0.0


@njit(cache=True, fastmath=True)
def poly_blamp4(t: float, dt: float) -> float:
    """
    Quartic-integral polyBLAMP kernel (integral of order-4 polyBLEP).
    """
    if t < 2.0 * dt:
        x = t / dt
        return (-0.0083333 * x**5 + 0.0833333 * x**4
                - 0.277778 * x**3 + 0.333333 * x**2)
    elif t > 1.0 - 2.0 * dt:
        x = (t - 1.0) / dt + 2.0
        return (0.0083333 * x**5 - 0.0833333 * x**4
                + 0.277778 * x**3 - 0.333333 * x**2)
    return 0.0


@njit(cache=True, fastmath=True)
def poly_blep(t: float, dt: float, order: int) -> float:
    """
    Order-dispatching polyBLEP kernel.
    """
    if order == L2:
        return poly_blep2(t, dt)
    elif order == L4:
        return poly_blep4(t, dt)
    # default: L3
    return poly_blep3(t, dt)


@njit(cache=True, fastmath=True)
def poly_blamp(t: float, dt: float, order: int) -> float:
    """
    Order-dispatching polyBLAMP kernel.
    """
    if order == L2:
        return poly_blamp2(t, dt)
    elif order == L4:
        return poly_blamp4(t, dt)
    # default: L3
    return poly_blamp3(t, dt)


# --------------------------------------
# Naive waveforms (scalar, jitted)
# --------------------------------------

@njit(cache=True, fastmath=True)
def naive_triangle(phase: float) -> float:
    """Triangle in [-1, 1] from phase in [0, 1)."""
    y = phase * 4.0
    if y >= 3.0:
        y -= 4.0
    elif y > 1.0:
        y = 2.0 - y
    return y


# --------------------------------------
# Public API: init, update_state, tick, process
# --------------------------------------

def polyblep_osc_init(
    sample_rate: float,
    base_freq_hz: float = 110.0,
    amplitude: float = 1.0,
    pulse_width: float = 0.5,
    waveform: int = SAWTOOTH,
    blep_order: int = L3,
    freq_smooth_ms: float = 5.0,
    pw_smooth_ms: float = 5.0,
    start_phase: float = 0.0,
):
    """
    Initialize PolyBLEP oscillator state and params.

    Returns
    -------
    state : tuple
        (phase, freq_smooth, pw_smooth)
    params : tuple
        (sample_rate, amplitude, waveform, blep_order,
         freq_target, pw_target, freq_alpha, pw_alpha)
    """
    phase = start_phase % 1.0
    freq_smooth = base_freq_hz
    pw_smooth = pulse_width

    freq_alpha = one_pole_smoothing_alpha(freq_smooth_ms, sample_rate)
    pw_alpha = one_pole_smoothing_alpha(pw_smooth_ms, sample_rate)

    state = (phase, freq_smooth, pw_smooth)
    params = (
        sample_rate,
        amplitude,
        float(waveform),
        float(blep_order),
        base_freq_hz,
        pulse_width,
        freq_alpha,
        pw_alpha,
    )
    return state, params


def polyblep_osc_update_state(
    state,
    params,
    base_freq_hz: float = None,
    amplitude: float = None,
    pulse_width: float = None,
    waveform: int = None,
    blep_order: int = None,
    freq_smooth_ms: float = None,
    pw_smooth_ms: float = None,
):
    """
    Functional "setter": returns updated (state, params) without mutating.

    All arguments are optional; pass only the ones you want to change.
    Smoothing times cause recomputation of smoothing coefficients.
    """
    phase, freq_smooth, pw_smooth = state
    (
        sample_rate,
        amp_old,
        waveform_old,
        blep_order_old,
        freq_target_old,
        pw_target_old,
        freq_alpha_old,
        pw_alpha_old,
    ) = params

    if base_freq_hz is None:
        base_freq_hz = freq_target_old
    if amplitude is None:
        amplitude = amp_old
    if pulse_width is None:
        pulse_width = pw_target_old
    if waveform is None:
        waveform = int(waveform_old)
    if blep_order is None:
        blep_order = int(blep_order_old)

    if freq_smooth_ms is None:
        freq_alpha = freq_alpha_old
    else:
        freq_alpha = one_pole_smoothing_alpha(freq_smooth_ms, sample_rate)

    if pw_smooth_ms is None:
        pw_alpha = pw_alpha_old
    else:
        pw_alpha = one_pole_smoothing_alpha(pw_smooth_ms, sample_rate)

    # keep smoothed values where they were, but clamp PW
    pw_smooth = min(0.9999, max(0.0001, pw_smooth))

    new_state = (phase, freq_smooth, pw_smooth)
    new_params = (
        sample_rate,
        amplitude,
        float(waveform),
        float(blep_order),
        base_freq_hz,
        pulse_width,
        freq_alpha,
        pw_alpha,
    )
    return new_state, new_params


@njit(cache=True, fastmath=True)
def polyblep_osc_tick(
    x_fm_hz: float,
    state,
    params,
):
    """
    Compute one sample of the PolyBLEP oscillator.

    Parameters
    ----------
    x_fm_hz : float
        Instantaneous FM offset in Hz (can be 0.0).
    state : tuple
        (phase, freq_smooth, pw_smooth)
    params : tuple
        (sample_rate, amplitude, waveform, blep_order,
         freq_target, pw_target, freq_alpha, pw_alpha)

    Returns
    -------
    y : float
        Output sample.
    new_state : tuple
        Updated state tuple.
    """
    phase, freq_smooth, pw_smooth = state
    (
        sample_rate,
        amplitude,
        waveform_f,
        blep_order_f,
        freq_target,
        pw_target,
        freq_alpha,
        pw_alpha,
    ) = params

    waveform = int(waveform_f)
    blep_order = int(blep_order_f)

    # --- Parameter smoothing ---
    freq_smooth = freq_smooth + freq_alpha * (freq_target - freq_smooth)
    pw_smooth = pw_smooth + pw_alpha * (pw_target - pw_smooth)
    # clamp PW to safe range
    if pw_smooth < 1e-4:
        pw_smooth = 1e-4
    elif pw_smooth > 0.9999:
        pw_smooth = 0.9999

    # --- Instantaneous frequency & phase increment ---
    f_inst = freq_smooth + x_fm_hz
    if f_inst < 0.0:
        f_inst = 0.0
    f_max = 0.25 * sample_rate
    if f_inst > f_max:
        f_inst = f_max

    inc = f_inst / sample_rate

    # --- Waveform generation with PolyBLEP/BLAMP ---
    if waveform == SINE:
        y = sin(TWO_PI * phase)
    elif waveform == COSINE:
        y = cos(TWO_PI * phase)
    elif waveform == SAWTOOTH:
        y = 2.0 * phase - 1.0
        y -= poly_blep(phase, inc, blep_order)
    elif waveform == RAMP:
        y = 1.0 - 2.0 * phase
        y += poly_blep(phase, inc, blep_order)
    elif waveform == SQUARE or waveform == RECTANGLE:
        pw = pw_smooth
        t2 = phase + (1.0 - pw)
        if t2 >= 1.0:
            t2 -= 1.0

        if phase < pw:
            y = 1.0
        else:
            y = -1.0

        y += poly_blep(phase, inc, blep_order) - poly_blep(t2, inc, blep_order)
    elif waveform == TRIANGLE:
        # naive triangle
        y = naive_triangle(phase)
        # BLAMP correction at slope changes
        t1 = phase + 0.25
        if t1 >= 1.0:
            t1 -= 1.0
        t2 = phase + 0.75
        if t2 >= 1.0:
            t2 -= 1.0
        y += 4.0 * inc * (poly_blamp(t1, inc, blep_order) - poly_blamp(t2, inc, blep_order))
    else:
        # default: silence
        y = 0.0

    # --- Apply amplitude ---
    y_out = amplitude * y

    # --- Phase update ---
    phase = phase + inc
    if phase >= 1.0:
        phase = phase - floor(phase)

    new_state = (phase, freq_smooth, pw_smooth)
    return y_out, new_state


@njit(cache=True, fastmath=True)
def polyblep_osc_process(
    x_fm_hz: np.ndarray,
    state,
    params,
):
    """
    Block processing wrapper around polyblep_osc_tick, analogous to lax.scan.

    Parameters
    ----------
    x_fm_hz : np.ndarray, shape (N,)
        Instantaneous FM offsets in Hz per-sample (0 array for no FM).
    state : tuple
        Initial state.
    params : tuple
        Parameter tuple.

    Returns
    -------
    y : np.ndarray, shape (N,)
        Output audio block.
    new_state : tuple
        Final state after processing the block.
    """
    n = x_fm_hz.shape[0]
    out = np.empty(n, dtype=np.float64)
    cur_state = state

    for i in range(n):
        y, cur_state = polyblep_osc_tick(x_fm_hz[i], cur_state, params)
        out[i] = y

    return out, cur_state


# --------------------------------------
# Smoke test / example usage
# --------------------------------------

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    fs = 48000.0
    base_freq = 110.0
    amp = 0.8
    pw = 0.5

    # Init square wave oscillator
    state, params = polyblep_osc_init(
        sample_rate=fs,
        base_freq_hz=base_freq,
        amplitude=amp,
        pulse_width=pw,
        waveform=SQUARE,
        blep_order=L3,
        freq_smooth_ms=5.0,
        pw_smooth_ms=5.0,
        start_phase=0.0,
    )

    # 4 cycles of audio
    nframes = int(fs * 4.0 / base_freq)
    x_fm = np.zeros(nframes, dtype=np.float64)  # no FM

    block, state = polyblep_osc_process(x_fm, state, params)

    t = np.arange(nframes) / fs

    plt.figure(figsize=(12, 4))
    plt.plot(t, block, linewidth=1.5)
    plt.xlim(0, 4.0 / base_freq)  # show the first ~4 cycles
    plt.title(f"PolyBLEP Square Wave ({base_freq} Hz, order L3)")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Amplitude")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # --- Listen example (if sounddevice is available) ---
    try:
        import sounddevice as sd

        print("Playing test tone...")
        sd.play(block, int(fs))
        sd.wait()
    except Exception as e:
        print("sounddevice not available or playback failed:", e)

```
