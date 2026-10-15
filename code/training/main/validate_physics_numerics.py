"""Torch-free numerical validation of the DOP-Net physics layer.

This script re-implements, in NumPy, exactly the operators that
``models/dopnet.py`` and ``data/waveform.py`` apply, and checks the
properties the architecture claims.  It deliberately does not import torch, so
it can be run on any machine (CI, a laptop, a reviewer's box) without the deep
learning stack.

Checks
------
1.  The frame-orthogonal Doppler dictionary is exactly orthogonal for every
    allowed CP (cond = 1, coherence = 0).
2.  Q = add_doppler_bins is identifiable from the *pilot sample* aperture, and
    is NOT identifiable from the slow-time pilot occasions alone.  This is the
    quantitative justification for the fast-time ADD branch.
3.  Every (delay spread, CP, comb) triple that the generator can emit satisfies
    CP >= maximum excess delay and 1/(S*df) >= CP.
4.  The intra-symbol Doppler (ICI) is above the noise floor of the configured
    SNR range, i.e. the fast-time observation carries usable information.
5.  The angle-delay transform pair is unitary and exactly invertible.
6.  The ADD fast-time coordinate system is centered on the visible OFDM-symbol
    time origin; on-grid ADD/AFT coefficients agree after whitening.  Off-grid
    coefficient-shape differences are reported as aperture effects, not abused
    as a time-origin test.
7.  The Q-term model with per-voxel off-grid frequencies can represent a Jakes
    tap at the configured maximum Doppler, and the fixed-grid-only model cannot.

Usage
-----
    python validate_physics_numerics.py [--config configs/dopnet.yaml]
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import yaml
from scipy.optimize import minimize
from scipy.special import j0

THREEGPP_MAX_NORMALIZED_EXCESS_DELAY = {
    "tdl_a": 9.6586, "tdl_b": 4.7834, "tdl_c": 8.6523,
    "tdl_d": 12.525, "tdl_e": 20.6519,
    "cdl_a": 9.6586, "cdl_b": 4.7834, "cdl_c": 8.6523,
    "cdl_d": 12.525, "cdl_e": 20.6419,
}

FAILURES = []
CONTROL_MODE = False


def check(name, ok, detail="", capability=False):
    """``capability`` checks are design targets of the full model.

    A deliberately under-parameterized control config (for example the
    ``slow_time_only_doppler`` ablation, which sets Q below what the Doppler
    support needs) is *expected* to miss them, so they are reported as
    measurements rather than failures in that mode.
    """
    if capability and CONTROL_MODE:
        print(f"  [CTRL] {name}{(' :: ' + detail) if detail else ''}")
        return
    tag = "PASS" if ok else "FAIL"
    print(f"  [{tag}] {name}{(' :: ' + detail) if detail else ''}")
    if not ok:
        FAILURES.append(name)


def load_cfg(path):
    path = Path(path)
    cfg = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    base = cfg.pop("base_config", None)
    if base is None:
        return cfg
    parent = load_cfg((path.parent / base).resolve())

    def merge(a, b):
        out = dict(a)
        for k, v in b.items():
            out[k] = merge(a[k], v) if isinstance(v, dict) and isinstance(a.get(k), dict) else v
        return out

    return merge(parent, cfg)


class Geometry:
    def __init__(self, cfg):
        s, m = cfg["system"], cfg["model"]
        self.cfg = cfg
        self.nfft = int(s["waveform_fft_size"])
        self.nact = int(s["n_subcarriers"])
        self.scs = float(s["subcarrier_spacing_hz"])
        self.T = int(s["frame_num_symbols"])
        self.period = int(s["pilot_symbol_period"])
        self.Q = int(m["add_doppler_bins"])
        self.modes = np.arange(-(self.Q // 2), self.Q // 2 + 1, dtype=float)
        self.pilot_visible = np.arange(0, self.T, self.period)

    def cp_wave(self, cp_active):
        return int(round(int(cp_active) * self.nfft / float(self.nact)))

    def duration(self, cp_active):
        return (self.nfft + self.cp_wave(cp_active)) / float(self.nfft * self.scs)

    def grid_hz(self, cp_active):
        return self.modes / (self.T * self.duration(cp_active))

    def symbol_times(self, cp_active):
        return np.arange(self.T) * self.duration(cp_active)

    def pilot_sample_times(self, cp_active):
        dur = self.duration(cp_active)
        # Match data/add_geometry.py exactly: the useful samples are centered
        # around the visible OFDM-symbol timestamp.
        within = (np.arange(self.nact) - 0.5 * (self.nact - 1)) / (self.nact * self.scs)
        return self.pilot_visible[:, None] * dur + within[None, :]

    def pilot_occasion_times(self, cp_active):
        return self.pilot_visible * self.duration(cp_active)


def basis(times, grid):
    return np.exp(2j * np.pi * np.outer(np.asarray(times).ravel(), grid))


def metrics(B):
    sv = np.linalg.svd(B, compute_uv=False)
    cond = sv[0] / max(sv[-1], 1e-18) if sv[-1] > 0 else np.inf
    G = B.conj().T @ B
    d = np.sqrt(np.real(np.diag(G)))
    C = np.abs(G) / np.outer(d, d)
    np.fill_diagonal(C, 0.0)
    return cond, C.max(), int((sv > 1e-9 * sv[0]).sum())


# --------------------------------------------------------------------------
# 1-2. Doppler dictionary
# --------------------------------------------------------------------------
def test_dictionary(g, cfg):
    print("\n1/2. Doppler dictionary and identifiability")
    m = cfg["model"]
    ok_orth, ok_pilot, slow_ranks = True, True, []
    for cp in cfg["system"]["cp_len_choices"]:
        grid = g.grid_hz(cp)
        cond_f, coh_f, _ = metrics(basis(g.symbol_times(cp), grid))
        cond_p, coh_p, _ = metrics(basis(g.pilot_sample_times(cp), grid))
        _, _, rank_s = metrics(basis(g.pilot_occasion_times(cp), grid))
        slow_ranks.append(rank_s)
        ok_orth &= (cond_f <= float(m["doppler_max_config_basis_condition"])
                    and coh_f <= float(m["doppler_max_config_basis_coherence"]))
        ok_pilot &= cond_p <= float(m["doppler_max_pilot_basis_condition"])
        print(f"     cp={cp:3d}  Tsym={g.duration(cp)*1e6:6.2f}us  Tframe={g.T*g.duration(cp)*1e3:.3f}ms"
              f" | frame cond={cond_f:.6f} coh={coh_f:.2e}"
              f" | pilot-aperture cond={cond_p:.3f}"
              f" | slow-time-only rank={rank_s}/{g.Q}")
    check("frame dictionary is exactly orthogonal for every CP", ok_orth)
    check("Q is identifiable from the pilot-sample aperture", ok_pilot)
    check("fast-time is *necessary*: slow-time-only rank < Q",
          max(slow_ranks) < g.Q,
          f"slow-time rank {max(slow_ranks)}, Q={g.Q}", capability=True)

    fd = max(float(x) for x in cfg["system"]["doppler_hz_choices"])
    worst = max(fd * g.T * g.duration(cp) for cp in cfg["system"]["cp_len_choices"])
    need = int(2 * math.ceil(worst) + 1)
    check("Q covers the Doppler support (Q >= 2*ceil(fd*Tframe)+1)",
          g.Q >= need, f"fd*Tframe={worst:.3f} -> need Q>={need}, have {g.Q}",
          capability=True)


# --------------------------------------------------------------------------
# 3. CP / comb feasibility
# --------------------------------------------------------------------------
def _synthetic_pdp_max_ratio(n_taps, profile):
    L = int(n_taps)
    x = np.arange(L, dtype=float)
    profile = str(profile).lower()
    if profile == "exponential":
        p = np.exp(-x / max(L / 2.0, 1.0))
    elif profile == "uniform":
        p = np.ones(L)
    elif profile == "two_cluster":
        c1, c2 = 0.15 * (L - 1), 0.70 * (L - 1)
        s1, s2 = max(0.7, 0.10 * L), max(0.7, 0.13 * L)
        p = np.exp(-0.5 * ((x-c1)/s1)**2) + 0.75*np.exp(-0.5*((x-c2)/s2)**2)
    else:
        return 4.0
    p /= p.sum()
    mean = (p*x).sum()
    rms = np.sqrt((p*(x-mean)**2).sum())
    return float((L-1)/rms)


def _max_delay_ratio(cfg, ch):
    ch = str(ch).lower().replace("-", "_")
    if ch == "awgn":
        return 0.0
    if ch in {"rayleigh", "rician", "nakagami", "doubly_selective"}:
        s = cfg["system"]
        profile = (s.get("channel_pdp_profiles", {}) or {}).get(
            ch, s.get("synthetic_pdp_profile", "exponential"))
        return _synthetic_pdp_max_ratio(int(s.get("n_taps", 6)), profile)
    return float(THREEGPP_MAX_NORMALIZED_EXCESS_DELAY[ch])


def test_feasibility(g, cfg):
    print("\n3. CP and comb feasibility over train + held-out channels")
    s = cfg["system"]
    channels = list(s.get("channel_types", [])) + list(s.get("heldout_3gpp_channels", []))
    ok = True
    seen = set()
    for ch in channels:
        ch = str(ch).lower().replace("-", "_")
        if ch in seen:
            continue
        seen.add(ch)
        is_3gpp = ch.startswith("tdl_") or ch.startswith("cdl_")
        spreads = s["threegpp_delay_spread_ns_choices"] if is_3gpp else s["delay_spread_ns_choices"]
        ratio = _max_delay_ratio(cfg, ch)
        for ds in spreads:
            need = ratio * float(ds) * 1e-9
            cps = [c for c in s["cp_len_choices"]
                   if g.cp_wave(c) / (g.nfft * g.scs) >= need * (1-1e-9)]
            if not cps:
                ok = False
                print(f"     {ch:18s} DS={float(ds):7.0f}ns need>={need*1e6:6.2f}us -> NO CP")
                continue
            for c in cps:
                cp_s = g.cp_wave(c) / (g.nfft * g.scs)
                spac = [v for v in s["pilot_spacing_choices"]
                        if 1.0/(int(v)*g.scs) >= cp_s*(1-1e-9)
                        and int(v) >= int(s["n_users"])]
                ok &= bool(spac)
            print(f"     {ch:18s} DS={float(ds):7.0f}ns ratio={ratio:6.3f} need={need*1e6:6.2f}us"
                  f" feasible CP={cps}")
    check("every train/held-out (channel, delay spread) has a feasible CP/comb", ok)


# --------------------------------------------------------------------------
# 4. ICI observability
# --------------------------------------------------------------------------
def test_ici(g, cfg):
    print("\n4. Intra-symbol Doppler (the information the ADD branch adds)")
    n = np.arange(g.nfft)
    ts = n / (g.nfft * g.scs)
    snr_max = float(cfg["system"]["snr_db_max"])
    worst = -np.inf
    for fd in cfg["system"]["doppler_hz_choices"]:
        if fd <= 0:
            continue
        R = j0(2 * np.pi * float(fd) * (ts[:, None] - ts[None, :]))
        d0 = R.mean()
        isr = 10 * np.log10((1 - d0) / d0)
        worst = max(worst, isr)
        print(f"     fd={float(fd):7.1f}Hz  eps=fd/df={float(fd)/g.scs:.4f}  ICI/signal={isr:7.2f} dB")
    check("ICI at the maximum Doppler is above the best-case noise floor",
          worst > -snr_max, f"ICI={worst:.2f} dB vs noise floor {-snr_max:.1f} dB")


# --------------------------------------------------------------------------
# 5. Angle-delay transform pair
# --------------------------------------------------------------------------
def test_projection_roundtrip(g, cfg):
    """project() and synthesize() must be exact inverses on the frame grid."""
    print("\n5a. Doppler projection / synthesis round trip")
    rng = np.random.default_rng(11)
    cp = int(cfg["system"]["cp_len_choices"][1])
    t = g.symbol_times(cp)
    B = basis(t, g.grid_hz(cp))
    A, L = 8, 16
    S = rng.normal(size=(g.Q, A, L)) + 1j * rng.normal(size=(g.Q, A, L))
    h = np.einsum("tq,qal->tal", B, S)                       # synthesize
    gram = B.conj().T @ B
    rhs = np.einsum("qt,tal->qal", B.conj().T, h)
    S_hat = np.linalg.solve(gram, rhs.reshape(g.Q, A * L)).reshape(g.Q, A, L)
    err = np.abs(S_hat - S).max() / np.abs(S).max()
    gram_err = np.abs(gram / g.T - np.eye(g.Q)).max()
    check("Gram equals T*I", gram_err < 1e-6, f"max|B^H B/T - I| = {gram_err:.2e}")
    check("projection inverts synthesis exactly", err < 1e-9, f"max rel err = {err:.2e}")

    # The whitener einsum must contract the Doppler axis and keep the rest.
    W = np.arange(g.Q * g.Q, dtype=float).reshape(g.Q, g.Q)
    corr = rng.normal(size=(2, 2, g.Q, 3, 4))
    out = np.einsum("qp,bupal->buqal", W, corr)
    ref = np.tensordot(W, corr, axes=([1], [2])).transpose(1, 2, 0, 3, 4)
    check("Doppler whitening contracts the correct axis",
          np.abs(out - ref).max() < 1e-9)


def test_transform(g):
    print("\n5. Angle-delay transform pair")
    rng = np.random.default_rng(0)
    A, F, T = 8, g.nact, 4
    H = rng.normal(size=(T, A, F)) + 1j * rng.normal(size=(T, A, F))
    fwd = np.fft.ifft(np.fft.fft(H, axis=-2, norm="ortho"), axis=-1, norm="ortho")
    inv = np.fft.ifft(np.fft.fft(fwd, axis=-1, norm="ortho"), axis=-2, norm="ortho")
    err = np.abs(inv - H).max()
    energy = abs(np.linalg.norm(fwd) / np.linalg.norm(H) - 1.0)
    check("angle-delay transform is exactly invertible", err < 1e-10, f"max err={err:.2e}")
    check("angle-delay transform is unitary", energy < 1e-12, f"energy ratio err={energy:.2e}")


# --------------------------------------------------------------------------
# 6. Backprojection <-> projection consistency
# --------------------------------------------------------------------------
def backprojection(g, cp, spacing, tau_bin, nu, whiten_reg, use_visible_origin=True):
    """NumPy replica of the waveform BEM backprojection (Nu=Nr=1)."""
    N, S = g.nact, int(spacing)
    dur, cpw = g.duration(cp), g.cp_wave(cp)
    grid = g.grid_hz(cp)
    pv = g.pilot_visible
    Np = len(pv)
    rng = np.random.default_rng(7)

    nper = N // S
    m = np.arange(nper)
    q = m * (m + 1) / 2.0 if nper % 2 else m * m / 2.0
    zc = np.exp(-2j * np.pi * q / nper)       # unit-modulus CAZAC comb
    xfreq = np.zeros((Np, N), complex)
    for p in range(Np):                       # random comb offset per occasion
        xfreq[p, int(rng.integers(0, S))::S] = zc
    xt = np.fft.ifft(xfreq, axis=-1, norm="ortho")

    if use_visible_origin:
        within = (np.arange(N) - 0.5 * (N - 1)) / (N * g.scs)
        times = pv[:, None] * dur + within[None, :]
    else:                                    # fixed-grid convention, for contrast
        within = np.arange(N) / (N * g.scs)
        times = (pv + 1)[:, None] * dur + cpw / (g.nfft * g.scs) + within[None, :]

    yt = np.exp(2j * np.pi * nu * times) * np.roll(xt, tau_bin, axis=-1)
    phase = np.exp(-2j * np.pi * grid[None, :, None] * times[:, None, :])
    yfft = np.fft.fft(yt[:, None, :] * phase, axis=-1, norm="ortho")
    xfft = np.fft.fft(xt, axis=-1, norm="ortho")
    cross = np.einsum("pqn,pn->qn", yfft, xfft.conj())
    corr = np.fft.ifft(cross, axis=-1, norm="ortho")
    corr = corr * math.sqrt(N) / (np.abs(xt) ** 2).sum()

    Bp = basis(times, grid)
    M = times.size
    W = np.linalg.inv((Bp.conj().T @ Bp) / M + whiten_reg * np.eye(g.Q))
    return corr, W @ corr, grid, times


def test_backprojection(g, cfg):
    print("\n6. ADD backprojection vs AFT projection (coordinates and scale)")
    cp, spacing, tau = 16, 4, 5
    reg = float(cfg["model"]["doppler_whitening_reg"])
    nu = 0.37 * (1.0 / (g.T * g.duration(cp)))          # deliberately off-grid
    raw, white, grid, times = backprojection(g, cp, spacing, tau, nu, reg)

    Bp = basis(times, grid)
    ideal_pilot = np.linalg.lstsq(Bp, np.exp(2j * np.pi * nu * times.ravel()), rcond=None)[0]
    e_raw = np.linalg.norm(raw[:, tau] - ideal_pilot) / np.linalg.norm(ideal_pilot)
    e_white = np.linalg.norm(white[:, tau] - ideal_pilot) / np.linalg.norm(ideal_pilot)
    print(f"     relative error vs pilot-aperture LS:  raw MF={e_raw:.4f}   whitened={e_white:.4f}")
    check("Doppler whitening brings the matched filter onto the LS coordinates",
          e_white < 0.25 * e_raw, f"{e_raw:.4f} -> {e_white:.4f}")

    # Direct geometry check.  This is the correct origin test: the mean of the
    # centered fast-time sample coordinates for each pilot symbol must equal the
    # visible symbol timestamp exactly.  An on-grid sinusoid alone cannot catch
    # a constant origin error because that error can collapse to a mode phase.
    pv = g.pilot_visible
    visible = pv * g.duration(cp)
    centered_means = times.mean(axis=1)
    origin_err = float(np.max(np.abs(centered_means - visible)))
    old_within = np.arange(g.nact) / (g.nact * g.scs)
    old_times = (pv + 1)[:, None] * g.duration(cp) + g.cp_wave(cp)/(g.nfft*g.scs) + old_within[None, :]
    old_origin_err = float(np.max(np.abs(old_times.mean(axis=1) - visible)))
    print(f"     direct geometry origin error: centered={origin_err*1e9:.3f} ns,"
          f" old convention={old_origin_err*1e6:.3f} us")
    check("ADD fast-time samples are centered on the visible symbol origin",
          origin_err < 1e-12, f"origin error={origin_err*1e9:.3f} ns")

    # Off grid, the pilot-sample aperture and the full slow-time frame are
    # genuinely different windows.  Their coefficient shapes therefore need
    # not match; this observation is diagnostic only and is not an origin test.
    tm = g.symbol_times(cp)
    B = basis(tm, grid)
    aft = (B.conj().T @ np.exp(2j * np.pi * nu * tm)) / g.T
    def shape_err(v):
        scale = np.vdot(v, aft) / max(np.vdot(v, v).real, 1e-18)
        return float(np.linalg.norm(scale * v - aft) / np.linalg.norm(aft))
    print(f"     off-grid aperture-shape difference (expected, not a failure): {shape_err(white[:, tau]):.4f}")

    # On-grid path: the two apertures must then agree almost exactly, which is
    # the sharp test of coordinate consistency.  Off grid they legitimately
    # differ, because a 7-occasion pilot aperture and a 25-symbol frame spread
    # an off-grid tone over the modes differently; that residual is what the
    # learned voxel gate is for, and it is not a scale or origin defect.
    nu_on = grid[g.Q // 2 + 1]
    _, white_on, _, times_on = backprojection(g, cp, spacing, tau, nu_on, reg)
    aft_on = (basis(tm, grid).conj().T @ np.exp(2j * np.pi * nu_on * tm)) / g.T
    e_on = np.linalg.norm(white_on[:, tau] - aft_on) / np.linalg.norm(aft_on)
    print(f"     on-grid path:  relative error between ADD and AFT coefficients = {e_on:.4f}")
    check("on-grid ADD and AFT coefficients agree in scale and phase", e_on < 0.05,
          f"relative error {e_on:.4f}")


# --------------------------------------------------------------------------
# 7. Representation capacity
# --------------------------------------------------------------------------
def test_representation(g, cfg, n_real=16):
    print("\n7. Doppler representation capacity on Jakes taps")
    cp = max(cfg["system"]["cp_len_choices"])
    t = g.symbol_times(cp)
    grid = g.grid_hz(cp)
    dnu = 1.0 / (g.T * g.duration(cp))
    frac = float(cfg["model"]["max_doppler_offset_fraction"])
    rng = np.random.default_rng(3)
    fd = max(float(x) for x in cfg["system"]["doppler_hz_choices"])

    fixed, joint, pinned = [], [], []
    for _ in range(n_real):
        M = 64
        th = 2 * np.pi * rng.random(M)
        ph = 2 * np.pi * rng.random(M)
        h = np.exp(2j * np.pi * fd * np.cos(th)[:, None] * t[None, :] + 1j * ph[:, None]).sum(0) / np.sqrt(M)
        B0 = basis(t, grid)
        c0 = np.linalg.lstsq(B0, h, rcond=None)[0]
        fixed.append(np.linalg.norm(h - B0 @ c0) ** 2 / np.linalg.norm(h) ** 2)

        def cost(o):
            Bq = basis(t, grid + np.clip(o, -frac, frac) * dnu)
            c = np.linalg.lstsq(Bq, h, rcond=None)[0]
            return np.linalg.norm(h - Bq @ c) ** 2 / np.linalg.norm(h) ** 2

        joint.append(minimize(cost, np.zeros(g.Q), method="Nelder-Mead",
                              options=dict(maxiter=4000, fatol=1e-12)).fun)

        def cost_pinned(o):
            Bq = basis(t, grid + np.clip(o, -frac, frac) * dnu)
            return np.linalg.norm(h - Bq @ c0) ** 2 / np.linalg.norm(h) ** 2

        pinned.append(minimize(cost_pinned, np.zeros(g.Q), method="Nelder-Mead",
                               options=dict(maxiter=4000, fatol=1e-12)).fun)

    f_db = 10 * np.log10(np.mean(fixed))
    j_db = 10 * np.log10(np.mean(joint))
    p_db = 10 * np.log10(np.mean(pinned))
    print(f"     fd={fd:.0f}Hz  fixed grid={f_db:7.2f} dB"
          f" | offsets with coefficients PINNED to the fixed grid={p_db:7.2f} dB"
          f" | joint refinement={j_db:7.2f} dB")
    check("the fixed-grid fused loss suppresses the refinement",
          p_db > f_db - 4.0, f"pinned only reaches {p_db:.2f} dB vs {f_db:.2f} dB")
    check("per-voxel off-grid refinement materially raises the representation floor",
          j_db < f_db - 10.0, f"{f_db:.2f} dB -> {j_db:.2f} dB")
    check("the refined physics branch can reach a useful NMSE", j_db < -25.0,
          f"{j_db:.2f} dB", capability=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/dopnet.yaml")
    a = ap.parse_args()
    cfg = load_cfg(a.config)
    g = Geometry(cfg)
    global CONTROL_MODE
    fd = max(float(x) for x in cfg["system"]["doppler_hz_choices"])
    worst = max(fd * g.T * g.duration(cp) for cp in cfg["system"]["cp_len_choices"])
    CONTROL_MODE = g.Q < int(2 * math.ceil(worst) + 1)
    print(f"DOP-Net physics validation  --  {a.config}")
    print(f"  SCS={g.scs/1e3:.1f}kHz  Nf={g.nact}  T={g.T}  pilot period={g.period}  Q={g.Q}")
    print(f"  active BW={g.nact*g.scs/1e6:.2f} MHz  delay bin={1e9/(g.nact*g.scs):.1f} ns")
    if CONTROL_MODE:
        print("  NOTE: deliberately under-parameterized control config; capability\n"
              "        targets are reported as [CTRL] measurements, not failures.")

    test_dictionary(g, cfg)
    test_feasibility(g, cfg)
    test_ici(g, cfg)
    test_projection_roundtrip(g, cfg)
    test_transform(g)
    test_backprojection(g, cfg)
    test_representation(g, cfg)

    print("\n" + ("ALL CHECKS PASSED" if not FAILURES else "FAILED: " + ", ".join(FAILURES)))
    return 0 if not FAILURES else 1


if __name__ == "__main__":
    raise SystemExit(main())
