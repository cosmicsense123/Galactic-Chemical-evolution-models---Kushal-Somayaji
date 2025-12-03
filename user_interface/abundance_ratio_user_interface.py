import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import os, sys
# Make the repo root importable so we can load slr_config.py from the parent dir
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from slr_config import SLR_DATA, lambda_per_Gyr, mean_life_Myr, mass_number


def configure_sidebar() -> dict:
    st.sidebar.title("Adjust parameters")
    st.sidebar.markdown("Pick an isotope pair from the list **or** enter a custom pair.")

    mode = st.sidebar.radio("Parameter mode", ["Pre-calculated", "Custom"], index=0)

    if mode == "Pre-calculated":
        # Build dropdown of available pairs from SLR_DATA
        pair_to_data = {f"{d['slr']}/{d['stable']}": d for d in SLR_DATA}
        pair_options = sorted(pair_to_data.keys())
        selected_pair = st.sidebar.selectbox("Isotope pair", pair_options, index=0)
        d = pair_to_data[selected_pair]

        # Auto-fill from the data file
        y_ratio = float(d['yield_ratio'])
        t_half_myr = float(d['half_life_Myr'])
        lambda_SLR = lambda_per_Gyr(t_half_myr)
        A_SLR = mass_number(d['slr'])
        A_st  = mass_number(d['stable'])
        slr_label = d['slr']
        stable_label = d['stable']

    else:
        # Custom entry: labels, yields, decay parameters
        st.sidebar.subheader("Custom isotope pair")
        slr_label = st.sidebar.text_input("SLR label (e.g., 97Tc)", value="97Tc").strip()
        stable_label = st.sidebar.text_input("Stable label (e.g., 98Mo)", value="98Mo").strip()

        y_ratio = st.sidebar.number_input("Yield ratio P_R/P_S (number)", min_value=0.0, max_value=10.0, value=3.8e-3, step=1e-4, format="%e")

        decay_mode = st.sidebar.radio("Enter decay as:", ["Half-life (Myr)", "Lambda (1/Gyr)"], index=0)
        if decay_mode == "Half-life (Myr)":
            t_half_myr = st.sidebar.number_input("SLR half-life t₁/₂ (Myr)", min_value=0.0, max_value=5000.0, value=2.6, step=0.1)
            lambda_SLR = lambda_per_Gyr(max(t_half_myr, 1e-12))
        else:
            lambda_SLR = st.sidebar.number_input("SLR decay λ (1/Gyr)", min_value=0.0, max_value=1e5, value=266.6, step=0.1)
            # Back-compute half-life only for display
            t_half_myr = (np.log(2.0) / max(lambda_SLR, 1e-12)) * 1e3

        # Mass numbers (pre-fill from labels, allow override)
        try:
            A_SLR_default = mass_number(slr_label)
        except Exception:
            A_SLR_default = 1
        try:
            A_st_default = mass_number(stable_label)
        except Exception:
            A_st_default = 1
        A_SLR = st.sidebar.number_input("A_SLR (mass number)", min_value=1, max_value=300, value=int(A_SLR_default), step=1)
        A_st  = st.sidebar.number_input("A_stable (mass number)", min_value=1, max_value=300, value=int(A_st_default), step=1)

        selected_pair = f"{slr_label}/{stable_label}"

    # Inflow/consumption sliders (common)
    st.sidebar.markdown("---")
    alpha = st.sidebar.slider("Inflow rate α (1/Gyr)", min_value=0.0, max_value=2.0, value=1.0, step=0.01)
    nu_eff = st.sidebar.slider("Effective consumption rate ν_eff (1/Gyr)", min_value=0.0, max_value=2.0, value=0.4, step=0.01)

    st.sidebar.markdown("---")
    # Optional number-abundance conversion
    show_number = st.sidebar.checkbox("Show number ratio (× A_st/A_SLR)", value=False)

    st.sidebar.markdown("---")
    # Plot options
    t_max = st.sidebar.slider("Plot to t_max (Gyr)", min_value=1.0, max_value=15.0, value=8.0, step=0.5)
    yscale = st.sidebar.selectbox("Y-axis scale", ["log", "linear"], index=0)

    # Display the constants (read-only summary)
    with st.sidebar.expander("Pair details", expanded=False):
        st.write({
            "mode": mode,
            "pair": selected_pair,
            "y_ratio (number)": y_ratio,
            "t1/2 (Myr)": t_half_myr,
            "lambda (1/Gyr)": lambda_SLR,
            "A_SLR": A_SLR,
            "A_stable": A_st,
        })

    return dict(
        selected_pair=selected_pair,
        alpha=alpha,
        nu_eff=nu_eff,
        y_ratio=y_ratio,
        lambda_SLR=lambda_SLR,
        show_number=show_number,
        A_SLR=A_SLR,
        A_st=A_st,
        t_max=t_max,
        yscale=yscale,
    )


results = configure_sidebar()


def R_of_t(t, y_ratio, lam, beta):
    """Compute R(t) with numerically stable primitives.

    Formula (Case C):
      R = y_ratio * beta * [ lam*(e^{beta t}-1) + beta*(e^{-lam t}-1) ]
                      / [ lam*(lam+beta)*(e^{beta t}-1 - beta t) ]
    We use expm1 for small arguments and a β→0 limit when |β t| is tiny.
    """
    t = np.asarray(t, dtype=float)

    # Guard: invalid lam or near-singular lam+beta
    if lam <= 0.0:
        return np.full_like(t, np.nan, dtype=float)

    # Stable exponentials
    expm1_bt = np.expm1(beta * t)                               # e^{βt}-1
    exp_mlt = np.where(lam * t > 50.0, 0.0, np.exp(-lam * t))   # e^{-λt} with underflow guard

    denom_core = expm1_bt - beta * t                            # e^{βt}-1-βt
    num = beta * (lam * expm1_bt + beta * (exp_mlt - 1.0))
    den = lam * (lam + beta) * denom_core

    # Main formula where well-conditioned
    R_main = y_ratio * np.divide(num, den, out=np.full_like(t, np.nan), where=np.abs(den) > 1e-15)

    # β→0 limit on a per-point basis when |β t| is very small
    small = np.abs(beta * t) < 1e-6
    if np.any(small):
        # Limit derived via series:
        #   R_lim = y_ratio * [ lam*t + (e^{-lam t} - 1) ] / (0.5 * lam^2 * t^2)
        # which tends to y_ratio as t->0.
        num_lim = (lam * t + (exp_mlt - 1.0))
        den_lim = 0.5 * (lam ** 2) * (t ** 2)
        R_lim = y_ratio * np.divide(num_lim, den_lim, out=np.full_like(t, np.nan), where=np.abs(den_lim) > 0)
        R_main[small] = R_lim[small]

    # Final guards: undefined if lam+beta ~ 0
    R_main = np.where(np.isclose(lam + beta, 0.0, atol=1e-12), np.nan, R_main)

    return R_main


def compute_model(params: dict) -> dict:
    # time grid from UI
    n_pts = 1000
    t_min = 0.0
    t_max = float(params.get("t_max", 8.0))

    # parameters
    alpha = float(params.get("alpha", 1.0))
    nu_eff = float(params.get("nu_eff", 0.4))
    y_ratio = float(params.get("y_ratio", 3.8e-3))
    lambda_SLR = float(params.get("lambda_SLR", 266.6))
    show_number = bool(params.get("show_number", False))
    A_SLR = int(params.get("A_SLR", 97))
    A_st  = int(params.get("A_st", 98))

    # derived
    beta = nu_eff - alpha
    if abs(beta) < 1e-12:
        beta = 1e-12 if beta >= 0 else -1e-12

    # compute curves
    t = np.linspace(t_min, t_max, n_pts)
    R_mass = R_of_t(t, y_ratio, lambda_SLR, beta)
    conv = (A_st / A_SLR) if (show_number and A_SLR > 0) else 1.0
    R_plot = R_mass * conv

    # value at 8 Gyr (if within range)
    target_t = 8.0
    target_R_mass = R_of_t(np.array([target_t]), y_ratio, lambda_SLR, beta)[0]
    target_R_plot = float(target_R_mass * conv) if np.isfinite(target_R_mass) else np.nan

    return {
        "alpha": alpha,
        "nu_eff": nu_eff,
        "y_ratio": y_ratio,
        "lambda_SLR": lambda_SLR,
        "beta": beta,
        "A_SLR": A_SLR,
        "A_st": A_st,
        "show_number": show_number,
        "t": t.tolist(),
        "R_mass": R_mass.tolist(),
        "R_plot": R_plot.tolist(),
        "R_at_8Gyr": target_R_plot,
        "t_max": t_max,
    }


output = compute_model(results)

st.sidebar.markdown("---")
st.sidebar.write("### Model output (current)")
summary = {
    "pair": results.get("selected_pair"),
    "alpha": output["alpha"],
    "nu_eff": output["nu_eff"],
    "beta": output["beta"],
    "y_ratio": output["y_ratio"],
    "lambda_SLR": output["lambda_SLR"],
    "A_SLR": output["A_SLR"],
    "A_st": output["A_st"],
    "show_number": output["show_number"],
}
st.sidebar.write(summary)

ratio_at_8 = output.get("R_at_8Gyr")
if ratio_at_8 is not None and np.isfinite(ratio_at_8):
    label_kind = "Number" if output.get("show_number") else "Mass"
    st.sidebar.metric(f"{label_kind} ratio R(t=8 Gyr)", f"{ratio_at_8:.4g}")
else:
    st.sidebar.write("R(t = 8 Gyr) is undefined for the current parameters.")

# prepare arrays for plotting
t_arr = np.array(output["t"], dtype=float)
R_plot_arr = np.array(output["R_plot"], dtype=float)

# If log scale, mask nonpositive values to avoid warnings
yscale = results.get("yscale", "log")
if yscale == "log":
    mask = R_plot_arr > 0
    t_plot = t_arr[mask]
    R_plot = R_plot_arr[mask]
else:
    t_plot = t_arr
    R_plot = R_plot_arr

fig = plt.figure(figsize=(8, 5))
plt.plot(t_plot, R_plot)
plt.xlabel('time (Gyr)')
label_kind = "Number ratio" if results.get("show_number") else "Mass ratio"
plt.ylabel(r'$\mathcal{R}(t)$' + (" (number)" if results.get("show_number") else " (mass)"))

selected_pair = results.get("selected_pair", "")
title = (
    f"{label_kind} for {selected_pair}: y_SLR/y_st={output['y_ratio']}, "
    f"lambda={output['lambda_SLR']:.3f}, alpha={output['alpha']}, nu_eff={output['nu_eff']} ( beta={output['beta']:.3g} )"
)
plt.title(title)
plt.grid(True)
plt.yscale(yscale)

st.pyplot(fig)


st.title("Model interactive preview")
st.write("Use the sidebar to change parameters; the plot and the 8 Gyr value update live.")

ratio_at_8 = output.get("R_at_8Gyr")
if ratio_at_8 is not None and np.isfinite(ratio_at_8):
    st.write(f"R(8 Gyr) = {ratio_at_8:.6g}" + (" (number)" if output.get("show_number") else " (mass)"))
else:
    st.write("R(8 Gyr) is undefined for the current parameters.")

# Show a compact JSON of arrays length only
st.write({
    "pair": results.get("selected_pair"),
    "t_max": output.get("t_max"),
    "num_points": len(output.get("t", [])),
    "A_SLR": output.get("A_SLR"),
    "A_st": output.get("A_st"),
    "beta": output.get("beta"),
})