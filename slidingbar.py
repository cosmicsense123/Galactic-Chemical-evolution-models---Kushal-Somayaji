import streamlit as st
import numpy as np
import matplotlib.pyplot as pyplot


def configure_sidebar() -> dict:
    st.sidebar.title("adjust parameters")
    st.sidebar.markdown("Use the sliders below to adjust the parameters of the model.")
    alpha = st.sidebar.slider("Inflow rate (alpha)", min_value=0.0, max_value=1.0, value=0.5, step=0.01)
    nu_eff = st.sidebar.slider("Effective consumtion rate (nu_eff)", min_value=0.0, max_value=1.0, value=0.5, step=0.01)
    lambda_SLR = st.sidebar.number_input("SLR decay rate ($\\lambda_{SLR}$ Myr)", min_value=0.0, max_value=1000.0, value=1.0)
    # keep the input in the sidebar so all controls are together
    yield_ratios = st.sidebar.number_input("Yield ratios ($Y_{SLR}/Y_{Stable}$)", min_value=0.0, max_value=100.0, value=0.0)
    return dict(alpha=alpha, nu_eff=nu_eff, yield_ratios=yield_ratios, lambda_SLR=lambda_SLR)


results = configure_sidebar()


def R_of_t(t, y_ratio, lam, beta):
    exp_bt = np.exp(beta * t)
    exp_mlt = np.exp(-lam * t)
    num = beta * (lam * (exp_bt - 1.0) + beta * (exp_mlt - 1.0))
    den = lam * (lam + beta) * (exp_bt - 1.0 - beta * t)

    # avoid division warnings by masking tiny denominators
    safe_den = np.where(np.abs(den) < 1e-12, np.nan, den)
    ratio = np.divide(num, safe_den, out=np.full_like(num, np.nan), where=~np.isnan(safe_den))
    return y_ratio * ratio


def compute_model(params: dict) -> dict:
    """Compute model outputs using current slider params.

    This function retrieves parameters from `params`, computes the
    time grid and R(t) using `R_of_t`, and returns a serializable
    dictionary with parameters and arrays (as lists) for display.
    """
    # time grid and defaults
    n_pts = 1000
    t_min, t_max = 0.0, 15.0

    # parameters (defaults are fallbacks, but configure_sidebar always provides values)
    alpha = float(params.get("alpha", 1.0))
    nu_eff = float(params.get("nu_eff", 0.4))
    yield_ratios = params.get("yield_ratios", 0.0)
    lambda_SLR = float(params.get("lambda_SLR", 1.0))

    if yield_ratios is None:
        st.error("yield_ratios parameter is required")

    # compute beta from the parameters (use local variables — avoid relying on globals)
    beta = nu_eff - alpha
    if abs(beta) < 1e-12:
        beta = np.sign(beta) * 1e-12 if beta != 0 else 1e-12

    # generate time array and compute R(t)
    t = np.linspace(t_min, t_max, n_pts)
    R = R_of_t(t, yield_ratios, lambda_SLR, beta)

    # compute ratio at a specific time (8 Myr)
    target_t = 8.0
    target_R = R_of_t(np.array([target_t]), yield_ratios, lambda_SLR, beta)[0]

    return {
        "alpha": alpha,
        "nu_eff": nu_eff,
        "yield_ratios": yield_ratios,
        "lambda_SLR": lambda_SLR,
        "beta": beta,
        "t": t.tolist(),
        "R": R.tolist(),
        "R_at_8Myr": float(target_R),
    }


output = compute_model(results)

# display the outputs so they update immediately when sliders move
st.sidebar.markdown("---")
st.sidebar.write("### Model output")
st.sidebar.write({k: v for k, v in output.items() if k in ("alpha", "nu_eff", "yield_ratios", "lambda_SLR", "beta")})

ratio_at_8 = output.get("R_at_8Myr")
if ratio_at_8 is not None and np.isfinite(ratio_at_8):
    st.sidebar.metric("R(t = 8 Myr)", f"{ratio_at_8:.4g}")
else:
    st.sidebar.write("R(t = 8 Myr) is undefined for the current parameters.")

# prepare arrays for plotting
t_arr = np.array(output["t"])
R_arr = np.array(output["R"])

fig = pyplot.figure(figsize=(8, 5))
pyplot.plot(t_arr, R_arr)
pyplot.xlabel('time (Gyr)')
pyplot.ylabel(r'$\mathcal{R}(t)$')
title = (
    f"SLR/stable ratio with y_SLR/y_st={output['yield_ratios']}, "
    f"lambda={output['lambda_SLR']}, alpha={output['alpha']}, nu_eff={output['nu_eff']} ( beta={output['beta']:.3g} )"
)
pyplot.title(title)
pyplot.grid(True)
pyplot.yscale("log")

st.pyplot(fig)


# also show a main-page summary
st.title("Model interactive preview")
st.write("Use the sidebar sliders to change parameters; output updates live.")
st.json(output)

if ratio_at_8 is not None and np.isfinite(ratio_at_8):
    st.write(f"R(8 Myr) = {ratio_at_8:.6g}")
else:
    st.write("R(8 Myr) is undefined for the current parameters.")