# Auto-generated SLR config: yield ratios (number) and half-lives from user's sources.
# Units: half-life in Myr; lambda in 1/Gyr; yield_ratio is P_R/P_S (number)
import math
import numpy as np
import matplotlib.pyplot as plt
import csv

LN2 = math.log(2.0)

SLR_DATA = [
    {
        'slr': '26Al', 'stable': '27Al', 'yield_ratio': 0.003225, 'half_life_Myr': 0.717, 'source': 'single-star (Brad table)',
    },
    {
        'slr': '41Ca', 'stable': '40Ca', 'yield_ratio': 0.003872, 'half_life_Myr': 0.0994, 'source': 'single-star (Brad table)',
    },
    {
        'slr': '36Cl', 'stable': '35Cl', 'yield_ratio': 0.007791, 'half_life_Myr': 0.301, 'source': 'single-star (Brad table)',
    },
    {
        'slr': '60Fe', 'stable': '56Fe', 'yield_ratio': 0.003997, 'half_life_Myr': 2.62, 'source': 'single-star (Brad table)',
    },
    {
        'slr': '53Mn', 'stable': '55Mn', 'yield_ratio': 2.631, 'half_life_Myr': 3.8, 'source': 'single-star (Brad table)',
    },
    {
        'slr': '107Pd', 'stable': '108Pd', 'yield_ratio': 0.0796, 'half_life_Myr': 6.5, 'source': 'single-star (Brad table)',
    },
    {
        'slr': '182Hf', 'stable': '180Hf', 'yield_ratio': 0.08122, 'half_life_Myr': 8.896, 'source': 'single-star (Brad table)',
    },
    {
        'slr': '129I', 'stable': '127I', 'yield_ratio': 0.1271, 'half_life_Myr': 16.14, 'source': 'single-star (Brad table)',
    },
    {
        'slr': '205Pb', 'stable': '204Pb', 'yield_ratio': 0.3991, 'half_life_Myr': 17.3, 'source': 'single-star (Brad table)',
    },
    {
        'slr': '92Nb', 'stable': '93Nb', 'yield_ratio': 0.05087, 'half_life_Myr': 34.7, 'source': 'single-star (Brad table)',
    },
    {
        'slr': '146Sm', 'stable': '144Sm', 'yield_ratio': 0.02236, 'half_life_Myr': 103.5, 'source': 'single-star (Brad table)',
    },
    {
        'slr': '135Cs', 'stable': '133Cs', 'yield_ratio': 0.3814, 'half_life_Myr': 2.3, 'source': 'single-star (Brad table)',
    },
    {
        'slr': '97Tc', 'stable': '98Mo', 'yield_ratio': 0.00384, 'half_life_Myr': 2.6, 'source': 'single-star (Brad email)',
    },
]

def lambda_per_Gyr(half_life_Myr: float) -> float:
    return (LN2 / half_life_Myr) * 1e3

def mean_life_Myr(half_life_Myr: float) -> float:
    return half_life_Myr / LN2

# --- Helpers ------------------------------------------------------------

def mass_number(nuclide_label: str) -> int:
    """Extract leading integer mass number from a label like '26Al' -> 26."""
    s = nuclide_label.strip()
    digits = []
    for ch in s:
        if ch.isdigit():
            digits.append(ch)
        else:
            break
    if not digits:
        raise ValueError(f"No mass number found in label: {nuclide_label}")
    return int("".join(digits))


def R_mass_caseC(t, y_ratio: float, lam_Gyr: float, alpha: float, nu_eff: float):
    """Mass-fraction ratio R(t) = z_SLR/z_st for Case C, numerically stable.

    Parameters
    ----------
    t : float or array-like
        Time in Gyr.
    y_ratio : float
        Production (yield) number ratio P_R/P_S. Note: the model returns a mass ratio;
        convert to number with A_st/A_slr afterwards if desired.
    lam_Gyr : float
        Decay constant of the SLR in 1/Gyr.
    alpha : float
        Inflow decay rate (1/Gyr).
    nu_eff : float
        Effective consumption rate (1/Gyr).

    Returns
    -------
    ndarray or float
        R_mass(t) as array matching t's shape, or a float if t is scalar.
    """
    t = np.asarray(t, dtype=float)
    lam = float(lam_Gyr)
    beta = float(nu_eff - alpha)

    # Guards
    if lam <= 0.0:
        out = np.full_like(t, np.nan, dtype=float)
        return float(out) if out.shape == () else out

    # Underflow-safe exponentials
    expm1_bt = np.expm1(beta * t)                 # e^{βt} - 1
    exp_mlt  = np.where(lam * t > 50.0, 0.0, np.exp(-lam * t))  # e^{-λt}

    # Denominator core and full numerator/denominator
    denom_core = expm1_bt - beta * t              # e^{βt} - 1 - β t
    num = beta * (lam * expm1_bt + beta * (exp_mlt - 1.0))
    den = lam * (lam + beta) * denom_core

    # Main formula
    with np.errstate(divide='ignore', invalid='ignore', over='ignore', under='ignore'):
        R_main = y_ratio * np.divide(num, den, out=np.full_like(t, np.nan), where=~np.isclose(den, 0.0, atol=1e-15))

    # β -> 0 limit (series), apply pointwise when |β t| is tiny
    small = np.abs(beta * t) < 1e-6
    if np.any(small):
        # R_lim = y_ratio * 2 * (lam*t + e^{-lam t} - 1) / (lam^2 * t^2)
        num_lim = (lam * t + (exp_mlt - 1.0))
        den_lim = 0.5 * (lam ** 2) * (t ** 2)
        with np.errstate(divide='ignore', invalid='ignore'):
            R_lim = y_ratio * np.divide(num_lim, den_lim, out=np.full_like(t, np.nan), where=np.abs(den_lim) > 0)
        R_main = np.where(small, R_lim, R_main)

    # Guard singular lam + beta ~ 0
    if np.isclose(lam + beta, 0.0, atol=1e-12):
        R_main[:] = np.nan

    return float(R_main) if R_main.shape == () else R_main


def R_number_from_mass(R_mass, slr_label: str, stable_label: str):
    """Convert mass-ratio to number-ratio by multiplying by A_st/A_slr."""
    A_slr = mass_number(slr_label)
    A_st  = mass_number(stable_label)
    return np.asarray(R_mass, dtype=float) * (A_st / A_slr)

# --- Convenience: dict keyed by pair string 'SLR/Stable'
SLR_BY_PAIR = {f"{d['slr']}/{d['stable']}": d for d in SLR_DATA}

# --- Public API ---------------------------------------------------------

def ratio_for_pair(pair_key: str, t_Gyr: float, alpha: float, nu_eff: float, as_number: bool = True):
    """Compute R(t) for a given pair key like '26Al/27Al'.

    Parameters
    ----------
    pair_key : str
        Key of the form 'SLR/Stable' matching SLR_BY_PAIR.
    t_Gyr : float
        Time in Gyr at which to evaluate the ratio.
    alpha, nu_eff : float
        Model parameters (1/Gyr).
    as_number : bool
        If True, return number ratio; if False, return mass ratio.

    Returns
    -------
    dict with fields: pair, y_ratio, lambda_Gyr, R_mass, R_number, alpha, nu_eff, beta, t_Gyr
    """
    d = SLR_BY_PAIR[pair_key]
    y_ratio = float(d['yield_ratio'])
    lam = lambda_per_Gyr(float(d['half_life_Myr']))
    beta = nu_eff - alpha

    R_mass = R_mass_caseC(t_Gyr, y_ratio, lam, alpha, nu_eff)
    R_num  = R_number_from_mass(R_mass, d['slr'], d['stable'])

    return {
        'pair': pair_key,
        'y_ratio': y_ratio,
        'lambda_Gyr': lam,
        'R_mass': float(R_mass),
        'R_number': float(R_num),
        'alpha': float(alpha),
        'nu_eff': float(nu_eff),
        'beta': float(beta),
        't_Gyr': float(t_Gyr),
    }


def ratios_for_all_pairs(t_Gyr: float, alpha: float, nu_eff: float, as_number: bool = True):
    """Compute ratios for all entries in SLR_DATA at time t_Gyr.

    Returns a list of dicts in the same schema as ratio_for_pair.
    """
    out = []
    for d in SLR_DATA:
        key = f"{d['slr']}/{d['stable']}"
        out.append(ratio_for_pair(key, t_Gyr, alpha, nu_eff, as_number=as_number))
    return out

def save_ratios_csv(filepath: str, t_Gyr: float, alpha: float, nu_eff: float, as_number: bool = True) -> str:
    """Compute ratios for all SLR pairs at (t_Gyr, alpha, nu_eff) and write a CSV.

    Columns:
      pair, slr, stable, y_ratio_number, half_life_Myr, lambda_Gyr^-1, mean_life_Myr,
      alpha, nu_eff, beta, t_Gyr, R_mass, R_number, source
    """
    rows = []
    for d in SLR_DATA:
        pair_key = f"{d['slr']}/{d['stable']}"
        lam = lambda_per_Gyr(d['half_life_Myr'])
        tau = mean_life_Myr(d['half_life_Myr'])
        res = ratio_for_pair(pair_key, t_Gyr, alpha, nu_eff, as_number=True)
        rows.append({
            'pair': pair_key,
            'slr': d['slr'],
            'stable': d['stable'],
            'y_ratio_number': d['yield_ratio'],
            'half_life_Myr': d['half_life_Myr'],
            'lambda_Gyr^-1': lam,
            'mean_life_Myr': tau,
            'alpha': alpha,
            'nu_eff': nu_eff,
            'beta': nu_eff - alpha,
            't_Gyr': t_Gyr,
            'R_mass': res['R_mass'],
            'R_number': res['R_number'],
            'source': d.get('source', ''),
        })

    fieldnames = [
        'pair','slr','stable','y_ratio_number','half_life_Myr','lambda_Gyr^-1','mean_life_Myr',
        'alpha','nu_eff','beta','t_Gyr','R_mass','R_number','source'
    ]
    with open(filepath, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    return filepath

def scatter_all_pairs(t_Gyr: float = 8.0, alpha: float = 1.0, nu_eff: float = 0.4,
                      annotate: bool = True, savepath = None):
    """Create a *points-only* scatter plot of number ratios vs mean life.

    X-axis: mean life τ (Myr), log scale.
    Y-axis: number ratio R(t) = N_SLR/N_stable, log scale.

    If annotate=True, each point is labeled with its isotope pair name (e.g., '26Al/27Al').

    Returns (taus, Rnums, labels) arrays in the plotting order.
    """
    # Compute number ratios for all pairs
    results = ratios_for_all_pairs(t_Gyr, alpha, nu_eff, as_number=True)

    # Build arrays of mean life (Myr), labels, and number ratios
    taus, labels, rnums = [], [], []
    for r in results:
        d = SLR_BY_PAIR[r['pair']]
        tau = mean_life_Myr(float(d['half_life_Myr']))
        taus.append(tau)
        labels.append(r['pair'])
        rnums.append(float(r['R_number']))

    taus = np.asarray(taus, dtype=float)
    rnums = np.asarray(rnums, dtype=float)
    labels = np.asarray(labels, dtype=object)

    # Sort by mean life to make the plot easier to read
    order = np.argsort(taus)
    taus = taus[order]
    rnums = rnums[order]
    labels = labels[order]

    # Points-only scatter
    plt.figure(figsize=(7, 5))
    plt.scatter(taus, rnums, s=45)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Mean life τ (Myr)')
    plt.ylabel('R(t) = N_SLR / N_stable')
    plt.title(f'Model ratios at t={t_Gyr} Gyr (\u03B1={alpha}, ν_eff={nu_eff})')

    if annotate:
        # Use adjustText to prevent overlapping labels
        try:
            from adjustText import adjust_text
            texts = []
            for x, y, lab in zip(taus, rnums, labels):
                if not np.isfinite(y):
                    continue
                text_obj = plt.annotate(lab, (x, y), fontsize=8)
                texts.append(text_obj)
            # Automatically adjust text positions to avoid overlaps
            adjust_text(texts, arrowprops=dict(arrowstyle='->', color='gray', lw=0.5))
        except ImportError:
            # Fallback to manual offsets if adjustText is not available
            offsets = [(3, 3), (3, -9), (3, 15), (3, -15), (3, 21), (3, -21), (-60, 3), (-60, -9)]
            for i, (x, y, lab) in enumerate(zip(taus, rnums, labels)):
                if not np.isfinite(y):
                    continue
                dx, dy = offsets[i % len(offsets)]
                plt.annotate(lab, (x, y), textcoords='offset points', xytext=(dx, dy), fontsize=8, 
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='none', alpha=0.7))

    plt.tight_layout()
    if savepath:
        plt.savefig(savepath, dpi=200)

    return taus, rnums, labels

if __name__ == '__main__':
    # Parameters for this export
    T = 8.0
    alpha = 1.0
    nu_eff = 0.4

    # Build an informative filename
    out_csv = f"ratios_t{T:.1f}Gyr_alpha{alpha:.2f}_nueff{nu_eff:.2f}.csv"
    path = save_ratios_csv(out_csv, T, alpha, nu_eff, as_number=True)
    print(f"Wrote {path}")
