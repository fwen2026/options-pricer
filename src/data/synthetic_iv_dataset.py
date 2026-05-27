from src.util.synthetic_iv_config import resolve_config
from src.pricing.heston_cf import CarrMadanSolver, HestonCFModel
from src.pricing.implied_vol_solver import implied_vol_solver
import numpy as np


# these are mostly a list of helper funcs for dataset generation scripts

def generate_single_surface(surface_id, cfg=None, rng=None):
    """Generate one Heston price surface on the FFT strike grid."""
    cfg = resolve_config(cfg)
    rng = np.random.default_rng(cfg["seed"]) if rng is None else rng
    model = HestonCFModel.from_cfg(cfg, rng=rng)
    solver = CarrMadanSolver(model, cfg=cfg)

    cfg_surface = cfg["surface_grid"]

    # log distrib. iv surfaces tend to become more unstable at low T values.
    if cfg_surface["log_t"]:
        T_grid = np.geomspace(cfg_surface["t_min"], cfg_surface["t_max"], cfg_surface["t_steps"])
    else:
        T_grid = np.linspace(cfg_surface["t_min"], cfg_surface["t_max"], cfg_surface["t_steps"])
    T_grid = np.sort(T_grid)

    K_grid = None
    price_slices = []

    for T in T_grid:
        K_slice, C_slice = solver.solve_fft(T)
        if K_grid is None:
            K_grid = K_slice # determinism is lovely
        price_slices.append(C_slice)

    C_grid = np.stack(price_slices, axis=0)
    lm_grid = np.log(model.S0 / K_grid)

    lm_filter = (lm_grid >= cfg_surface["lm_min"]) & (lm_grid <= cfg_surface["lm_max"])
    lm_grid = lm_grid[lm_filter]
    C_grid = C_grid[:, lm_filter]

    sort_idx = np.argsort(lm_grid)
    lm_grid = lm_grid[sort_idx]
    C_grid = C_grid[:, sort_idx]
    K_sorted = model.S0 / np.exp(lm_grid) # resort to re sorting :p

    arbitrage_mask = arbitrage_filter(np.broadcast_to(K_sorted, C_grid.shape), C_grid)

    TT, LM = np.meshgrid(T_grid, lm_grid, indexing="ij")
    KK = model.S0 / np.exp(LM)

    T_flat = TT.ravel()
    LM_flat = LM.ravel()
    K_flat = KK.ravel()
    C_flat = C_grid.ravel()
    discount_factor_flat = np.exp(-model.r * T_flat)

    lower_bounds = np.maximum(model.S0 - K_flat * discount_factor_flat, 0.0)
    # Require minimum time value: deep-ITM, short-T cells have C ≈ intrinsic,
    # so vega → 0 and the IV inversion diverges regardless of solver tolerance.
    time_value = C_flat - lower_bounds
    candidate_mask = (
        arbitrage_mask.ravel()
        & np.isfinite(C_flat)
        & (time_value >= 1e-3)
        & (C_flat <= model.S0)
    )

    iv_flat = np.full(T_flat.shape, np.nan) 
    
    if candidate_mask.any():
        iv_flat[candidate_mask] = implied_vol_solver(
            model.S0,
            K_flat[candidate_mask],
            T_flat[candidate_mask],
            model.r,
            C_flat[candidate_mask],
            option_type="call",
            cfg=cfg,
        )

    sqrt_T_flat = np.sqrt(T_flat)
    forward_flat = model.S0 * np.exp(model.r * T_flat)
    forward_LM_flat = np.log(forward_flat / K_flat)
    params_flat = np.tile(model.params, (T_flat.size, 1))
    S0_flat = np.full_like(T_flat, model.S0)
    surface_id_flat = np.full(T_flat.shape, surface_id, dtype=int)
    total_variance_flat = iv_flat ** 2 * T_flat
    valid = np.isfinite(iv_flat)
    
    return {
        "surface_id": surface_id_flat[valid],
        "S0": S0_flat[valid],
        "K": K_flat[valid],
        "T": T_flat[valid],
        "sqrt_T": sqrt_T_flat[valid],
        "r": np.full_like(T_flat, model.r)[valid],
        "log_moneyness": LM_flat[valid],
        "forward_log_moneyness": forward_LM_flat[valid],
        "forward": forward_flat[valid],
        "discount_factor": discount_factor_flat[valid],
        "C": C_flat[valid],
        "iv": iv_flat[valid],
        "total_variance": total_variance_flat[valid],
        "params": params_flat[valid],
    }
    

def generate_full_dataset(cfg=None):
    """Generates multiple surfaces and concatenates them into one dataset."""
    cfg = resolve_config(cfg)
    rng = np.random.default_rng(cfg["seed"])
    cfg_dataset = cfg["dataset"]
    surfaces = []
    total = cfg_dataset["num_samples"]
    for surface_id in range(total):
        surface_data = generate_single_surface(surface_id, cfg=cfg, rng=rng)
        surfaces.append(surface_data)
        if surface_id % 50 == 0:
            print(f"  {surface_id}/{total}", flush=True)
    return {
        key: np.concatenate([surface[key] for surface in surfaces])
        for key in surfaces[0].keys()
    }


def arbitrage_filter(K_grid, C_grid, tolerance=1e-8):
    """Validity check for generated prices (should be monotonically increasing)"""
    finite = np.isfinite(K_grid) & np.isfinite(C_grid)
    mask = finite.copy()

    # check if C increases monotonically with log moneyness. 
    dC = np.diff(C_grid, axis=1)
    monotone_pairs = dC >= -tolerance
    mask[:, :-1] &= monotone_pairs
    mask[:, 1:] &= monotone_pairs

    for row_idx in range(C_grid.shape[0]):
        K_desc = K_grid[row_idx]
        C_desc = C_grid[row_idx]
        K_asc = K_desc[::-1]
        C_asc = C_desc[::-1]
        slopes = np.diff(C_asc) / np.diff(K_asc)
        convex_pairs = np.diff(slopes) >= -tolerance
        convex_points_asc = np.ones(C_asc.shape, dtype=bool)
        convex_points_asc[1:-1] = convex_pairs
        mask[row_idx] &= convex_points_asc[::-1]

    return mask
