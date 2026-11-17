"""Convert the long-format synthetic IV dataset into dense per-surface arrays.

Run from the repo root:  python3 scripts/clean_data.py

Writes data/heston_dense.npz with
  iv           (surfaces, expiries, strikes) float32; NaN where the generator dropped the cell
  tv           same shape, float32: call time value C - max(S0 - K*exp(-r*T), 0); NaN where dropped
  T, lm        the shared grids; lm = log(S0 / K), the generator's sign convention
  params       (surfaces, 5) Heston parameters, one row per surface
  param_names  column names of params

Why tv is stored: a stored IV is only meaningful if the price it was inverted from has a
time value well above the solver tolerance (1e-10). Below that, the solver returns its
seed (0.01), its floor (1e-6), or a tolerance artefact. Keeping tv makes "which cells
count" a filter on this file (tv >= some threshold) instead of a regeneration.

Columns are loaded one at a time so peak memory stays near 2.5 GB.
"""
import zipfile

import numpy as np

SRC = "data/synthetic_iv_dataset.npz"
DST = "data/heston_dense.npz"
PARAM_NAMES = ["v0", "theta", "kappa", "xi", "rho"]   # order of HestonCFModel.params
CHUNK = 8_000_000


def chunks(n):
    for start in range(0, n, CHUNK):
        yield slice(start, min(start + CHUNK, n))


def grid_and_index(column):
    """Sorted unique values of a column, and each row's position in that grid."""
    grid = np.empty(0, dtype=column.dtype)
    for sl in chunks(column.size):
        grid = np.union1d(grid, np.unique(column[sl]))
    index = np.empty(column.size, dtype=np.int32)
    for sl in chunks(column.size):
        index[sl] = np.searchsorted(grid, column[sl])
        assert np.array_equal(grid[index[sl]], column[sl]), "grid values differ between rows"
    return grid, index


def first_row_per_surface(key, starts):
    """Read rows `starts` of a 2-D array that is too large to load whole."""
    with zipfile.ZipFile(SRC) as z, z.open(key + ".npy") as f:
        version = np.lib.format.read_magic(f)
        read_header = (np.lib.format.read_array_header_1_0 if version == (1, 0)
                       else np.lib.format.read_array_header_2_0)
        (n_rows, n_cols), _, dtype = read_header(f)
        out = np.empty((len(starts), n_cols), dtype=dtype)
        pos, rows_per_read = 0, 2_000_000
        while pos < n_rows:
            n = min(rows_per_read, n_rows - pos)
            block = np.frombuffer(f.read(n * n_cols * dtype.itemsize), dtype=dtype).reshape(n, n_cols)
            a, b = np.searchsorted(starts, [pos, pos + n])
            out[a:b] = block[starts[a:b] - pos]
            pos += n
    return out


def main():
    src = np.load(SRC)

    sid = src["surface_id"]
    assert np.all(np.diff(sid) >= 0), "rows are not grouped by surface"
    n_surfaces = int(sid[-1]) + 1
    starts = np.searchsorted(sid, np.arange(n_surfaces))
    assert np.array_equal(sid[starts], np.arange(n_surfaces)), "a surface has no rows at all"
    cell = sid.astype(np.int32)
    del sid

    T_grid, t_idx = grid_and_index(src["T"])
    lm_grid, k_idx = grid_and_index(src["log_moneyness"])
    n_T, n_lm = len(T_grid), len(lm_grid)
    print(f"{n_surfaces} surfaces, {n_T} expiries, {n_lm} strikes")

    cell *= n_T
    cell += t_idx
    cell *= n_lm
    cell += k_idx                    # flat position of each row in the dense array

    S0_col, r_col = src["S0"], src["r"]
    S0, r = float(S0_col[0]), float(r_col[0])
    assert (S0_col == S0).all() and (r_col == r).all(), "S0 or r varies between rows"
    del S0_col, r_col

    # Time value, in float64 on the long rows (float32 cannot resolve 1e-10 next to a price of 50).
    K_grid = S0 / np.exp(lm_grid)
    discount = np.exp(-r * T_grid)
    C = src["C"]
    tv = np.full(n_surfaces * n_T * n_lm, np.nan, dtype=np.float32)
    for sl in chunks(C.size):
        intrinsic = np.maximum(S0 - K_grid[k_idx[sl]] * discount[t_idx[sl]], 0.0)
        tv[cell[sl]] = C[sl] - intrinsic
    del C, t_idx, k_idx

    iv_col = src["iv"]
    iv = np.full(n_surfaces * n_T * n_lm, np.nan, dtype=np.float32)
    for sl in chunks(iv_col.size):
        iv[cell[sl]] = iv_col[sl]
    del iv_col, cell

    params = first_row_per_surface("params", starts)

    shape = (n_surfaces, n_T, n_lm)
    np.savez(DST, iv=iv.reshape(shape), tv=tv.reshape(shape), T=T_grid, lm=lm_grid,
             params=params, param_names=np.array(PARAM_NAMES))
    stored = ~np.isnan(iv)
    print(f"stored cells: {stored.mean():.2%} of the grid")
    for tv_min in (1e-10, 1e-8):
        print(f"cells with time value >= {tv_min:g}: {(tv >= tv_min).mean():.2%} of the grid")
    print("wrote", DST)


if __name__ == "__main__":
    main()
