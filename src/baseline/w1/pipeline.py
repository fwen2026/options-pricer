"""Week 1 toy data: a clean rectangular block of Heston IV surfaces for the MLP.

Run from the repo root:  python3 -m src.baseline.w1.pipeline

Toy runs only, never reported. A surface is kept only if every cell of the block is
valid, which is not a random subset: the dropped surfaces are mostly low-vol ones with
strongly negative rho. Week 3 replaces this with a per-cell validity mask.

Sign convention: lm = log(S0 / K), so positive lm means a low strike.
"""
import numpy as np


DATAFILE = "data/heston_dense.npz"   # built by scripts/clean_data.py

# A cell counts only if the call price it was inverted from has time value >= TV_MIN.
# Below 1e-10 the IV solver returns non-values: 0.01 (its seed), 1e-6 (its floor),
# or a smooth-looking artefact of its price tolerance.
TV_MIN = 1e-8
LM_MAX = 0.3      # keep |lm| <= LM_MAX
T_START = 12      # skip the 12 shortest expiries (T < 0.15): their wings run out of time value
T_END = 31        # skip T = 2.0: mispriced for a few low-kappa, high-xi surfaces
T_STRIDE = 2
LM_STRIDE = 3
MAX_NEIGHBOUR_JUMP = 0.03   # a real surface moves about 0.01 between neighbouring cells here


def tvt_split(ds : np.ndarray, train_frac=0.7, val_frac=0.15, test_frac=0.15):
    assert np.isclose(train_frac + val_frac + test_frac, 1.0)

    n = ds.shape[0]
    train_end = int(n * train_frac)
    val_end = train_end + int(n * val_frac)
    return ds[:train_end], ds[train_end:val_end], ds[val_end:]


def load_toy_block(path=DATAFILE):
    """Returns (iv, T, lm). iv is float32 with shape (surfaces, expiries, strikes), no bad cells."""
    ds = np.load(path)
    T, lm = ds["T"], ds["lm"]
    in_window = np.abs(lm) <= LM_MAX

    tv = ds["tv"][:, T_START:T_END, in_window]
    clean = (tv >= TV_MIN).all(axis=(1, 2))      # NaN compares False, so dropped cells fail too
    del tv

    iv = ds["iv"][:, T_START:T_END, in_window][clean]
    T, lm = T[T_START:T_END], lm[in_window]

    # Checked at full resolution, before striding hides anything.
    assert iv.shape[0] > 0 and iv.shape[1] > 0 and iv.shape[2] > 0
    assert not np.isnan(iv).any()
    assert np.abs(np.diff(iv, axis=2)).max() < MAX_NEIGHBOUR_JUMP, "jump between neighbouring strikes"
    assert np.abs(np.diff(iv, axis=1)).max() < MAX_NEIGHBOUR_JUMP, "jump between neighbouring expiries"

    print(f"{clean.sum()} of {len(clean)} surfaces kept")
    return iv[:, ::T_STRIDE, ::LM_STRIDE], T[::T_STRIDE], lm[::LM_STRIDE]


if __name__ == "__main__":
    iv, T, lm = load_toy_block()
    train, val, test = tvt_split(iv)

    print("block:", iv.shape, iv.dtype, "->", iv.shape[1] * iv.shape[2], "values per surface")
    print("expiries:", T.round(3))
    print("lm from", lm[0].round(3), "to", lm[-1].round(3), "in", len(lm), "points")
    print("IV range:", iv.min().round(4), "to", iv.max().round(4))
    print("train / val / test surfaces:", len(train), len(val), len(test))
