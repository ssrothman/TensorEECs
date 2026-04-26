import numpy as np

try:
    import _tensoreecs as te
except Exception as exc:
    raise SystemExit(f"python smoke import failed: {exc}")

w = np.array([0.5, 0.3, 0.2], dtype=np.float64)
R = np.array(
    [
        [0.0, 0.2, 0.5],
        [0.2, 0.0, 0.4],
        [0.5, 0.4, 0.0],
    ],
    dtype=np.float64,
)

out2 = te.compute_eec2(w, R, mode="exact", storage_mode="sparse", dtype="float64")
out3 = te.compute_eec3(w, R, mode="exact", storage_mode="sparse", dtype="float64")

assert out2.ndim == 1
assert out3.ndim == 1
assert np.all(out2 >= 0)
assert np.all(out3 >= 0)
print("python smoke passed")
