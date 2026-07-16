import numpy as np


class Lazy3DGrid:
    """
    Given grid coordinates (not necessarily regular, but intended to be meshed),
    creates a lazily instantiated 3D mesh grid.

    You can index each coordinate individually if three indices are provided, or
    you can iterate through the coordinate triplets in sequential order.

    Why do I order it as t, y, x? Since time frequencies vary the widest, I want
    it to be the slowest varying index when flattened.
    """
    def __init__(self, t, y, x):
        self.x = np.asarray(x, dtype=np.float64)
        self.y = np.asarray(y, dtype=np.float64)
        self.t = np.asarray(t, dtype=np.float64)
        self.shape = (len(t), len(y), len(x))
        self.ndim = 3
        self.size = np.prod(self.shape, dtype=np.int64)

    def __getitem__(self, idx):
        """
        Supports:
          - grid[k, j, i] → (t, y, x)
          - grid[n] → flattened index access (single or small slices)
          - grid[start:stop:step] → flattened slice, order of coordinate iteration
           from slowest to fastest: t -> y -> x
        """
        # 3D indexing
        if isinstance(idx, tuple) and len(idx) == 3:
            k, j, i = idx
            return (self.t[k], self.y[j], self.x[i])

        # 1D flattened indexing or slice
        if np.isscalar(idx):
            k, j, i = np.unravel_index(idx, self.shape)
            return (self.t[k], self.y[j], self.x[i])
        elif isinstance(idx, slice):
            start, stop, step = idx.indices(self.size)
            flat = np.arange(start, stop, step)
            k, j, i = np.unravel_index(flat, self.shape)
            return (self.t[k], self.y[j], self.x[i])
        elif isinstance(idx, (list, np.ndarray)):
            idx = np.asarray(idx)
            k, j, i = np.unravel_index(idx, self.shape)
            return (self.t[k], self.y[j], self.x[i])
        else:
            raise IndexError(f"Unsupported index type: {type(idx)}")

    def iter_chunks(self, chunk_size):
        """Yield triplets in chunks of at most chunk_size."""
        for start in range(0, self.size, chunk_size):
            stop = min(start + chunk_size, self.size)
            yield self[start:stop]
