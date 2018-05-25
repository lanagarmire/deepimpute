import numpy as np
from scipy.stats import expon


class MaskedArray(object):

    def __init__(self, data=None, mask=None, distr="exp", dropout=0.01, seed=1):
        self.data = np.array(data)
        self._binMask = np.array(mask)
        self.shape = data.shape
        self.distr = distr
        self.dropout = dropout
        self.seed = seed

    @property
    def binMask(self):
        return self._binMask

    @binMask.setter
    def binMask(self, value):
        self._binMask = value.astype(bool)

    def getMaskedMatrix(self):
        maskedMatrix = self.data.copy()
        maskedMatrix[~self.binMask] = 0
        return maskedMatrix

    def getMasked(self, rows=True):
        """ Generator for row or column mask """
        compt = 0
        if rows:
            while compt < self.shape[0]:
                yield [
                    self.data[compt, idx]
                    for idx in range(self.shape[1])
                    if not self.binMask[compt, idx]
                ]
                compt += 1
        else:
            while compt < self.shape[1]:
                yield [
                    self.data[idx, compt]
                    for idx in range(self.shape[0])
                    if not self.binMask[idx, compt]
                ]
                compt += 1

    def getMasked_flat(self):
        return self.data[~self.binMask]

    def copy(self):
        args = {"data": self.data.copy(), "mask": self.binMask.copy()}
        return MaskedArray(**args)

    def get_probs(self, vec):
        return {
            "exp": expon.pdf(vec, 0, 20),
            "uniform": np.tile([1. / len(vec)], len(vec)),
        }.get(self.distr)

    def get_Nmasked(self, idx):
        cells_g = self.data[:, idx]
        dp_i = (1 + (cells_g == 0).sum() * 1.) / self.shape[0]
        dp_f = np.exp(-2 * np.log10(cells_g.mean()) ** 2)
        return 1 + int((cells_g == 0).sum() * dp_f / dp_i)

    # def generate_sim(self):
    #     np.random.seed(self.seed)
    #     self.binMask = np.ones(self.shape).astype(bool)

    #     for g in range(self.shape[1]):
    #         cells_g = self.data[:,g]
    #         # Retrieve indices of positive values
    #         ind_pos = np.arange(self.shape[0])[cells_g>0]
    #         cells_g_pos = cells_g[ind_pos]
    #         # Get masking probability of each value
    #         if cells_g_pos.size > 5:
    #             probs = self.get_probs(cells_g_pos)
    #             n_masked = self.get_Nmasked(g)
    #             if n_masked >= cells_g_pos.size:
    #                 print('Warning: too many cells masked for gene {} ({}/{})'.
    #                       format(g,n_masked,cells_g_pos.size))
    #                 n_masked = 1+int(0.5*cells_g_pos.size)
    #             masked_idx = np.random.choice(cells_g_pos.size,n_masked,
    #                                           p=probs/probs.sum(),
    #                                           replace=False)
    #             self.binMask[ind_pos[sorted(masked_idx)],g] = False

    def generate(self):
        np.random.seed(self.seed)
        self.binMask = np.ones(self.shape).astype(bool)

        for c in range(self.shape[0]):
            cells_c = self.data[c, :]
            # Retrieve indices of positive values
            ind_pos = np.arange(self.shape[1])[cells_c > 0]
            cells_c_pos = cells_c[ind_pos]
            # Get masking probability of each value

            if cells_c_pos.size > 5:
                probs = self.get_probs(cells_c_pos)
                n_masked = 1 + int(self.dropout * len(cells_c_pos))
                if n_masked >= cells_c_pos.size:
                    print(
                        "Warning: too many cells masked for gene {} ({}/{})".format(
                            c, n_masked, cells_c_pos.size
                        )
                    )
                    n_masked = 1 + int(0.5 * cells_c_pos.size)

                masked_idx = np.random.choice(
                    cells_c_pos.size, n_masked, p=probs / probs.sum(), replace=False
                )
                self.binMask[c, ind_pos[sorted(masked_idx)]] = False
