"""qtl.pca: helper functions for PCA of expression data"""

__author__ = "Francois Aguet"
__copyright__ = "Copyright 2018-2020, The Broad Institute"
__license__ = "BSD3"

import numpy as np
import pandas as pd
import sklearn.decomposition

from . import norm
from . import stats


def normalize_counts(gct_df, C=None, threshold=10, threshold_frac=0.1):
    """
    Normalize (size factors), threshold, residualize, center, unit norm

      gct_df: read counts or TPMs
      C: covariates matrix
    """

    gct_norm_df = gct_df.copy() / norm.deseq2_size_factors(gct_df)
    for x in gct_norm_df.values:
        m = x == 0
        if not all(m):
            x[m] = np.min(x[~m])/2

    # threshold low expressed genes: >=10 counts in >10% of samples (default)
    mask = np.mean(gct_norm_df >= threshold, axis=1) > threshold_frac
    gct_norm_df = np.log10(gct_norm_df[mask])

    if C is not None:
        gct_norm_df = stats.residualize(gct_norm_df, C, center=False)

    gct_norm_std_df = stats.center_normalize(gct_norm_df)
    return gct_norm_std_df


def get_pcs(gct_df, normalize=True, C=None, n_components=5, return_loadings=False, random_state=None):
    """
    Scale input GCT, threshold, normalize and calculate PCs
    """
    if normalize:
        gct_norm_std_df = normalize_counts(gct_df, C=C)
    else:
        gct_norm_std_df = gct_df

    pca = sklearn.decomposition.PCA(n_components=n_components, svd_solver='auto', random_state=random_state)
    pca.fit(gct_norm_std_df.T)
    P = pca.transform(gct_norm_std_df.T)
    pc_df = pd.DataFrame(P, index=gct_norm_std_df.columns,
                        columns=[f'PC{i}' for i in range(1, P.shape[1]+1)])
    pve_s = pd.Series(pca.explained_variance_ratio_ * 100, index=pc_df.columns, name='pve')
    if not return_loadings:
        return pc_df, pve_s
    else:
        loadings_df = pd.DataFrame(pca.components_.T, index=gct_norm_std_df.index, columns=pc_df.columns)
        return pc_df, pve_s, loadings_df
