# Author: Francois Aguet
import numpy as np
import pandas as pd
import scipy.stats as stats
import warnings


#--------------------------------------
#  eQTL expression normalization
#--------------------------------------
def normalize_quantiles(df):
    """
    Quantile normalization to the average empirical distribution
    Note: replicates behavior of R function normalize.quantiles
          from library("preprocessCore")

    Reference:
     [1] Bolstad et al., Bioinformatics 19(2), pp. 185-193, 2003

    Adapted from https://github.com/andrewdyates/quantile_normalize
    """
    M = df.values.copy()

    Q = M.argsort(axis=0)
    m,n = M.shape

    # compute quantile vector
    quantiles = np.zeros(m)
    for i in range(n):
        quantiles += M[Q[:,i],i]
    quantiles = quantiles / n

    for i in range(n):
        # Get equivalence classes; unique values == 0
        dupes = np.zeros(m, dtype=np.int)
        for j in range(m-1):
            if M[Q[j,i],i]==M[Q[j+1,i],i]:
                dupes[j+1] = dupes[j]+1

        # Replace column with quantile ranks
        M[Q[:,i],i] = quantiles

        # Average together equivalence classes
        j = m-1
        while j >= 0:
            if dupes[j] == 0:
                j -= 1
            else:
                idxs = Q[j-dupes[j]:j+1,i]
                M[idxs,i] = np.median(M[idxs,i])
                j -= 1 + dupes[j]
        assert j == -1

    return pd.DataFrame(M, index=df.index, columns=df.columns)


def inverse_normal_transform(M):
    """Transform rows to a standard normal distribution"""
    if isinstance(M, pd.Series):
        r = stats.mstats.rankdata(M)
        return pd.Series(stats.norm.ppf(r/(M.shape[0]+1)), index=M.index, name=M.name)
    else:
        R = stats.mstats.rankdata(M, axis=1)  # ties are averaged
        Q = stats.norm.ppf(R/(M.shape[1]+1))
        if isinstance(M, pd.DataFrame):
            Q = pd.DataFrame(Q, index=M.index, columns=M.columns)
        return Q

#--------------------------------------
#  DESeq size factor normalization
#--------------------------------------
def deseq2_size_factors(counts_df):
    """
    Calculate DESeq size factors
    median of ratio to reference sample (geometric mean of all samples)

    References:
     [1] Anders & Huber, 2010
     [2] R functions:
          DESeq::estimateSizeFactorsForMatrix
    """
    idx = np.all(counts_df>0, axis=1)
    tmp_df = np.log(counts_df.loc[idx.values])
    s = np.exp(np.median(tmp_df.T - np.mean(tmp_df, axis=1), axis=1))
    return s


def deseq2_normalized_counts(counts_df):
    """
    Equivalent to DESeq2:::counts.DESeqDataSet; counts(x, normalized=T)
    """
    return counts_df / deseq2_size_factors(counts_df)


def deseq2_cpm(counts_df):
    """Calculate CPM normalized by DESeq size factors"""
    cpm_df = counts_df/counts_df.sum(axis=0)*1e6
    s = deseq2_size_factors(cpm_df)
    return cpm_df / s

#--------------------------------------
#  edgeR TMM normalization
#--------------------------------------
def edger_calcnormfactors(counts_df, ref=None, logratio_trim=0.3,
                          sum_trim=0.05, acutoff=-1e10, verbose=False):
    """
    Calculate TMM (Trimmed Mean of M values) normalization.
    Reproduces edgeR::calcNormFactors.default

    Scaling factors for the library sizes that minimize
    the log-fold changes between the samples for most genes.

    Effective library size: TMM scaling factor * library size

    References:
     [1] Robinson & Oshlack, 2010
     [2] R functions:
          edgeR::calcNormFactors.default
          edgeR:::.calcFactorWeighted
          edgeR:::.calcFactorQuantile
    """

    # discard genes with all-zero counts
    Y = counts_df.values.copy()
    allzero = np.sum(Y>0,axis=1)==0
    if np.any(allzero):
        Y = Y[~allzero,:]

    # select reference sample
    if ref is None:  # reference sample index
        f75 = np.percentile(Y/np.sum(Y,axis=0), 75, axis=0)
        ref = np.argmin(np.abs(f75-np.mean(f75)))
        if verbose:
            print('Reference sample index: '+str(ref))

    N = np.sum(Y, axis=0)  # total reads in each library

    # with np.errstate(divide='ignore'):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        # log fold change; Mg in [1]
        logR = np.log2((Y/N).T / (Y[:,ref]/N[ref])).T
        # average log relative expression; Ag in [1]
        absE = 0.5*(np.log2(Y/N).T + np.log2(Y[:,ref]/N[ref])).T
        v = (N-Y)/N/Y
        v = (v.T + v[:,ref]).T  # w in [1]

    ns = Y.shape[1]
    tmm = np.zeros(ns)
    for i in range(ns):
        fin = np.isfinite(logR[:,i]) & np.isfinite(absE[:,i]) & (absE[:,i] > acutoff)
        n = np.sum(fin)

        loL = np.floor(n*logratio_trim)+1
        hiL = n + 1 - loL
        loS = np.floor(n*sum_trim)+1
        hiS = n + 1 - loS
        rankR = stats.rankdata(logR[fin,i])
        rankE = stats.rankdata(absE[fin,i])
        keep = (rankR >= loL) & (rankR <= hiL) & (rankE >= loS) & (rankE <= hiS)
        # in [1], w erroneously defined as 1/v ?
        tmm[i] = 2**(np.nansum(logR[fin,i][keep]/v[fin,i][keep]) / np.nansum(1/v[fin,i][keep]))

    tmm = tmm / np.exp(np.mean(np.log(tmm)))
    return tmm


def edger_cpm_default(counts_df, lib_size=None, log=False, prior_count=0.25):
    """
    edgeR normalized counts

    Reproduces edgeR::cpm.default
    """
    if lib_size is None:
        lib_size = counts_df.sum(axis=0)
    if log:
        prior_count_scaled = lib_size/np.mean(lib_size) * prior_count
        lib_size <- lib_size + 2 * prior_count_scaled
    lib_size = 1e-6 * lib_size
    if log:
        return np.log2((counts_df + prior_count_scaled)/lib.size)
    else:
        return counts_df / lib_size


def edger_cpm(counts_df, tmm=None, normalized_lib_sizes=True):
    """
    Return edgeR normalized/rescaled CPM (counts per million)

    Reproduces edgeR::cpm.DGEList
    """
    lib_size = counts_df.sum(axis=0)
    if normalized_lib_sizes:
        if tmm is None:
            tmm = edger_calcnormfactors(counts_df)
        lib_size = lib_size * tmm
    return counts_df / lib_size * 1e6

#--------------------------------------
#  limma-voom functions
#--------------------------------------
def voom_transform(counts_df):
    """Apply counts transformation from limma-voom"""
    lib_size = counts_df.sum(0)
    norm_factors = edger_calcnormfactors(counts_df)
    return np.log2((counts_df + 0.5) / (lib_size*norm_factors + 1) * 1e6)

#--------------------------------------
#  PoissonSeq size factor normalization
#--------------------------------------
def poissonseq_size_factors(counts_df, maxiter=10):
    """
    PoissonSeq normalization from Li et al., Biostatistics, 2012
    """
    gsum = counts_df.sum(1)

    # initialize
    ix = counts_df.index
    libsize = counts_df.sum(0)
    d_est = libsize / libsize.sum()

    # v = [d_est]
    i = 0
    meandiff = 1
    while i<maxiter and meandiff>1e-10:
        d = np.outer(gsum, d_est)
        gof = ((counts_df - d).pow(2) / d).sum(1)
        lb, ub = np.percentile(gof, [25,75])
        ix = gof[(lb<=gof) & (gof<=ub)].index
        d_est0 = d_est
        d_est = counts_df.loc[ix].sum(0) / gsum.loc[ix].sum()
        meandiff = (d_est - d_est0).pow(2).sum() / counts_df.shape[1]
        i += 1
        # print(meandiff)
        # v.append(d_est)
    return d_est
