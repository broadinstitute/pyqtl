import numpy as np
import pandas as pd
import scipy.stats as stats
from statsmodels.formula.api import ols

# Code adapted from
# https://github.com/chr1swallace/coloc/blob/master/R/claudia.R


def var_data(f, N):
    """
    Variance of MLE of beta for quantitative trait, assuming var(y) = 1

    Args:
      f: minor allele freq
      N: sample size

    Returns:
      variance of MLE beta
    """
    return 1 / (2 * N * f * (1 - f))


def var_data_cc(f, N, s):
    """
    Variance of MLE of beta for case-control

    Args:
      f: minor allele freq
      N: sample size
      s: proportion of samples that are cases

    Returns:
      variance of MLE beta
    """
    return 1 / (2 * N * f * (1 - f) * s * (1 - s))


def logsum(x):
    """Computes log(sum(ABF)), where x = log(ABF)"""
    mmax = np.max(x)
    return mmax + np.log(np.sum(np.exp(x-mmax)))


def logdiff(x, y):
    """"""
    mmax = np.maximum(np.max(x), np.max(y))
    return mmax + np.log(np.exp(x - mmax) - np.exp(y - mmax))


def approx_bf_p(p, f, N, s=None, type='quant'):
    """
    Calculate approximate Bayes Factors

    Args:
      p: p-value
      f: minor allele frequency
      N: sample size
      s: proportion of samples that are cases
      type: 'quant' or 'cc'

    Returns:
      Data frame with lABF and intermediate calculations
    """
    if type == 'quant':
        sd_prior = 0.15
        v = var_data(f, N)
    else:
        sd_prior = 0.2
        v = var_data_cc(f, N, s)
    z = stats.norm.isf(0.5 * p)
    # shrinkage factor: ratio of the prior variance to the total variance
    r = sd_prior**2 / (sd_prior**2 + v)
    # Approximate BF  # I want ln scale to compare in log natural scale with LR diff
    labf = 0.5 * (np.log(1 - r) + r*z*z)
    return pd.DataFrame({'v':v, 'z':z, 'r':r, 'lABF':labf})


def approx_bf_estimates(z, v, type='quant', sdy=1):
    """
    Calculates approximate Bayes Factors using the variance of the regression coefficients

    See eq. (2) in Wakefield, 2009 and Supplementary methods from Giambartolomei et al., 2014.

    Args:
      z: normal deviate associated with regression coefficient and its variance (in effect the t-statistic, beta/beta_se)
      v: variance of the regression coefficient (beta_se**2)
      sdy: standard deviation of the trait

    Returns:
      Data frame with lABF and intermediate calculations
    """
    if type == 'quant':
        sd_prior = 0.15*sdy
    else:
        sd_prior = 0.2
    r = sd_prior**2 / (sd_prior**2 + v)
    labf = 0.5 * (np.log(1 - r) + r*z*z)
    return pd.DataFrame({'v':v, 'z':z, 'r':r, 'lABF':labf})


def combine_abf(l1, l2, p1=1e-4, p2=1e-4, p12=1e-5, verbose=False):
    """
    Calculate posterior probabilities for configurations, given logABFs for each SNP and prior probabilities

    Args:
      l1:  logABFs for trait 1
      l2:  logABFs for trait 2
      p1:  prior probability a SNP is associated with trait 1, default 1e-4
      p2:  prior probability a SNP is associated with trait 2, default 1e-4
      p12: p12 prior probability a SNP is associated with both traits, default 1e-5

    Returns:
      pd.Series of posterior probabilities
    """
    lsum = l1 + l2
    lh0_abf = 0
    lh1_abf = np.log(p1) + logsum(l1)
    lh2_abf = np.log(p2) + logsum(l2)
    lh3_abf = np.log(p1) + np.log(p2) + logdiff(logsum(l1) + logsum(l2), logsum(lsum))
    lh4_abf = np.log(p12) + logsum(lsum)
    all_abf = [lh0_abf, lh1_abf, lh2_abf, lh3_abf, lh4_abf]
    my_denom_log_abf = logsum(all_abf)  # denominator in eq. 2
    pp_abf = np.exp(all_abf - my_denom_log_abf)
    pp_abf = pd.Series(pp_abf, index=['pp_h{}_abf'.format(i) for i in range(5)])
    if verbose:
        print(pp_abf)
        print('PP abf for shared variant: {:.1f}%'.format(pp_abf['pp_h4_abf']*100))
    return pp_abf


def process_dataset(df, N=None, sdy=None, type='quant'):
    """
    Preprocessing steps, including calculation of approximate Bayes Factors

    Args:
      df: data frame with columns: beta, beta_se -or- pval, maf
      N: sample size
      sdy: standard deviation of the trait. Estimated from the beta_se and MAF if not provided.

    Returns a data frame with the additional columns:
      v (beta_se**2), z (z-score), r (shrinkage factor), lABF (log ABF)
    """
    if 'beta' in df and 'beta_se' in df:
        beta_var = df['beta_se']**2
        if sdy is None:
            print('WARNING: estimating sdy from the data')
            sdy = sdy_est(beta_var, df['maf'], N)
        res_df = approx_bf_estimates(df['beta']/df['beta_se'], beta_var, type=type, sdy=sdy)
    else:
        pval_col = df.columns[df.columns.str.startswith('pval')][0]
        res_df = approx_bf_p(df[pval_col], df['maf'], type=type, N=N, s=None)
    return df.join(res_df)


def sdy_est(vbeta, maf, n):
    """
    Estimate trait standard deviation given vectors of variance of coefficients,  MAF and sample size

      Estimate is based on var(beta-hat) = var(Y) / (n * var(X))
      var(X) = 2*maf*(1-maf)
      so we can estimate var(Y) by regressing n*var(X) against 1/var(beta)

    Args:
      vbeta: vector of variance of coefficients
      maf: vector of MAF (same length as vbeta)
      n: sample size

    Returns:
      estimated standard deviation of Y
    """
    print('Warning: estimating sdY from MAF and varbeta, provide this if known.')
    oneover = 1/vbeta
    nvx = 2 * n * maf * (1-maf)  # n * var(X)
    res = ols('nvx ~ oneover - 1', {'nvx':nvx, 'oneover':oneover}).fit()
    cf = res.params[0]
    return np.sqrt(cf)


def coloc_abf(df1, df2, N=None, sdy=None, p1=1e-4, p2=1e-4, p12=1e-5, verbose=False):
    """
    
    Args:
      df1, df2: DataFrames with columns 
                  'beta' and 'beta_se' -or-
                  'pval_nominal' and 'maf'
      N: sample size, must be provided if using p-values and MAF
    
    """

    if 'sample_size' in df1:
        n1 = int(df1['sample_size'][0])
    else:
        assert N is not None
        n1 = N

    if 'sample_size' in df2:
        n2 = int(df2['sample_size'][0])
    else:
        assert N is not None
        n2 = N

    mdf1 = process_dataset(df1, N=n1, sdy=sdy)
    mdf2 = process_dataset(df2, N=n2, sdy=sdy)
    merged_df = pd.merge(mdf1.reset_index(drop=True), mdf2.reset_index(drop=True), suffixes=('_1', '_2'), left_index=True, right_index=True)
    # merged_df = merged_df.sort_values('snp_1')
    internal_sum_lABF = merged_df['lABF_1'] + merged_df['lABF_2']
    merged_df['internal_sum_lABF'] = internal_sum_lABF
    my_denom_log_abf = logsum(internal_sum_lABF)
    merged_df['snp_pp_h4'] = np.exp(internal_sum_lABF - my_denom_log_abf)
    pp_abf = combine_abf(mdf1['lABF'], mdf2['lABF'], p1=p1, p2=p2, p12=p12, verbose=verbose)
    return pp_abf, merged_df
