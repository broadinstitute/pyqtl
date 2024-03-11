import numpy as np
import pandas as pd
import scipy.stats as stats
from statsmodels.formula.api import ols
import itertools

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
    pp_abf = pd.Series(pp_abf, index=[f'pp_h{i}_abf' for i in range(5)])
    if verbose:
        print(pp_abf)
        print(f"PP abf for shared variant: {pp_abf['pp_h4_abf']*100:.3f}%")
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


def abf(df1, df2, N=None, sdy=None, p1=1e-4, p2=1e-4, p12=1e-5, verbose=False):
    """

    Args:
      df1, df2: DataFrames with columns
                  'beta' and 'beta_se' -or-
                  'pval_nominal' and 'maf'
      N: sample size, must be provided if using p-values and MAF

    """

    if 'sample_size' in df1:
        n1 = int(df1['sample_size'].values[0])
    else:
        assert N is not None
        n1 = N

    if 'sample_size' in df2:
        n2 = int(df2['sample_size'].values[0])
    else:
        assert N is not None
        n2 = N

    if 'p_std' in df1:
        sdy1 = float(df1['p_std'].values[0])
    else:
        sdy1 = sdy
    if 'p_std' in df2:
        sdy2 = float(df2['p_std'].values[0])
    else:
        sdy2 = sdy
    mdf1 = process_dataset(df1, N=n1, sdy=sdy1)
    mdf2 = process_dataset(df2, N=n2, sdy=sdy2)

    merged_df = pd.merge(mdf1.reset_index(drop=True), mdf2.reset_index(drop=True), suffixes=('_1', '_2'), left_index=True, right_index=True)
    # merged_df = merged_df.sort_values('snp_1')
    internal_sum_lABF = merged_df['lABF_1'] + merged_df['lABF_2']
    merged_df['internal_sum_lABF'] = internal_sum_lABF
    my_denom_log_abf = logsum(internal_sum_lABF)
    merged_df['snp_pp_h4'] = np.exp(internal_sum_lABF - my_denom_log_abf)
    pp_abf = combine_abf(mdf1['lABF'], mdf2['lABF'], p1=p1, p2=p2, p12=p12, verbose=verbose)
    return pp_abf, merged_df


def susie(s1, s2, p1=1e-4, p2=1e-4, p12=5e-6, verbose=False, is_sorted=True):
    """
    Colocalisation with multiple causal variants using SuSiE

    s1, s2: outputs from SuSiE

    Note: this function assumes that 'lbf_variable' are indexed by 'cs_index':
      res['lbf_variable'] = res['lbf_variable'][res['sets']['cs_index']]
    See tensorqtl.susie.map() for additional details.

    """
    cs1 = s1['sets']
    cs2 = s2['sets']
    lbf1 = s1['lbf_variable']
    lbf2 = s2['lbf_variable']
    if not isinstance(lbf1, pd.DataFrame):
        lbf1 = pd.DataFrame(lbf1, columns=s1['pip'].index)
    if not isinstance(lbf2, pd.DataFrame):
        lbf2 = pd.DataFrame(lbf2, columns=s2['pip'].index)
    isnps = lbf1.columns[lbf1.columns.isin(lbf2.columns)]
    n = len(isnps)
    if cs1['cs'] is None or cs2['cs'] is None or len(cs1['cs']) == 0 or len(cs2['cs']) == 0 or n == 0:
        return None
    if verbose:
        print(f"Using {n} shared variants (of {lbf1.shape[1]} and {lbf2.shape[1]})")
    idx1 = cs1['cs_index']
    idx2 = cs2['cs_index']
    if not is_sorted:
        bf1 = lbf1.loc[idx1, isnps]
        bf2 = lbf2.loc[idx2, isnps]
    else:
        bf1 = lbf1[isnps]
        bf2 = lbf2[isnps]

    ret = bf_bf(bf1, bf2, p1=p1, p2=p2, p12=p12)

    ret['summary']['idx1'] = idx1[ret['summary']['idx1']]
    ret['summary']['idx2'] = idx2[ret['summary']['idx2']]
    # ret$summary[, `:=`(idx1, cs1$cs_index[idx1])]
    # ret$summary[, `:=`(idx2, cs2$cs_index[idx2])]
    return ret


def bf_bf(bf1, bf2, p1=1e-4, p2=1e-4, p12=5e-6, overlap_min=0.5, trim_by_posterior=True, verbose=False):
    """Colocalize two datasets represented by Bayes factors"""
    if isinstance(bf1, pd.Series):
        bf1 = bf1.to_frame().T
    if isinstance(bf2, pd.Series):
        bf2 = bf2.to_frame().T

    # combinations to test
    todo_df = pd.DataFrame(itertools.product(range(len(bf1)), range(len(bf2))), columns=['i', 'j'])
    todo_df['pp4'] = 0

    isnps = bf1.columns[bf1.columns.isin(bf2.columns)]
    if len(isnps) == 0:
        return None

    pp1 = logbf_to_pp(bf1, p1, last_is_null=True)
    pp2 = logbf_to_pp(bf2, p2, last_is_null=True)
    ph0_1 = 1 - np.sum(pp1, 1)
    ph0_2 = 1 - np.sum(pp2, 1)

    prop1 = pp1[isnps].sum(1) / pp1.sum(1)
    prop2 = pp2[isnps].sum(1) / pp2.sum(1)

    if trim_by_posterior:
        # drop combinations with insufficient overlapping variants
        drop = (prop1.values[todo_df['i']] < overlap_min) | (prop2.values[todo_df['j']] < overlap_min)
        if all(drop):
            print("WARNING: snp overlap too small between datasets: too few snps with high posterior in one trait represented in other")
            return None
#             return(list(summary = cbind(data.table(nsnps = length(isnps),
#                 hit1 = colnames(pp1)[apply(pp1, 1, which.max)][todo$i],
#                 hit2 = colnames(pp2)[apply(pp2, 1, which.max)][todo$j],
#                 PP.H0.abf = pmin(ph0.1[todo$i], ph0.2[todo$j]),
#                 PP.H1.abf = NA, PP.H2.abf = NA, PP.H3.abf = NA,
#                 PP.H4.abf = NA), todo[, .(idx1 = i, idx2 = j)])))
        elif any(drop):
            todo_df = todo_df[~drop]

    bf1 = bf1[isnps]
    bf2 = bf2[isnps]

    results = []
    PP = []
    for k in range(len(todo_df)):
        df = pd.DataFrame({'snp': isnps, 'bf1': bf1.values[todo_df['i'][k]].astype(np.float64),
                           'bf2': bf2.values[todo_df['j'][k]].astype(np.float64)})
        df['internal_sum_lABF'] = df['bf1'] + df['bf2']
        df['snp_pp_h4'] =  np.exp(df['internal_sum_lABF'] - logsum(df['internal_sum_lABF']))
        pp_abf = combine_abf(df['bf1'], df['bf2'], p1, p2, p12, verbose=verbose)

        PP.append(df['snp_pp_h4'])
        if df['snp_pp_h4'].isnull().all():
            df['snp_pp_h4'] = 0
            pp_abf = pd.Series([1, 0, 0, 0, 0], index=pp_abf.index, dtype=np.float64)
        hit1 = bf1.columns[np.argmax(bf1.values[todo_df['i'][k]])]
        # if (is.null(hit1)) {
        #     hit1 = "-"
        #     pp.abf[c(1, 3)] = c(0, 1)
        # }
        hit2 = bf2.columns[np.argmax(bf2.values[todo_df['j'][k]])]
        # if (is.null(hit2)) {
        #     hit2 = "-"
        #     pp.abf[c(1, 2)] = c(0, 1)
        # }
        results.append([df.shape[0], hit1, hit2] + pp_abf.tolist())
    results = pd.DataFrame(results, columns=['nsnps', 'hit1', 'hit2'] + pp_abf.index.tolist())
    results = pd.concat([results, todo_df[['i','j']].rename(columns={'i':'idx1', 'j':'idx2'})], axis=1)
    PP = pd.DataFrame(PP).T
    if len(todo_df) > 1:
        PP.columns = [f"snp_pp_h4_row{i}" for i in range(len(todo_df))]
    else:
        PP.columns = ["snp_pp_h4_abf"]

    m = results[['hit1', 'hit2']].duplicated()
    if any(m):
        results = results[~m]
        PP = PP[PP.columns[~m]]

    PP = pd.concat([pd.Series(isnps, name='snp'), PP], axis=1)
    return {'summary': results, 'results': PP, 'priors': pd.Series({'p1':p1, 'p2':p2, 'p12':p12})}


def logbf_to_pp(bf, pi, last_is_null=True):
    """
    Convert logBF matrix to PP matrix
    bf: Bayes Factors --- L by p or p+1 matrix?
    pi: prior probability
    last_is_null: True if the last value of the BF matrix corresponds to the null hypythesis of no associations.
    """
    if isinstance(bf, pd.DataFrame):
        cols = bf.columns
        index = bf.index
        bf = bf.values.copy()
    else:
        cols = None
        bf = bf.copy()

    n = bf.shape[1]
    if last_is_null:
        n -= 1
    if np.ndim(pi) == 0:
        if pi > 1/n:
            pi = 1/n
        if last_is_null:
            pi = np.r_[np.full(n, pi), 1-n*pi]
        else:
            pi = np.full(n, pi)
    m = pi == 0
    if any(m):
        pi[m] = 1e-16
        pi /= np.sum(pi)
    if last_is_null:
        bf -= bf[:, [-1]]
    priors = np.tile(np.log(pi), [bf.shape[0], 1])

    x = bf + priors
    mmax = np.max(x, 1, keepdims=True)
    denom = mmax + np.log(np.sum(np.exp(x - mmax), 1, keepdims=True))
    pp = np.exp(bf + priors - denom)
    if cols is not None:
        pp = pd.DataFrame(pp, columns=cols, index=index)
    return pp
