"""qtl.map: functions for mapping QTLs"""

__author__ = "Francois Aguet"
__copyright__ = "Copyright 2018-2020, The Broad Institute"
__license__ = "BSD3"

import numpy as np
import pandas as pd
import scipy.stats
import re
from . import stats
from . import genotype as gt
from . import locusplot
try:
    import rpy2
    has_rpy2 = True
except:
    has_rpy2 = False


def calculate_association(genotype, phenotype_s, covariates_df=None, impute=True, logp=False):
    """Compute genotype-phenotype associations"""
    if logp and not has_rpy2:
        raise ValueError("The rpy2 package is required to compute log p-values.")
    if isinstance(genotype, pd.Series):
        genotype_df = genotype.to_frame().T
    elif isinstance(genotype, pd.DataFrame):
        genotype_df = genotype
    else:
        raise ValueError('Input type not supported')

    # assert np.all(genotype_df.columns==phenotype_s.index)
    if covariates_df is not None:
        assert covariates_df.index.equals(genotype_df.columns)

    # impute missing genotypes
    if impute:
        gt.impute_mean(genotype_df, verbose=False)

    # residualize genotypes and phenotype
    if covariates_df is not None:
        r = stats.Residualizer(covariates_df)
        gt_res_df = r.transform(genotype_df)
        p_res_s = r.transform(phenotype_s)
        num_covar = covariates_df.shape[1]
    else:
        gt_res_df = genotype_df
        p_res_s = phenotype_s
        num_covar = 0

    if isinstance(p_res_s, pd.Series):
        n = p_res_s.std() / gt_res_df.std(axis=1)
    else:
        n = p_res_s.std(axis=1) / gt_res_df.std(axis=1).values

    gt_res_df = stats.center_normalize(gt_res_df, axis=1)
    if isinstance(p_res_s, pd.Series):
        p_res_s = stats.center_normalize(p_res_s)
    else:
        p_res_s = stats.center_normalize(p_res_s, axis=1)

    if isinstance(p_res_s, pd.Series):
        r = gt_res_df.dot(p_res_s)
    else:  # single genotype x phenotypes
        r = gt_res_df.dot(p_res_s.T).squeeze()
    dof = gt_res_df.shape[1] - 2 - num_covar
    tstat = r * np.sqrt(dof/(1-r*r))

    if not logp:
        pval = 2*scipy.stats.t.cdf(-np.abs(tstat), dof)
    else:
        r_pt = rpy2.robjects.r['pt']
        rt = rpy2.robjects.vectors.FloatVector(-np.abs(tstat))
        pval = -(np.array(r_pt(rt, dof, lower_tail=True, log=True)) + np.log(2)) * np.log10(np.e)

    df = pd.DataFrame(pval, index=tstat.index, columns=['pval_nominal'])
    df['slope'] = r * n
    df['slope_se'] = df['slope'] / tstat
    df['corr_r2'] = r*r
    df['tstat'] = tstat
    n2 = 2 * genotype_df.shape[1]
    af = genotype_df.sum(1) / n2
    if isinstance(p_res_s, pd.Series):
        df['af'] = af
    else:
        assert len(af) == 1
        df['af'] = af.values[0]
    ix = df['af'] <= 0.5
    m = genotype_df > 0.5
    a = m.sum(1).astype(int)
    b = (genotype_df < 1.5).sum(1).astype(int)
    df['ma_samples'] = np.where(ix, a, b)
    a = (genotype_df * m).sum(1).round().astype(int)  # round for missing/imputed genotypes
    df['ma_count'] = np.where(ix, a, n2-a)
    if isinstance(genotype, pd.DataFrame):
        if logp:
            df['r2'] = locusplot.compute_ld(genotype, df['pval_nominal'].idxmax())
        else:
            df['r2'] = locusplot.compute_ld(genotype, df['pval_nominal'].idxmin())

    # if isinstance(df.index[0], str) and '_' in df.index[0]:  # assume variant IDs in format chr_pos_ref_alt_build
    if isinstance(df.index[0], str) and len(re.findall("^(?:chr)?\w_?\d+_", df.index[0])) == 1:
        df['chr'] = df.index.map(lambda x: x.split('_')[0])
        df['position'] = df.index.map(lambda x: int(x.split('_')[1]))
    if isinstance(p_res_s, pd.Series):
        df.index.name = 'variant_id'
    else:
        df.index.name = 'phenotype_id'
    m = df['pval_nominal'] == 0
    if any(m):
        e = np.nextafter(0, 1)  # np.finfo(np.float64).tiny
        print(f"Warning: underflow detected (setting to {e}), use logp=True to compute p-values as -log10(P).")
        df.loc[m, 'pval_nominal'] = e
    return df


def map_pairs(genotype_df, phenotype_df, covariates_df=None, impute=True):
    """Calculates association statistics for arbitrary phenotype-variant pairs"""
    assert genotype_df.shape[0] == phenotype_df.shape[0]
    assert genotype_df.columns.equals(phenotype_df.columns)
    assert genotype_df.columns.equals(covariates_df.index)
    if impute:
        gt.impute_mean(genotype_df, verbose=False)

    # residualize genotypes and phenotype
    if covariates_df is not None:
        r = stats.Residualizer(covariates_df)
        gt_res_df = r.transform(genotype_df)
        p_res_df = r.transform(phenotype_df)
        num_covar = covariates_df.shape[1]
    else:
        gt_res_df = genotype_df
        p_res_df = phenotype_df
        num_covar = 0

    n = p_res_df.std(axis=1).values / gt_res_df.std(axis=1).values

    gt_res_df = stats.center_normalize(gt_res_df, axis=1)
    p_res_df = stats.center_normalize(p_res_df, axis=1)

    r = np.sum(gt_res_df.values * p_res_df.values, axis=1)
    dof = gt_res_df.shape[1] - 2 - num_covar

    tstat2 = dof*r*r / (1-r*r)
    pval = scipy.stats.f.sf(tstat2, 1, dof)

    df = pd.DataFrame({'phenotype_id':phenotype_df.index, 'variant_id':genotype_df.index, 'pval_nominal':pval})
    df['slope'] = r * n
    df['slope_se'] = df['slope'].abs() / np.sqrt(tstat2)
    df['af'] = genotype_df.sum(1).values / (2*genotype_df.shape[1])
    df['maf'] = np.where(df['af'] <= 0.5, df['af'], 1-df['af'])
    return df


def calculate_interaction(genotype_s, phenotype_s, interaction_s, covariates_df=None, impute=True):

    assert genotype_s.index.equals(interaction_s.index)

    # impute missing genotypes
    if impute:
        gt.impute_mean(genotype_s, verbose=False)

    # interaction term
    gi = genotype_s * interaction_s

    # center
    g0 = genotype_s - genotype_s.mean()
    gi0 = gi - gi.mean()
    i0 = interaction_s - interaction_s.mean()
    p0 = phenotype_s - phenotype_s.mean()

    dof = phenotype_s.shape[0] - 4
    # residualize
    if covariates_df is not None:
        r = stats.Residualizer(covariates_df)
        g0 =  r.transform(g0.values.reshape(1,-1), center=False)
        gi0 = r.transform(gi0.values.reshape(1,-1), center=False)
        p0 =  r.transform(p0.values.reshape(1,-1), center=False)
        i0 =  r.transform(i0.values.reshape(1,-1), center=False)
        dof -= covariates_df.shape[1]
    else:
        g0 = g0.values.reshape(1,-1)
        gi0 = gi0.values.reshape(1,-1)
        p0 = p0.values.reshape(1,-1)
        i0 = i0.values.reshape(1,-1)

    # regression
    X = np.r_[g0, i0, gi0].T
    Xinv = np.linalg.inv(np.dot(X.T, X))
    b = np.dot(np.dot(Xinv, X.T), p0.reshape(-1,1))
    r = np.squeeze(np.dot(X, b)) - p0
    rss = np.sum(r*r)
    b_se = np.sqrt(np.diag(Xinv) * rss / dof)
    b = np.squeeze(b)
    tstat = b / b_se
    pval = 2*scipy.stats.t.cdf(-np.abs(tstat), dof)

    return pd.Series({
        'b_g':b[0], 'b_g_se':b_se[0], 'pval_g':pval[0],
        'b_i':b[1], 'b_i_se':b_se[1], 'pval_i':pval[1],
        'b_gi':b[2],'b_gi_se':b_se[2],'pval_gi':pval[2],
    })#, r[0]


def compute_ld(genotype_df, variant_id):
    """Compute LD (r2)"""
    # return gt_df.corrwith(gt_df.loc[variant_id], axis=1, method='pearson')**2
    g0 = genotype_df - genotype_df.values.mean(1, keepdims=True)
    d = (g0**2).sum(1) * (g0.loc[variant_id]**2).sum()
    return (g0 * g0.loc[variant_id]).sum(1)**2 / d


def get_conditional_pvalues(group_df, genotypes, phenotype_df, covariates_df,
                            phenotype_id=None, window=200000, maf_threshold=0):
    """
    Compute conditional p-values for a set of variants defined in group_df

    group_df : dataframe with columns 'variant_id' and 'phenotype_id'
    genotypes : pd.DataFrame or qtl.genotype.GenotypeIndexer
    phenotype_df : pd.DataFrame
    covariates_df : pd.DataFrame
    """
    assert phenotype_df.columns.equals(covariates_df.index)
    variant_id = group_df['variant_id'].iloc[0]

    if isinstance(genotypes, gt.GenotypeIndexer):
        gt_df = genotypes.get_genotype_window(variant_id, window=window)
    elif isinstance(genotypes, pd.DataFrame):
        gt_df = genotypes
    else:
        raise ValueError('Unsupported input format')

    if maf_threshold > 0:
        maf = gt_df.sum(1) / (2*gt_df.shape[1])
        maf = np.where(maf<=0.5, maf, 1-maf)
        gt_df = gt_df[maf >= maf_threshold].copy()

    res = []
    if phenotype_id is not None:
        pval_df = calculate_association(gt_df, phenotype_df.loc[phenotype_id], covariates_df=covariates_df)
        pval_df['r2'] = compute_ld(gt_df, variant_id)
        res.append(pval_df)

    for k,(variant_id, phenotype_id) in enumerate(zip(group_df['variant_id'], group_df['phenotype_id']), 1):
        print(f'\rProcessing {k}/{group_df.shape[0]}', end='')
        covariates = pd.concat([covariates_df, gt_df.loc[np.setdiff1d(group_df['variant_id'], variant_id)].T], axis=1)
        pval_df = calculate_association(gt_df, phenotype_df.loc[phenotype_id], covariates_df=covariates)
        pval_df['r2'] = compute_ld(gt_df, variant_id)

        res.append(pval_df)
    return res
