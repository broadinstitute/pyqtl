import numpy as np
import pandas as pd
import scipy.stats
from . import stats
from . import genotype as gt


def center_normalize(x, axis=0):
    """Center and normalize x"""
    if isinstance(x, pd.DataFrame):
        if axis==0:
            df = x - x.mean(axis=0)
            return df / np.sqrt(df.pow(2).sum(axis=0))
        elif axis==1:
            df = (x.T - x.mean(axis=1)).T
            return (df.T / np.sqrt(df.pow(2).sum(axis=1))).T
    elif isinstance(x, pd.Series):
        x0 = x - np.mean(x)
        return x0 / np.sqrt(np.sum(x0*x0, axis=0))
    elif isinstance(x, np.ndarray):
        x0 = x - np.mean(x, axis=axis, keepdims=True)
        return x0 / np.sqrt(np.sum(x0*x0, axis=axis))


def impute_mean(g):
    """Impute missing (np.NaN) genotypes to mean"""
    ix = g.isnull()
    if np.any(ix):
        g[ix] = np.nanmean(g)
    return g


def calculate_association(genotype, phenotype_s, covariates_df=None, impute=True):
    """Compute genotype-phenotype associations"""
    if isinstance(genotype, pd.Series):
        genotype_df = genotype.to_frame().T
    elif isinstance(genotype, pd.DataFrame):
        genotype_df = genotype
    else:
        raise ValueError('Input type not supported')

    assert np.all(genotype_df.columns==phenotype_s.index)

    # impute missing genotypes
    if impute:
        genotype_df = genotype_df.apply(impute_mean, axis=1)

    # residualize genotypes and phenotype
    if covariates_df is not None:
        r = stats.Residualizer(covariates_df)
        gt_res_df = r.transform(genotype_df)
        p_res_s = r.transform(phenotype_s)
        num_covar = covariates_df.shape[1]
    else:
        gt_res_df = genotype_df
        p_res_s = phenotype_s
        num_covar=0

    n = p_res_s.std()/gt_res_df.std(axis=1)

    gt_res_df = center_normalize(gt_res_df, axis=1)
    p_res_s = center_normalize(p_res_s)

    r = gt_res_df.dot(p_res_s)
    dof = gt_res_df.shape[1] - 2 - num_covar

    tstat2 = dof*r*r / (1-r*r)
    pval = scipy.stats.f.sf(tstat2, 1, dof)

    if isinstance(genotype, pd.Series):
        b = r * n
        b_se = np.abs(b) / np.sqrt(tstat2)
        maf = np.sum(genotype) / (2*len(genotype))
        return pval[0], b.iloc[0], b_se.iloc[0], maf
    else:
        df = pd.DataFrame(pval, index=tstat2.index, columns=['pval_nominal'])
        df['slope'] = r * n
        df['slope_se'] = df['slope'].abs() / np.sqrt(tstat2)
        df['maf'] = genotype_df.sum(1) / (2*genotype_df.shape[1])
        df['maf'] = np.where(df['maf']<=0.5, df['maf'], 1-df['maf'])
        df['chr'] = df.index.map(lambda x: x.split('_')[0])
        df['position'] = df.index.map(lambda x: int(x.split('_')[1]))
        return df


def get_conditional_pvalues(group_df, genotypes, phenotype_df, covariates_df, phenotype_id=None, window=1000000):
    assert np.all(phenotype_df.columns==covariates_df.index)
    variant_id = group_df['variant_id'].iloc[0]
    chrom, pos = variant_id.split('_')[:2]
    pos = int(pos)

    if isinstance(genotypes, gt.GenotypeIndexer):
        gt_df = genotypes.get_genotype_window(variant_id, window=200000)
    elif isinstance(genotypes, pd.DataFrame):
        gt_df = genotypes
    else:
        raise ValueError('Unsupported input format')

    maf = gt_df.sum(1) / (2*gt_df.shape[1])
    maf = np.where(maf<=0.5, maf, 1-maf)

    gt_df = gt_df[maf>0]

    res = []
    if phenotype_id is not None:
        pval_df = calculate_association(gt_df, phenotype_df.loc[phenotype_id], covariates_df=covariates_df)
        pval_df['r2'] = gt_df.corrwith(gt_df.loc[variant_id], axis=1, method='pearson')**2
        res.append(pval_df)

    for k,(variant_id, phenotype_id) in enumerate(zip(group_df['variant_id'], group_df['phenotype_id']), 1):
        print('\rProcessing {}/{}'.format(k, group_df.shape[0]), end='')
        covariates = np.hstack([
            covariates_df,
            # gi.get_genotypes(np.setdiff1d(group_df['variant_id'], variant_id)).T,
            gt_df.loc[np.setdiff1d(group_df['variant_id'], variant_id)].T,
        ])
        pval_df = calculate_association(gt_df, phenotype_df.loc[phenotype_id], covariates_df=covariates)
        pval_df['r2'] = gt_df.corrwith(gt_df.loc[variant_id], axis=1, method='pearson')**2

        res.append(pval_df)
    return res
