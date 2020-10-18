import numpy as np
import pandas as pd


class Residualizer(object):
    def __init__(self, C, fail_colinear=False):
        # center and orthogonalize
        self.Q, R = np.linalg.qr(C - np.mean(C,0))
        self.dof = C.shape[0] - 2 - C.shape[1]

        # check for colinearity
        colinear_ix = np.abs(np.diag(R)) < np.finfo(np.float64).eps * C.shape[1]
        if np.any(colinear_ix):
            if fail_colinear:
                raise ValueError("Colinear or zero covariates detected")
            else:  # drop colinear covariates
                print('  * dropped colinear covariates: {}'.format(np.sum(colinear_ix)))
                self.Q = self.Q[:, ~colinear_ix]

    def transform(self, df, center=False):
        """Residualize rows of df wrt columns of C"""
        # transform input
        if isinstance(df, pd.DataFrame) or isinstance(df, pd.Series):
            M = df.values
        else:
            M = df

        isvector = False
        if isinstance(M, list) or (hasattr(M, 'shape') and len(M.shape)==1):
            M = np.array(M).reshape(1,-1)
            isvector = True

        # residualize M relative to C
        M0 = M - np.mean(M, axis=1, keepdims=True)
        if center:
            M0 = M0 - np.dot(np.dot(M0, self.Q), self.Q.T)
        else:
            M0 = M -  np.dot(np.dot(M0, self.Q), self.Q.T)  # retain original mean

        if isvector:
            M0 = M0[0]

        if isinstance(df, pd.DataFrame):
            M0 = pd.DataFrame(M0, index=df.index, columns=df.columns)
        elif isinstance(df, pd.Series):
            M0 = pd.Series(M0, index=df.index, name=df.name)

        return M0


def residualize(df, C, center=False, fail_colinear=False):
    r = Residualizer(C, fail_colinear=fail_colinear)
    return r.transform(df, center=center)


def center_normalize(x, axis=0):
    """Center and normalize x"""
    if isinstance(x, pd.DataFrame):
        x0 = x - np.mean(x.values, axis=axis, keepdims=True)
        return x0 / np.sqrt(np.sum(x0.pow(2).values, axis=axis, keepdims=True))
    elif isinstance(x, pd.Series):
        x0 = x - x.mean()
        return x0 / np.sqrt(np.sum(x0*x0))
    elif isinstance(x, np.ndarray):
        x0 = x - np.mean(x, axis=axis, keepdims=True)
        return x0 / np.sqrt(np.sum(x0*x0, axis=axis))


def padjust_bh(p):
    """
    Benjamini-Hochberg adjusted p-values

    Replicates p.adjust(p, method="BH") from R
    """
    n = len(p)
    i = np.arange(n,0,-1)
    o = np.argsort(p)[::-1]
    ro = np.argsort(o)
    return np.minimum(1, np.minimum.accumulate(np.float(n)/i * np.array(p)[o]))[ro]


def pi0est(p, lambda_qvalue):
    """
    pi0 statistic (Storey and Tibshirani, 2003)

    For fixed values of 'lambda'; equivalent to the qvalue::pi0est
    from R package qvalue
    """
    if np.min(p) < 0 or np.max(p) > 1:
        raise ValueError("p-values not in valid range [0, 1]")
    elif np.min(lambda_qvalue) < 0 or np.max(lambda_qvalue) >= 1:
        raise ValueError("lambda must be within [0, 1)")

    pi0 = np.mean(p >= lambda_qvalue) / (1 - lambda_qvalue)
    pi0 = np.minimum(pi0, 1)

    if pi0<=0:
        raise ValueError("The estimated pi0 <= 0. Check that you have valid p-values or use a different range of lambda.")

    return pi0


def bootstrap_pi1(pval, lambda_qvalue=0.5, bounds=[2.5, 97.5], n=1000):
    """Compute confidence intervals for pi1 with bootstrapping"""
    pi1_boot = []
    nfail = 0
    for _ in range(n):
        try:
            pi1_boot.append(1 - pi0est(np.random.choice(pval, len(pval), replace=True), lambda_qvalue=lambda_qvalue))
        except:
            nfail += 1
    if nfail > 0:
        print('Warning: {} bootstraps failed'.format(nfail))
    pi1_boot = np.array(pi1_boot)
    if len(pi1_boot) > 0:
        ci = np.percentile(pi1_boot, bounds)
    else:
        ci = np.array([np.NaN, np.NaN])
    return ci
