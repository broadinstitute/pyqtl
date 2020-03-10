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
