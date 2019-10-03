import numpy as np
import pandas as pd
import scipy.stats as stats
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from . import plot

torus_dict = {
    'TF_BINDING_SITE': 'TF binding site',
    'CTCF_BINDING_SITE': 'CTCF binding site',
    'INTRON_VARIANT': 'Intron variant',
    'SYNONYMOUS_VARIANT': 'Synonymous variant',
    'SPLICE_DONOR_VARIANT': 'Splice donor variant',
    'NON_CODING_TRANSCRIPT_EXON_VARIANT': 'Non-coding transcript exon variant',
    'MISSENSE_VARIANT': 'Missense variant',
    'STOP_GAINED': 'Stop gained',
    '3_PRIME_UTR_VARIANT': "3' UTR variant",
    'FRAMESHIFT_VARIANT': 'Frameshift variant',
    'OPEN_CHROMATIN_REGION': 'Open chromatin region',
    'SPLICE_REGION_VARIANT': 'Splice region variant',
    '5_PRIME_UTR_VARIANT': "5' UTR variant",
    'PROMOTER': 'Promoter',
    'ENHANCER': 'Enhancer',
    'PROMOTER_FLANKING_REGION': 'Promoter-flanking region',
    'SPLICE_ACCEPTOR_VARIANT': 'Splice acceptor variant',
}

torus_short_dict = {i:i.replace(' variant', '') for i in torus_dict.values()}
torus_short_dict['Open chromatin region'] = 'Open chromatin'
torus_short_dict['Promoter-flanking region'] = 'Promoter-flanking'
torus_short_dict['Non-coding transcript exon variant'] = 'NC transcript'

# enhancer_d: Enhancer
# promoter_d: Promoter
# open_chromatin_region_d: Open chromatin
# promoter_flanking_region_d: Promoter-flanking
# CTCF_binding_site_d: CTCF binding site
# TF_binding_site_d: TF binding site
# 3_prime_UTR_variant_d: 3' UTR
# 5_prime_UTR_variant_d: 5' UTR
# frameshift_variant_d: Frameshift
# intron_variant_d: Intron
# missense_variant_d: Missense
# non_coding_transcript_exon_variant_d: NC transcript
# splice_acceptor_variant_d: Splice acceptor
# splice_donor_variant_d: Splice donor
# splice_region_variant_d: Splice region
# stop_gained_d: Stop gained
# synonymous_variant_d: Synonymous


def convert_torus(tensorqtl_files, out_file, phenotype_groups_file=None, mode='xQTL'):
    """Convert tensorQTL parquet files to Torus input format"""
    if os.path.exists(out_file):
        raise ValueError('Output file already exists')
    assert mode in ['xQTL', 'ixQTL']

    if phenotype_groups_file is not None:
        group_s = pd.read_csv(phenotype_groups_file, sep='\t', index_col=0, header=None, squeeze=True)
        group_size_s = group_s.value_counts()

    if mode=='xQTL':
        cols = ['phenotype_id', 'variant_id', 'tss_distance', 'pval_nominal', 'slope', 'slope_se']
    elif mode=='ixQTL':
        cols = ['phenotype_id', 'variant_id', 'tss_distance', 'pval_gi', 'b_gi', 'b_gi_se']

    for f in tensorqtl_files:
        print(f)
        df = pd.read_parquet(f, columns=cols)
        df['phenotype_id'] = df['phenotype_id'].apply(lambda x: x.rsplit(':',1)[-1])
        if phenotype_groups_file is not None:
            print('  * adjusting p-values by phenotype group size')
            if mode=='xQTL':
                df['pval_nominal'] = np.minimum(df['pval_nominal']*df['phenotype_id'].map(group_size_s), 1.0)
            elif mode=='ixQTL':
                df['pval_gi'] = np.minimum(df['pval_gi']*df['phenotype_id'].map(group_size_s), 1.0)
        df.to_csv(out_file, sep=' ', float_format='%.6g', compression='gzip', mode='a', index=False, header=None)


def load(torus_output, log2=True, short_labels=True):
    torus_df = pd.read_csv(torus_output, sep='\s+', index_col=0, header=None)
    torus_df.columns = ['mean', 'CI5', 'CI95']
    torus_df.index.name = 'feature'
    torus_df.drop('Intercept', axis=0, inplace=True)
    torus_df = torus_df[~torus_df.index.str.startswith('dtss')]
    torus_df.index = torus_df.index.map(lambda x: torus_dict.get(x.strip('.1').upper(), x.strip('.1').upper()))
    if short_labels:
        torus_df.index = torus_df.index.map(lambda x: torus_short_dict.get(x, x))
    if log2:
        torus_df *= np.log2(np.e)
    return torus_df


def load_summary(summary_file, log2=True):
    """Load aggregated output"""

    torus_df = pd.read_csv(summary_file, sep='\t', index_col=0)
    torus_df = torus_df[~torus_df.index.str.startswith('dtss')]
    torus_df.drop('Intercept', inplace=True)
    torus_df.index = torus_df.index.map(lambda x: torus_dict[x.strip('.1').upper()])

    lor_df = torus_df[torus_df.columns[torus_df.columns.str.endswith('lor')]].copy()
    lor_df.columns = lor_df.columns.str.replace('.lor','')

    if log2:
        lor_df = np.log2(np.exp(lor_df))

    return lor_df


def test_significance(torus_df1, torus_df2):
    assert np.all(torus_df1.index==torus_df2.index)
    se = (torus_df1['CI95']-torus_df1['CI5'] + torus_df2['CI95']-torus_df2['CI5']) / 3.919927969080108
    mu = torus_df1['mean'] - torus_df2['mean']
    zstat = mu / se
    pval = 2*stats.norm.sf(np.abs(zstat))
    m = pval<0.05/torus_df1.shape[0]
    return pd.DataFrame([pval, m], index=['pval', 'signif_bonferroni'], columns=torus_df1.index).T
