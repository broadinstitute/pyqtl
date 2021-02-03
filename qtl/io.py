import pandas as pd
import numpy as np
from collections import defaultdict
import subprocess
import gzip


def to_bgzip(df, path, header=True, float_format=None):
    """Write DataFrame to bgzip"""
    assert path.endswith('.gz')
    bgzip = subprocess.Popen('bgzip -c > '+path, stdin=subprocess.PIPE, shell=True, encoding='utf8')
    df.to_csv(bgzip.stdin, sep='\t', index=False, header=header, float_format=float_format)
    stdout, stderr = bgzip.communicate()
    subprocess.check_call('tabix -f '+path, shell=True)


def write_bed(bed_df, output_name, header=True, float_format=None):
    """Write DataFrame to BED format"""
    if header:  
        assert (bed_df.columns[0]=='chr' or bed_df.columns[0]=='#chr') and bed_df.columns[1]=='start' and bed_df.columns[2]=='end'
        # header must be commented in BED format
        header = bed_df.columns.values.copy()
        header[0] = '#chr'
    to_bgzip(bed_df, output_name, header=header, float_format=float_format)


def read_gct(gct_file, sample_ids=None, dtype=None, load_description=True, skiprows=2):
    """Load GCT as DataFrame"""
    if sample_ids is not None:
        sample_ids = ['Name']+list(sample_ids)

    if gct_file.endswith('.gct.gz') or gct_file.endswith('.gct'):
        if dtype is not None:
            with gzip.open(gct_file, 'rt') as gct:
                for _ in range(skiprows):
                    gct.readline()
                sample_ids = gct.readline().strip().split()
            dtypes = {i:dtype for i in sample_ids[2:]}
            dtypes['Name'] = str
            dtypes['Description'] = str
            df = pd.read_csv(gct_file, sep='\t', skiprows=skiprows, usecols=sample_ids, index_col=0, dtype=dtypes)
        else:
            df = pd.read_csv(gct_file, sep='\t', skiprows=skiprows, usecols=sample_ids, index_col=0)
    elif gct_file.endswith('.parquet'):
        df = pd.read_parquet(gct_file, columns=sample_ids)
    else:
        raise ValueError('Unsupported input format.')
    if not load_description and 'Description' in df.columns:
        df.drop('Description', axis=1, inplace=True)
    return df


def write_gct(df, gct_file, float_format='%.6g'):
    """Write DataFrame to GCT format"""
    assert df.index.name=='Name' and df.columns[0]=='Description'
    if gct_file.endswith('.gct.gz'):
        opener = gzip.open(gct_file, 'wt', compresslevel=6)
    else:
        opener = open(gct_file, 'w')

    with opener as gct:
        gct.write('#1.2\n{0:d}\t{1:d}\n'.format(df.shape[0], df.shape[1]-1))
        df.to_csv(gct, sep='\t', float_format=float_format)


def gtf_to_tss_bed(annotation_gtf, feature='gene', exclude_chrs=[], phenotype_id='gene_id'):
    """Parse genes and TSSs from GTF and return DataFrame for BED output"""
    chrom = []
    start = []
    end = []
    gene_id = []
    gene_name = []
    with open(annotation_gtf, 'r') as gtf:
        for row in gtf:
            row = row.strip().split('\t')
            if row[0][0]=='#' or row[2]!=feature: continue # skip header
            chrom.append(row[0])

            # TSS: gene start (0-based coordinates for BED)
            if row[6]=='+':
                start.append(np.int64(row[3])-1)
                end.append(np.int64(row[3]))
            elif row[6]=='-':
                start.append(np.int64(row[4])-1)  # last base of gene
                end.append(np.int64(row[4]))
            else:
                raise ValueError('Strand not specified.')

            attributes = defaultdict()
            for a in row[8].replace('"', '').split(';')[:-1]:
                kv = a.strip().split(' ')
                if kv[0]!='tag':
                    attributes[kv[0]] = kv[1]
                else:
                    attributes.setdefault('tags', []).append(kv[1])

            gene_id.append(attributes['gene_id'])
            gene_name.append(attributes['gene_name'])

    if phenotype_id=='gene_id':
        bed_df = pd.DataFrame(data={'chr':chrom, 'start':start, 'end':end, 'gene_id':gene_id}, columns=['chr', 'start', 'end', 'gene_id'], index=gene_id)
    elif phenotype_id=='gene_name':
        bed_df = pd.DataFrame(data={'chr':chrom, 'start':start, 'end':end, 'gene_id':gene_name}, columns=['chr', 'start', 'end', 'gene_id'], index=gene_name)
    # drop rows corresponding to excluded chromosomes
    mask = np.ones(len(chrom), dtype=bool)
    for k in exclude_chrs:
        mask = mask & (bed_df['chr']!=k)
    bed_df = bed_df[mask]

    # sort by start position
    bed_df = bed_df.groupby('chr', sort=False, group_keys=False).apply(lambda x: x.sort_values('start'))

    return bed_df
