import pandas as pd
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
