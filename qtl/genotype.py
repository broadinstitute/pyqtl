# Author: Francois Aguet
import numpy as np
import pandas as pd
import gzip
import subprocess
import os
import tempfile

MISSING = -9  # PLINK2 convention
gt_dosage_dict = {'0/0': 0, '0/1': 1, '1/1': 2, './.': MISSING,
                  '0|0': 0, '0|1': 1, '1|0': 1, '1|1': 2, '.|.': MISSING}

class GenotypeIndexer(object):
    def __init__(self, genotype_df, variant_df, sample_ids=None):
        self.genotype_df = genotype_df
        self.index_dict = {j:i for i,j in enumerate(variant_df.index)}
        self.variant_df = variant_df.copy()
        self.variant_df['index'] = np.arange(variant_df.shape[0])
        self.chr_variant_dfs = {c:g[['pos', 'index']] for c,g in self.variant_df.groupby('chrom')}
        if sample_ids is None:
            self.sample_ids = genotype_df.columns
            self.sample_ix = np.arange(genotype_df.shape[1])
        else:
            self.sample_ids = sample_ids
            self.sample_ix = np.array([genotype_df.columns.tolist().index(i) for i in sample_ids])

    def set_sample_ids(self, sample_ids):
        self.sample_ix = np.array([genotype_df.columns.tolist().index(i) for i in sample_ids])

    def get_indexes(self, variant_ids):
        return [self.index_dict[i] for i in variant_ids]

    def get_genotype(self, variant_id):
        return self.genotype_df.values[self.index_dict[variant_id], self.sample_ix]

    def get_genotypes(self, variant_ids):
        return self.genotype_df.values[[self.index_dict[i] for i in variant_ids]][:, self.sample_ix]

    def get_genotype_window(self, region_str):
        chrom, pos = region_str.split(':')
        start, end = pos.split('-')
        lb = np.searchsorted(self.chr_variant_dfs[chrom]['pos'].values, int(start))
        ub = np.searchsorted(self.chr_variant_dfs[chrom]['pos'].values, int(end), side='right')
        ub = np.minimum(ub, self.chr_variant_dfs[chrom].shape[0]-1)
        lb = self.chr_variant_dfs[chrom]['index'][lb]
        ub = self.chr_variant_dfs[chrom]['index'][ub]
        return self.genotype_df.iloc[lb:ub][self.sample_ids]


def get_sample_ids(vcf):
    """Get sample IDs"""
    if vcf.endswith('.bcf'):
        return subprocess.check_output(f'bcftools query -l {vcf}', shell=True).decode().strip().split('\n')
    else:
        with gzip.open(vcf, 'rt') as f:
            for line in f:
                if line[:2]=='##': continue
                break
        return line.strip().split('\t')[9:]


def get_contigs(vcfpath):
    """Get list of contigs"""
    chrs = subprocess.check_output('tabix --list-chroms '+vcfpath, shell=True, executable='/bin/bash')
    return chrs.decode().strip().split()


def get_variant_ids(vcf):
    """Get list of variant IDs ('ID' field)"""
    s = subprocess.check_output(f'zcat {vcf} | grep -v "#" | cut -f3', shell=True)
    return s.strip(b'\n').split(b'\n')


def get_cis_genotypes(chrom, tss, vcf, field='GT', dosages=True, window=1000000):
    """Get genotypes in cis window (using tabix)"""
    region_str = chrom+':'+str(np.maximum(tss-window, 1))+'-'+str(tss+window)
    return get_genotypes_region(vcf, region_str, field=field, dosages=dosages)


def get_genotypes_region(vcf, region, field='GT', dosages=True):
    """Get genotypes, using region (chr:start-end) string"""
    s = subprocess.check_output(f'tabix {vcf} {region}',
                                shell=True, executable='/bin/bash')
    s = s.decode().strip()
    if len(s) == 0:
        return None
    #     raise ValueError(f'No variants in region {region}')
    s = s .split('\n')
    variant_ids = [si.split('\t', 3)[-2] for si in s]
    field_ix = s[0].split('\t')[8].split(':').index(field)

    if dosages:
        if field == 'GT':
            s = [[gt_dosage_dict[i.split(':', field_ix+1)[field_ix]] for i in si.split('\t')[9:]] for si in s]
        elif field == 'DS':
            s = [[i.split(':', field_ix+1)[field_ix] for i in si.split('\t')[9:]] for si in s]
        dtype = np.float32
    else:
        s = [[i.split(':', field_ix+1)[field_ix] for i in si.split('\t')[9:]] for si in s]
        dtype = str

    return pd.DataFrame(data=s, index=variant_ids, columns=get_sample_ids(vcf), dtype=dtype)


def impute_mean(df, missing=lambda x: np.isnan(x), verbose=True):
    """Row-wise mean imputation (in place). Missing values: np.nan by default."""
    if isinstance(df, pd.DataFrame):
        genotypes = df.values
    else:
        genotypes = df

    n = 0
    for k,g in enumerate(genotypes,1):
        ix = missing(g)
        if np.any(ix):
            g[ix] = np.mean(g[~ix])
            n += 1

    if verbose and n > 0:
        print(f'  * imputed at least 1 sample in {n} sites')


def get_genotype(variant_id, vcf, field='GT', convert_gt=True, sample_ids=None):
    """
    Parse genotypes for given variant from VCF. Requires tabix.

      variant_id: {chr}_{pos}_{ref}_{alt}_{build}
      vcf:        vcf path
      field:      GT or DS
      convert_gt: convert GT to dosages
      sample_ids: VCF sample IDs
    """

    chrom, pos = variant_id.split('_')[:2]
    s = subprocess.check_output(f"tabix {vcf} {chrom}:{pos}-{pos}", shell=True)
    if len(s) == 0:
        raise ValueError(f"Variant '{variant_id}' not found in VCF.")

    s = s.decode().strip()
    if '\n' in s:
        s = s.split('\n')
        try:
            s = s[np.nonzero(np.array([i.split('\t',3)[-2] for i in s]) == variant_id)[0][0]]
        except:
            raise ValueError("Variant ID not found in VCF.")
    s = s.split('\t')
    fmt = s[8].split(':')

    if field == 'DS':
        if 'DS' in fmt:
            ds_ix = fmt.index('DS')
            s = np.array([np.float32(i.split(':')[ds_ix]) for i in s[9:]])  # dosages
        else:
            raise ValueError('No dosage (DS) values found in VCF.')
    # check format: use GT if DS not present
    else:
        assert fmt[0] == 'GT'
        s = [i.split(':', 1)[0] for i in s[9:]]

        if convert_gt:
            s = np.float32([gt_dosage_dict[i] for i in s])

    if sample_ids is None:
        sample_ids = get_sample_ids(vcf)
    s = pd.Series(s, index=sample_ids, name=variant_id)

    return s


def get_genotypes(variant_ids, vcf, field='GT', drop_duplicates=True):
    """"""

    variant_id_set = set(variant_ids)

    with tempfile.NamedTemporaryFile() as regions_file:
        df = pd.DataFrame([i.split('_')[:2] for i in variant_id_set], columns=['chr', 'pos'])
        df['pos'] = df['pos'].astype(int)
        df = df.sort_values(['chr', 'pos'])
        df.to_csv(regions_file.name, sep='\t', index=False, header=False)
        s = subprocess.check_output(f'tabix {vcf} --regions {regions_file.name}', shell=True)

    s = s.decode().strip().split('\n')
    s = [i.split('\t') for i in s]
    variant_ids2 = [i[2] for i in s]
    if field == 'GT':
        gt_ix = s[0][8].split(':').index('GT')
        dosages = [[gt_dosage_dict[j.split(':')[gt_ix]] for j in i[9:]] for i in s]
    elif field == 'DS':
        ds_ix = s[0][8].split(':').index('DS')
        dosages = np.float32([[j.split(':')[ds_ix] for j in i[9:]] for i in s])
    df = pd.DataFrame(dosages, index=variant_ids2, columns=get_sample_ids(vcf))
    df = df[df.index.isin(variant_id_set)]
    if drop_duplicates:
        df = df[~df.index.duplicated()]
    return df


def get_allele_stats(genotype_df):
    """Returns allele frequency, minor allele samples, and minor allele counts (row-wise)."""
    # allele frequency
    n2 = 2 * genotype_df.shape[1]
    af = genotype_df.sum(1) / n2
    # minor allele samples and counts
    ix = af <= 0.5
    m = genotype_df > 0.5
    a = m.sum(1)
    b = (genotype_df < 1.5).sum(1)
    ma_samples = np.where(ix, a, b)
    a = (genotype_df * m).sum(1).astype(int)
    ma_count = np.where(ix, a, n2-a)
    return af, ma_samples, ma_count


def load_vcf(vcf, field='GT', dtype=None, verbose=False):
    """Load VCF as DataFrame"""

    sample_ids = subprocess.check_output(f'bcftools query -l {vcf}', shell=True).decode().strip().split()
    n_samples = len(sample_ids)
    n_variants = int(subprocess.check_output(f'bcftools index -n {vcf}', shell=True).decode())

    if dtype is None:
        if field == 'GT':
            dtype = np.int8
        elif field == 'DS':
            dtype = np.float32
    dosages = np.zeros([n_variants, n_samples], dtype=dtype)

    variant_ids = []
    with gzip.open(vcf, 'rt') as f:
        for line in f:
            if line.startswith('#'): continue  # skip header lines
            break

        # parse format from first line
        line = line.strip().split('\t')
        if field not in line[8]:
            raise ValueError(f"FORMAT does not include {field}. Available fields: {', '.join(line[8].split(':'))}")
        format_ix = line[8].split(':').index(field)
        variant_ids.append(line[2])
        if field == 'GT':
            dosages[0,:] = [gt_dosage_dict.get(i.split(':')[format_ix], MISSING) for i in line[9:]]
        elif field == 'DS':
            d = [i.split(':')[format_ix] for i in line[9:]]
            d = [dtype(i) if i != '.' else dtype(MISSING) for i in d]
            dosages[0,:] = d

        for k,line in enumerate(f, 1):
            line = line.strip().split('\t')
            variant_ids.append(line[2])
            if field == 'GT':
                dosages[k,:] = [gt_dosage_dict.get(i.split(':')[format_ix], MISSING) for i in line[9:]]
            elif field == 'DS':
                d = [i.split(':')[format_ix] for i in line[9:]]
                d = [dtype(i) if i != '.' else dtype(MISSING) for i in d]
                dosages[k,:] = d  # array?
            if verbose and ((k+1) % 1000 == 0 or k+1 == n_variants):
                print(f'\rVariants parsed: {k+1:,}', end='' if k+1 < n_variants else None)

    return pd.DataFrame(dosages, index=variant_ids, columns=sample_ids)
