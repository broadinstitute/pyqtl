# Author: Francois Aguet
import numpy as np
import pandas as pd
import gzip
import subprocess
import os
import tempfile


gt_dosage_dict = {'0/0':0, '0/1':1, '1/1':2, './.':np.NaN,
                  '0|0':0, '0|1':1, '1|0':1, '1|1':2, '.|.':np.NaN}


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

    def get_genotype_window(self, variant_id, window=200000):
        chrom, pos = variant_id.split('_')[:2]
        pos = int(pos)
        lb = np.searchsorted(self.chr_variant_dfs[chrom]['pos'].values, pos - window)
        ub = np.searchsorted(self.chr_variant_dfs[chrom]['pos'].values, pos + window, side='right')
        ub = np.minimum(ub, self.chr_variant_dfs[chrom].shape[0]-1)
        lb = self.chr_variant_dfs[chrom]['index'][lb]
        ub = self.chr_variant_dfs[chrom]['index'][ub]
        return self.genotype_df.iloc[lb:ub][self.sample_ids]


def get_sample_ids(vcf):
    """Get sample IDs"""
    if vcf.endswith('.bcf'):
        return subprocess.check_output('bcftools query -l {}'.format(vcf), shell=True).decode().strip().split('\n')
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
    s = subprocess.check_output('zcat {} | grep -v "#" | cut -f3'.format(vcf), shell=True)
    return s.strip(b'\n').split(b'\n')


def get_cis_genotypes(chrom, tss, vcf, field='GT', dosages=True, window=1000000):
    """Get genotypes in cis window (using tabix)"""
    region_str = chrom+':'+str(np.maximum(tss-window, 1))+'-'+str(tss+window)
    return get_genotypes_region(vcf, region_str, field=field, dosages=dosages)


def get_genotypes_region(vcf, region, field='GT', dosages=True):
    """Get genotypes, using region (chr:start-end) string"""
    s = subprocess.check_output('tabix {} {}'.format(vcf, region),
                                shell=True, executable='/bin/bash')
    s = s.decode().strip()
    if len(s)==0:
        raise ValueError('No variants in region {}'.format(region))
    s = s .split('\n')
    variant_ids = [si.split('\t', 3)[-2] for si in s]
    field_ix = s[0].split('\t')[8].split(':').index(field)

    if dosages:
        s = [[gt_dosage_dict[i.split(':', field_ix+1)[field_ix]] for i in si.split('\t')[9:]] for si in s]
        dtype = np.float32
    else:
        s = [[i.split(':', field_ix+1)[field_ix] for i in si.split('\t')[9:]] for si in s]
        dtype = str

    return pd.DataFrame(data=s, index=variant_ids, columns=get_sample_ids(vcf), dtype=dtype)


def impute_mean(df, missing=lambda x: np.isnan(x), verbose=True):
    """Row-wise mean imputation (in place)"""
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
        print('  * imputed at least 1 sample in {} sites'.format(n))


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
    s = subprocess.check_output('tabix '+vcf+' '+chrom+':'+pos+'-'+str(int(pos)+1), shell=True)
    if len(s) == 0:
        raise ValueError("Variant '{}' not found in VCF.".format(variant_id))

    s = s.decode().strip()
    if '\n' in s:
        s = s.split('\n')
        try:
            s = s[np.nonzero(np.array([i.split('\t',3)[-2] for i in s])==variant_id)[0][0]]
        except:
            raise ValueError("Variant ID not found in VCF.")
    s = s.split('\t')
    fmt = s[8].split(':')

    if field=='DS':
        if 'DS' in fmt:
            s = np.array([np.float32(i.rsplit(':', 1)[-1]) for i in s[9:]])  # dosages
        else:
            raise ValueError('No dosage (DS) values found in VCF.')
    # check format: use GT if DS not present
    else:
        assert fmt[0]=='GT'
        s = [i.split(':', 1)[0] for i in s[9:]]

        if convert_gt:
            s = np.float32([gt_dosage_dict[i] for i in s])

    if sample_ids is None:
        sample_ids = get_sample_ids(vcf)
    s = pd.Series(s, index=sample_ids, name=variant_id)

    return s


def get_genotypes(variant_ids, vcf, field='GT'):
    """"""

    variant_id_set = set(variant_ids)

    with tempfile.NamedTemporaryFile() as regions_file:
        df = pd.DataFrame([i.split('_')[:2] for i in variant_id_set], columns=['chr', 'pos'])
        df['pos'] = df['pos'].astype(int)
        df = df.sort_values(['chr', 'pos'])
        df.to_csv(regions_file.name, sep='\t', index=False, header=False)
        s = subprocess.check_output('tabix {} --regions {}'.format(vcf, regions_file.name), shell=True)

    s = s.decode().strip().split('\n')
    s = [i.split('\t') for i in s]
    variant_ids2 = [i[2] for i in s]
    if field=='GT':
        gt_ix = s[0][8].split(':').index('GT')
        dosages = [[gt_dosage_dict[j.split(':')[gt_ix]] for j in i[9:]] for i in s]
    elif field=='DS':
        ds_ix = s[0][8].split(':').index('DS')
        dosages = np.float32([[j.split(':')[ds_ix] for j in i[9:]] for i in s])
    df = pd.DataFrame(dosages, index=variant_ids2, columns=get_sample_ids(vcf))
    df = df[df.index.isin(variant_id_set)]
    df = df[~df.index.duplicated()]
    return df



def load_vcf(vcf):
    """Load dosages as DataFrame"""

    nvariants = int(subprocess.check_output('bcftools index -n {}'.format(vcf), shell=True).decode())
    sample_ids = subprocess.check_output('bcftools query -l {}'.format(vcf), shell=True).decode().strip().split()
    nsamples = len(sample_ids)
    dosages = np.zeros([nvariants, nsamples], dtype=np.float32)

    gt_dosage_dict['./1'] = np.NaN
    gt_dosage_dict['1/.'] = np.NaN

    with gzip.open(vcf, 'rt') as f:
        for line in f:
            if line[:4]=='#CHR':
                break

        # parse first line
        line = f.readline().strip().split('\t')
        assert 'GT' in line[8]
        gt_ix = line[8].split(':').index('GT')
        variant_ids = [line[2]]
        dosages[0,:] = [gt_dosage_dict[i.split(':')[gt_ix]] for i in line[9:]]

        for k,line in enumerate(f, 1):
            line = line.strip().split('\t')
            variant_ids.append(line[2])
            dosages[k,:] = [gt_dosage_dict[i.split(':')[gt_ix]] for i in line[9:]]
            if np.mod(k,1000)==0:
                print('\rVariants processed: {}'.format(k), end='')

    return pd.DataFrame(dosages, index=variant_ids, columns=sample_ids)

