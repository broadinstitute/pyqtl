import pandas as pd
import numpy as np
import scipy as sp
import os
import subprocess
from . import io, liftover

#------------------------------------------------------------------------------
#  Helper functions for downloading, lifting over, and parsing UKB LD data
#------------------------------------------------------------------------------

def download_ld(region_str, dest_dir, source_dir="s3://broad-alkesgroup-ukbb-ld/UKBB_LD"):
    """
    Download UKB LD data

    Parameters
    ----------
    region_str : str
      chr14:100065517-102065517 -> chr14_100000001_103000001
    """
    lift_str = liftover.liftover_region(region_str, chain_file="/mnt/disks/scratch/references/hg38ToHg19.patched.over.chain")
    chrom_hg19, pos_hg19 = lift_str.split(':')
    start_hg19, end_hg19 = map(int, pos_hg19.split('-'))
    start = int(np.floor(start_hg19 / 1e6) * 1e6)
    end = int(np.ceil(end_hg19 / 1e6) * 1e6)
    if end - start < 3000000:
        end = start + 3000000
    ld_region = f"chr{chrom_hg19}_{start+1}_{end+1}"
    if not os.path.exists(os.path.join(dest_dir, f"{ld_region}.npz")):
        print(f"Downloading LD data for region {ld_region}")
        subprocess.check_call(f"aws s3 cp --no-sign-request {source_dir}/{ld_region}.npz {dest_dir}", shell=True)
        subprocess.check_call(f"aws s3 cp --no-sign-request {source_dir}/{ld_region}.gz {dest_dir}", shell=True)
    return ld_region


def liftover_ld(header_file):
    """Lift over UKB LD header file from hg19 to GRCh38 (creates *.GRCh38_liftover.bed.gz)"""
    header_df = pd.read_csv(header_file, sep='\t')
    header_df = header_df[['chromosome', 'position', 'allele1', 'allele2', 'rsid']]
    header_df.rename(columns={'chromosome':'chr'}, inplace=True)
    header_df['variant_id_b37'] = (header_df['chr'].astype(str) + '_' + header_df['position'].astype(str) + '_'
                                   + header_df['allele1'] + '_' + header_df['allele2'] + '_b37')
    header_df.rename(columns={'position':'start'}, inplace=True)
    header_df.insert(2, 'end', header_df['start'])
    header_df['start'] -= 1
    assert header_df.equals(io.sort_bed(header_df, inplace=False))
    header_bed = header_file.replace('.gz', '.bed.gz')
    io.write_bed(header_df, header_bed)
    liftover.liftover_bed(header_bed, out_file=None, header=True, delete_unmapped=True, overwrite=True)
    # delete hg19 BED
    os.remove(header_bed)
    os.remove(header_bed+'.tbi')


def load_ld(ld_file):
    """Load UKB LD matrix using lifted over GRCh38 header"""
    # load matrix
    header_df = pd.read_csv(ld_file.replace('.npz', '.GRCh38_liftover.bed.gz'), sep='\t')
    header_df.rename(columns={'#chr':'chr', 'end':'position'}, inplace=True)
    header_df.drop('start', axis=1, inplace=True)
    header_df['variant_id'] = (header_df['chr'] + '_' + header_df['position'].astype(str) + '_'
                               + header_df['allele1'] + '_' + header_df['allele2'] + '_b38')

    m2 = sp.sparse.load_npz(ld_file)
    m = m2.todense()
    m += m.T
    ld_df = pd.DataFrame(m, index=header_df['variant_id'], columns=header_df['variant_id'])
    return ld_df
