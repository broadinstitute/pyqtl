import os
import tempfile
import subprocess
from pathlib import Path

#------------------------------------------------------------------------------
# Helper functions for lifting over data from hg19 to hg38 using UCSC liftOver.
# To download and install liftOver, see https://genome.ucsc.edu/store.html
#------------------------------------------------------------------------------
if os.getenv("CHAIN_FILE"):
    CHAIN_FILE = Path(os.getenv("CHAIN_FILE"))
else:
    CHAIN_FILE = Path("/mnt/disks/scratch/references/hg19ToHg38.patched.over.chain")
CHAIN_FILE = CHAIN_FILE if CHAIN_FILE.is_file() else None


def download_chain_file(dest_dir=None):
    """Download hg19ToHg38 chain file from UCSC and patch chromosome names"""
    if dest_dir is None:
        dest_dir = Path.cwd()
    source_file = "https://hgdownload.soe.ucsc.edu/goldenPath/hg19/liftOver/hg19ToHg38.over.chain.gz"
    dest_file = os.path.join(dest_dir, "hg19ToHg38.patched.over.chain")
    subprocess.check_call(f"wget -qO- {source_file} | zcat | sed 's/chr//' > {dest_file}", shell=True)
    return dest_file


def liftover_bed(bed_file, chain_file=CHAIN_FILE, out_file=None, header=False, delete_unmapped=True, overwrite=False):
    """Lift over BED file"""
    if out_file is None:
        out_file = bed_file.replace('.bed', '.GRCh38_liftover.bed')
        unmapped_file = out_file.replace('.GRCh38_liftover.bed', '.GRCh38_liftover.unmapped.txt')

    if bed_file.endswith('.bed'):
        if header:
            cmd = f"liftOver -bedPlus=3 -tab <(tail -n+2 {bed_file}) {chain_file} >(cat <(head -1 {bed_file}) - > {out_file}) {unmapped_file}"
        else:
            cmd = f"liftOver -bedPlus=3 -tab {bed_file} {chain_file} {out_file} {unmapped_file}"
    elif bed_file.endswith('.bed.gz'):
        if header:
            cmd = f"liftOver -bedPlus=3 -tab <(zcat {bed_file} | tail -n+2) {chain_file} >(cat <(zcat {bed_file} | head -1) - | bgzip -c > {out_file}) >(bgzip -c > {unmapped_file})"
        else:
            cmd = f"liftOver -bedPlus=3 -tab <(zcat {bed_file}) {chain_file} >(bgzip -c {out_file}) >(bgzip -c {unmapped_file})"
    else:
        raise ValueError('Unsupported input format (must be .bed or .bed.gz)')
    assert not os.path.exists(out_file) or overwrite
    assert not os.path.exists(unmapped_file) or overwrite
    subprocess.check_call(cmd, shell=True, executable='/bin/bash')
    if delete_unmapped:
        os.remove(unmapped_file)


def liftover_region(region_str, chain_file=CHAIN_FILE):
    """Lift over genomic region in the format chr:start-end."""
    chrom, pos = region_str.split(':')
    start, end = map(int, pos.split('-'))

    with tempfile.NamedTemporaryFile('wt') as f, tempfile.NamedTemporaryFile('wt') as f2, tempfile.NamedTemporaryFile('wt') as f3:
        f.write(f"{chrom}\t{start-1}\t{end}\n")
        f.flush()
        cmd = f"liftOver \
            -bedPlus=3 -tab \
            {f.name} \
            {chain_file} \
            {f2.name} \
            {f3.name}"
        subprocess.check_call(cmd, stderr=subprocess.DEVNULL, shell=True)
        with open(f2.name) as f:
            chrom, start, end = f.read().strip().split('\t')
    return f"{chrom}:{int(start)+1}-{end}"
