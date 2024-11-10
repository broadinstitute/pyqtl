import numpy as np
import pandas as pd
import os
import tempfile
import copy
import subprocess
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.path as mpath
from matplotlib.colors import hsv_to_rgb
import gzip
import scipy.interpolate as interpolate
from collections import defaultdict
import pyBigWig
from bx.intervals.intersection import IntervalTree


def format_plot(ax, tick_direction='out', tick_length=4, hide=['top', 'right'], lw=1, fontsize=9):

    for i in ['left', 'bottom', 'right', 'top']:
        ax.spines[i].set_linewidth(lw)

    ax.tick_params(axis='both', which='both', direction=tick_direction, labelsize=fontsize)

    if 'left' in hide and 'right' in hide:
        ax.get_yaxis().set_ticks_position('none')
    elif 'left' in hide:
        ax.get_yaxis().set_ticks_position('right')
    elif 'right' in hide:
        ax.get_yaxis().set_ticks_position('left')
    else:
        ax.get_yaxis().set_ticks_position('both')

    for i in hide:
        ax.spines[i].set_visible(False)


def interval_union(intervals):
    """
    Calculate union of intervals
      intervals: list of tuples or 2-element lists
    """
    intervals.sort(key=lambda x: x[0])
    union = [intervals[0]]
    for i in intervals[1:]:
        if i[0] <= union[-1][1]:  # overlap w/ previous
            if i[1] > union[-1][1]:  # only extend if larger
                union[-1][1] = i[1]
        else:
            union.append(i)
    return np.array(union)


def load_fasta(fasta_file, header_parser=lambda x: x.split()[0]):
    if fasta_file.endswith('.gz'):
        opener = gzip.open(fasta_file, 'rt')
    else:
        opener = open(fasta_file, 'r')

    with opener as f:
        s = f.read().strip()
    s = [i.split('\n', 1) for i in s.split('>')[1:]]
    return {header_parser(i[0]):i[1].replace('\n','') for i in s}


def get_sequence(fasta, region_str, concat=False):
    """Get sequence corresponding to region_str (chr:start-end) or list of region_str"""
    if isinstance(region_str, str):
        s = subprocess.check_output(f'samtools faidx {fasta} {region_str} -n 1000000000',
                                    shell=True).decode().strip().split('\n')
        return s[1]
    else:  # list of region_str
        with tempfile.NamedTemporaryFile(mode='w+t') as f:
            f.write('\n'.join(region_str)+'\n')
            f.flush()
            s = subprocess.check_output(f'samtools faidx {fasta} -r {f.name} -n 1000000000',
                                        shell=True).decode().strip().split('\n')
        assert [i[1:] for i in s[::2]] == region_str
        s = s[1::2]
        if concat:
            s = ''.join(s)
        return s


def get_coord_transform(gene, max_intron=1000):
    """Interpolation function for exon/intron coordinates"""

    ce = gene.get_collapsed_coords()
    exon_lengths = ce[:,1]-ce[:,0]+1
    intron_lengths = ce[1:,0]-ce[:-1,1]-1
    # transformed_intron_lengths = np.sqrt(intron_lengths)
    transformed_intron_lengths = intron_lengths.copy()
    if max_intron is not None:
        transformed_intron_lengths[transformed_intron_lengths>max_intron] = max_intron

    coords = np.array([[d+0, d+e-1] for e,d in zip(exon_lengths, np.cumsum(np.r_[0, exon_lengths[:-1]+intron_lengths]))]).reshape(1,-1)[0]
    icoords = np.array([[d+0, d+e-1] for e,d in zip(exon_lengths, np.cumsum(np.r_[0, exon_lengths[:-1]+transformed_intron_lengths]))]).reshape(1,-1)[0]
    ifct = interpolate.interp1d(coords, icoords, kind='linear')
    return ifct


def reverse_complement(s):
    return s.translate(str.maketrans('ATCG', 'TAGC'))[::-1]


def _str_to_pos(region_str):
    s, e = region_str.split(':')[-1].split('-')
    return int(s), int(e)


class Exon(object):
    """Exon"""
    def __init__(self, exon_id, number, transcript, start_pos, end_pos):
        self.id = exon_id
        self.number = int(number)
        self.transcript = transcript
        self.start_pos = start_pos
        self.end_pos = end_pos
        self.length = end_pos-start_pos+1

    def __str__(self, ref=1):
        return f'exon_id: {self.id}; exon_number: {self.number:2d}'\
            + f'; pos.: {self.start_pos-ref+1:d}-{self.end_pos-ref+1:d}'\
            + f'; length: {self.length:d}'

    def __len__(self):
        return self.length

    def __eq__(self, other):
        return (self.start_pos, self.end_pos) == (other.start_pos, other.end_pos)

    def __lt__(self, other):
        return self.start_pos < other.start_pos or (self.start_pos == other.start_pos and self.end_pos < other.end_pos)

    def __gt__(self, other):
        return self.start_pos > other.start_pos or (self.start_pos == other.start_pos and self.end_pos > other.end_pos)

    def __le__(self, other):
        return self.start_pos < other.start_pos or (self.start_pos == other.start_pos and self.end_pos <= other.end_pos)

    def __ge__(self, other):
        return self.start_pos > other.start_pos or (self.start_pos == other.start_pos and self.end_pos >= other.end_pos)

    def __ne__(self, other):
        return self.start_pos != other.start_pos or self.end_pos != other.end_pos


class Transcript(object):
    """Represents a transcripts and its exons"""
    def __init__(self, transcript_id, transcript_name, transcript_type, gene, start_pos, end_pos):
        self.id = transcript_id
        self.name = transcript_name
        self.type = transcript_type
        self.gene = gene
        self.start_pos = start_pos
        self.end_pos = end_pos
        self.exons = []
        self.start_codon = []
        self.stop_codon = []
        self.utr5 = []
        self.utr3 = []

    def __str__(self, ref=1):
        """Print text representation of transcript structure"""
        rep = [f'Transcript: {self.id} ({self.name}): {self.type}' +\
            f'; {self.gene.chr}:{self.start_pos-ref+1:d}-{self.end_pos-ref+1:d} ({self.gene.strand})' +\
            f'; length: {sum([e.length for e in self.exons]):d}']
        rep += ['    '+i.__str__(ref) for i in self.exons]
        return '\n'.join(rep)

    def __eq__(self, other):
        return (self.id == other.id and
                len(self.exons) == len(other.exons) and
                np.all([i == j for i,j in zip(self.exons, other.exons)]))

    def get_cds_ranges(self, include_stop=True):
        cds_ranges = [e.CDS for e in self.exons if hasattr(e, 'CDS')]
        if cds_ranges and include_stop and len(self.stop_codon) > 0:
            cds_ranges.append(self.stop_codon[::2])
        return cds_ranges

    def get_cds_coords(self, include_stop=True):
        """Get genomic coordinates of coding sequence, including stop codon."""
        cds_ranges = self.get_cds_ranges(include_stop=include_stop)
        if self.gene.strand == '+':
            return np.concatenate([np.arange(r[0], r[1]+1) for r in cds_ranges])
        else:
            return np.concatenate([np.arange(r[1], r[0]-1, -1) for r in cds_ranges])

    def get_sequence(self, feature='all', fasta_dict=None, fasta=None, include_stop=False):
        """Load transcript sequence from FASTA file"""
        if fasta is not None:
            if feature.lower() == 'all':
                region_strs = [f"{self.gene.chr}:{e.start_pos}-{e.end_pos}" for e in self.exons]
            elif feature.lower() == 'utr3':
                if len(self.utr3) > 0:
                    region_strs = [f"{self.gene.chr}:{r[0]}-{r[1]}" for r in self.utr3]
                else:
                    return None
            elif feature.lower() == 'utr5':
                if len(self.utr5) > 0:
                    region_strs = [f"{self.gene.chr}:{r[0]}-{r[1]}" for r in self.utr5]
                else:
                    return None
            elif feature.lower() == 'cds':
                cds_ranges = self.get_cds_ranges(include_stop=include_stop)
                region_strs = [f"{self.gene.chr}:{r[0]}-{r[1]}" for r in cds_ranges]
            else:
                raise ValueError('Unsupported feature. Options: all, cds, utr5, utr3.')
            if self.gene.strand == '-':
                region_strs = region_strs[::-1]
            s = get_sequence(fasta, region_strs, concat=True)
        #elif hasattr(self.gene, 'annotation') and self.gene.annotation.fasta_dict is not None:
        elif fasta_dict is not None:
            if feature.lower() == 'all':
                s = [fasta_dict[self.gene.chr][e.start_pos-1:e.end_pos] for e in self.exons]
                if self.gene.strand == '-':
                    s = s[::-1]
                s = ''.join(s)
            elif feature.lower() == 'utr3':
                if len(self.utr3) > 0:
                    s = [fasta_dict[self.gene.chr][r[0]-1:r[1]] for r in self.utr3]
                    if self.gene.strand == '-':
                        s = s[::-1]
                    s = ''.join(s)
                else:
                    return None
            elif feature.lower() == 'utr5':
                if len(self.utr5) > 0:
                    s = [fasta_dict[self.gene.chr][r[0]-1:r[1]] for r in self.utr5]
                    if self.gene.strand == '-':
                        s = s[::-1]
                    s = ''.join(s)
                else:
                    return None
            elif feature.lower() == 'cds':
                cds_ranges = self.get_cds_ranges(include_stop=include_stop)
                s = [fasta_dict[self.gene.chr][r[0]-1:r[1]] for r in cds_ranges]
                if self.gene.strand == '-':
                    s = s[::-1]
                s = ''.join(s)
            else:
                raise ValueError('Unsupported feature. Options: all, cds, utr5, utr3.')
        else:
            raise ValueError('Reference genome FASTA must either be loaded with annotation.load_fasta() or provided as input.')

        if self.gene.strand == '-':
            s = reverse_complement(s)

        return s

class Gene(object):
    def __init__(self, gene_id, gene_name, gene_type, chrom, strand, start_pos, end_pos, transcript_list=None):
        self.id = gene_id
        self.name = gene_name
        self.type = gene_type
        self.chr = chrom
        self.strand = strand
        self.havana_id = '-'
        self.start_pos = start_pos
        self.end_pos = end_pos
        if strand == '+':
            self.tss = start_pos
        else:
            self.tss = end_pos
        self.transcripts = []
        self.exclude_biotypes = []
        self.mappability = None
        if transcript_list:
            self.set_transcripts(transcript_list)

    def __str__(self, ref=1):
        """Print gene/isoform structure"""
        rep = f'Gene: {self.name} ({self.id}): {self.type}; ' +\
             f"{self.chr}:{self.start_pos-ref+1:d}-{self.end_pos-ref+1:d} ({self.strand})"
        if len(self.transcripts) > 1:
            rep = rep + f'; {len(self.transcripts)} isoforms'
        if isinstance(self.mappability, float):
            rep = rep + f'; Mappability: {self.mappability:.4f}'
        rep = [rep] + [i.__str__(ref) for i in self.transcripts]
        return '\n'.join(rep)

    def __eq__(self, other):
        return (self.id == other.id and
                len(self.transcripts) == len(other.transcripts) and
                np.all([i == j for i,j in zip(self.transcripts, other.transcripts)]))

    def get_coverage(self, bigwig):
        """Returns coverage for the genomic region spanned by the gene"""
        with pyBigWig.open(bigwig) as bw:
            # pyBigWig returns values using BED intervals, e.g., in [start, end)
            c = bw.values(self.chr, self.start_pos-1, self.end_pos, numpy=True)
        return c

    def get_collapsed_coords(self):
        """Returns coordinates of collapsed exons (= union of exons)"""
        ecoord = [[e.start_pos, e.end_pos] for t in self.transcripts for e in t.exons if t.type not in self.exclude_biotypes]
        return interval_union(ecoord)

    def collapse(self, exclude_biotypes=['readthrough_transcript', 'retained_intron']):
        """Return collapsed version of the gene"""
        g = copy.deepcopy(self)
        g.exclude_biotypes = exclude_biotypes
        transcripts = [t for t in g.transcripts if t.type not in exclude_biotypes]
        start_pos = min([t.start_pos for t in transcripts])
        end_pos = max([t.end_pos for t in transcripts])
        t = Transcript(g.id, g.name, g.type, g, start_pos, end_pos)
        exon_coords = g.get_collapsed_coords()
        if g.strand == '-':
            exon_coords = exon_coords[::-1]
        t.exons = [Exon(f"{g.id}_{k}", k, t, i[0], i[1]) for k,i in enumerate(exon_coords, 1)]
        g.set_transcripts([t])
        return g

    def set_transcripts(self, transcripts):
        self.transcripts = transcripts
        self.start_pos = np.min([t.start_pos for t in transcripts])
        self.end_pos = np.max([t.end_pos for t in transcripts])

    def set_plot_coords(self, max_intron=1000, reference=None):
        """"""
        if reference is None:
            # self.reference = self.start_pos
            self.reference = np.min([t.start_pos for t in self.transcripts if t.type not in self.exclude_biotypes])
        else:
            self.reference = reference

        # cumulative lengths of exons and introns
        self.ce = self.get_collapsed_coords()
        exon_lengths = self.ce[:,1] - self.ce[:,0] + 1
        intron_lengths = np.r_[0, self.ce[1:,0] - self.ce[:-1,1] - 1]
        cumul_len = np.zeros(2*len(exon_lengths), dtype=np.int32)
        cumul_len[0::2] = intron_lengths
        cumul_len[1::2] = exon_lengths

        cumul_len_adj = cumul_len.copy()
        cumul_len_adj[2::2] = np.minimum(cumul_len_adj[2::2], max_intron)

        cumul_len = np.cumsum(cumul_len)

        # # adjusted lengths, truncating long introns
        # cumul_len_adj = np.zeros(2*len(exon_lengths), dtype=np.int32)
        # if max_intron is not None:
        #     intron_lengths[intron_lengths > max_intron] = max_intron
        # cumul_len_adj[0::2] = intron_lengths
        # cumul_len_adj[1::2] = exon_lengths
        cumul_len_adj = np.cumsum(cumul_len_adj)

        self.cumul_len = cumul_len
        self.cumul_len_adj = cumul_len_adj
        # self.map_pos = lambda x: np.interp(x - self.start_pos, cumul_len, cumul_len_adj) + self.reference

    def map_pos(self, x):
        start_pos = np.min([t.start_pos for t in self.transcripts if t.type not in self.exclude_biotypes])
        end_pos = np.max([t.end_pos for t in self.transcripts if t.type not in self.exclude_biotypes])

        map_fct = lambda x: np.interp(x - start_pos, self.cumul_len, self.cumul_len_adj) + self.reference

        x = np.array(x)

        y = np.zeros(x.shape, like=x)
        m = (start_pos <= x) & (x <= end_pos)
        y[m] = map_fct(x[m])
        m = x < start_pos
        y[m] = x[m] - start_pos + self.reference
        m = x > end_pos
        y[m] = x[m] - end_pos + map_fct(end_pos)

        return y

    def plot(self, coverage=None, max_intron=1000, scale=0.4, ax=None, show_cds=True, highlight_region=None,
             pc_color=[0.6, 0.88, 1], nc_color='#aaaaaa', ec=[0, 0.7, 1], wx=0.05, reference=None, ylabels='id',
             highlight_exons=None, highlight_introns=None, highlight_introns2=None,
             highlight_color='tab:red', clip_on=False, yoffset=0, xlim=None,
             highlight_transcripts=None, exclude_biotypes=[]):
        """
        Plots the transcript structure of the gene.

        highlight_introns : (list of) string(s) defining intron coordinates (first to last base of intron)
        """

        transcripts = [t for t in self.transcripts if t.type not in exclude_biotypes]
        self.exclude_biotypes = exclude_biotypes

        max_intron = int(max_intron)
        if reference is None:
            reference = np.min([t.start_pos for t in transcripts])
        self.set_plot_coords(max_intron=max_intron, reference=reference)

        axes_input = True
        if ax is None:
            axes_input = False

            ah = len(transcripts) * 0.275
            aw = 7
            db = 0.3
            dt = 0.3
            dl = 1.6
            dr = 2
            fh = db + ah + dt
            fw = dl + aw + dr
            if coverage is not None:
                ch = 0.6
                fh += ch + 0.1
            fig = plt.figure(facecolor=(1,1,1), figsize=(fw,fh))
            ax = fig.add_axes([dl/fw, db/fh, aw/fw, ah/fh])
            ax.margins(x=0)
            if coverage is not None:
                ac = fig.add_axes([dl/fw, (db+ah+0.1)/fh, aw/fw, ch/fh], sharex=ax)

        if highlight_exons is not None:
            if isinstance(highlight_exons, str):
                highlight_exons = {_str_to_pos(i) for i in highlight_exons.split(',')}
            else:
                highlight_exons = {_str_to_pos(i) for i in highlight_exons}

        if highlight_introns is not None:
            if isinstance(highlight_introns, str):
                highlight_introns = {_str_to_pos(highlight_introns)}
            else:
                highlight_introns = {_str_to_pos(i) for i in highlight_introns}

        if highlight_introns2 is not None:
            if isinstance(highlight_introns2, str):
                highlight_introns2 = {_str_to_pos(highlight_introns2)}
            else:
                highlight_introns2 = {_str_to_pos(i) for i in highlight_introns2}

        if highlight_transcripts is not None and isinstance(highlight_transcripts, str):
            highlight_transcripts = [highlight_transcripts]

        def get_vertices(start, end, utr5=None, utr3=None):
            # utr5 and utr3 are defined relative to + strand here
            # get transformed coordinates for plotting exons
            assert end >= start
            if utr5 is not None:
                assert utr5[1] >= utr5[0]
            if utr3 is not None:
                assert utr3[1] >= utr3[0]

            if utr5 is None and utr3 is None:
                x = [start, end]
                y = [1, 1]
            elif utr5 is not None and utr3 is not None:
                if start > utr5[1] and end < utr3[0]:
                    x = [start, end]
                    y = [1, 1]
                elif (utr5[0] <= start and end <= utr5[1]) or (utr3[0] <= start and end <= utr3[1]):
                    x = [start, end]
                    y = [0.5, 0.5]
                elif utr5[0] <= start and start <= utr5[1] and utr3[0] <= end and end <= utr3[1]:
                    x = [start, utr5[1]+1, utr5[1]+1, utr3[0]-1, utr3[0]-1, end]
                    y = [0.5, 0.5, 1, 1, 0.5, 0.5]
                elif utr5[0] <= start and start <= utr5[1] and utr5[1] < end and end < utr3[0]:
                    x = [start, utr5[1]+1, utr5[1]+1, end]
                    y = [0.5, 0.5, 1, 1]
                elif utr5[1] < start and start < utr3[0] and utr3[0] <= end and end <= utr3[1]:
                    x = [start, utr3[0]-1, utr3[0]-1, end]
                    y = [1, 1, 0.5, 0.5]
                else:
                    raise ValueError('Unexpected vertices')
            elif utr5 is not None:
                if start > utr5[1]:
                    x = [start, end]
                    y = [1, 1]
                elif utr5[0] <= start and end <= utr5[1]:
                    x = [start, end]
                    y = [0.5, 0.5]
                elif utr5[0] <= start and start <= utr5[1] and end > utr5[1]:
                    x = [start, utr5[1]+1, utr5[1]+1, end]
                    y = [0.5, 0.5, 1, 1]
            elif utr3 is not None:
                if end < utr3[0]:
                    x = [start, end]
                    y = [1, 1]
                if utr3[0] <= start and end <= utr3[1]:
                    x = [start, end]
                    y = [0.5, 0.5]
                elif start < utr3[0] and utr3[0] <= end and end <= utr3[1]:
                    x = [start, utr3[0]-1, utr3[0]-1, end]
                    y = [1, 1, 0.5, 0.5]

            x = self.map_pos(x)
            x = np.r_[x, x[::-1]]
            y = np.r_[y, -np.array(y[::-1])]
            return x, y

        # plot transcripts; positions are in genomic coordinates
        for i,t in enumerate(transcripts[::-1], yoffset):
            # plot background line
            s = self.map_pos(t.start_pos)
            e = self.map_pos(t.end_pos)
            y = i - wx/2
            # background line
            if highlight_transcripts is not None and t.id in highlight_transcripts:
                patch = patches.Rectangle((s, y), e-s, wx, fc='k', zorder=0, clip_on=clip_on)
            else:
                patch = patches.Rectangle((s, y), e-s, wx, fc=pc_color if t.type == 'protein_coding' else nc_color, zorder=0, clip_on=clip_on)
            ax.add_patch(patch)

            # plot highlighted introns
            if highlight_introns is not None or highlight_introns2 is not None:
                if self.strand == '+':
                    introns = {(t.exons[i].end_pos+1, t.exons[i+1].start_pos-1) for i in range(len(t.exons)-1)}
                else:
                    introns = {(t.exons[i+1].end_pos+1, t.exons[i].start_pos-1) for i in range(len(t.exons)-1)}

                if highlight_introns is not None:
                    for ic in highlight_introns:
                        if ic in introns:
                            s = self.map_pos(ic[0]-1)+1
                            e = self.map_pos(ic[1]+1)-1
                            patch = patches.Rectangle((s, y), e-s, wx, fc='lightskyblue', zorder=1, clip_on=clip_on)
                            ax.add_patch(patch)

                if highlight_introns2 is not None:
                    for ic in highlight_introns2:
                        if ic in introns:
                            s = self.map_pos(ic[0]-1)+1
                            e = self.map_pos(ic[1]+1)-1
                            patch = patches.Rectangle((s, y), e-s, wx, fc='#ffad33', zorder=1, clip_on=clip_on)
                            ax.add_patch(patch)

            # plot exons
            # utrs = t.utr5 + t.utr3
            for e in t.exons:
                # check for overlapping UTRs
                utr5 = [i for i in t.utr5 if i[0] == e.start_pos or i[1] == e.end_pos]
                utr3 = [i for i in t.utr3 if i[0] == e.start_pos or i[1] == e.end_pos]
                if self.strand == '-':
                    utr5, utr3 = utr3, utr5
                if len(utr5) == 0:
                    utr5 = None
                else:
                    assert len(utr5) == 1
                    utr5 = utr5[0]
                if len(utr3) == 0:
                    utr3 = None
                else:
                    assert len(utr3) == 1
                    utr3 = utr3[0]
                x, y = get_vertices(e.start_pos, e.end_pos, utr5=utr5, utr3=utr3)
                if not show_cds:
                    y[np.abs(y) == 0.5] *= 2
                # u = [i for i in utrs if i[0] == e.start_pos or i[1] == e.end_pos]
                # if u:
                #     assert len(u) == 1
                #     u = u[0]
                #     x, y = get_vertices(e.start_pos, e.end_pos, utr=u)
                # else:
                #     u = None
                #     x, y = get_vertices(e.start_pos, e.end_pos)

                #     if u[0] == e.start_pos and u[1] == e.end_pos:
                #         x = u
                #         y = [0.5, 0.5]
                #     elif u[0] == e.start_pos and u[1] < e.end_pos:
                #         x = u + [u[1], e.end_pos]
                #         y = [0.5, 0.5, 1, 1]
                #     elif u[0] > e.start_pos and u[1] == e.end_pos:
                #         x = [e.start_pos, u[0]] + u
                #         y = [1, 1, 0.5, 0.5]
                #     else:
                #         raise ValueError(f"Misspecified UTR in {t.id}")
                # else:
                #     x = [e.start_pos, e.end_pos]
                #     y = [1, 1]
                # x = self.map_pos(x)
                # x = np.r_[x, x[::-1]]
                # y = i+scale*np.r_[y, -np.array(y[::-1])]
                y = i + scale*y
                ax.add_patch(patches.PathPatch(mpath.Path(np.c_[x, y], closed=False), ec='none', lw=0,
                                               fc=pc_color if t.type == 'protein_coding' else nc_color,
                                               zorder=2, clip_on=clip_on))

                if highlight_exons is not None:
                    match = [i for i in highlight_exons if e.start_pos <= i[0] and i[1] <= e.end_pos]
                    if match:
                        assert len(match) == 1
                        match = match[0]
                        x, y = get_vertices(match[0], match[1], utr5=utr5, utr3=utr3)
                        y = i + scale*y
                        ax.add_patch(patches.PathPatch(mpath.Path(np.c_[x, y], closed=False), ec='none', lw=0,
                                                       fc=highlight_color,
                                                       zorder=3, clip_on=clip_on))

                # ev = np.ones(e.end_pos-e.start_pos+1)  # height
                # ev[utr[e.start_pos-t.start_pos:e.end_pos-t.start_pos+1] == 1] = 0.5  # UTRs
                # ex = np.arange(self.map_pos(e.start_pos), self.map_pos(e.end_pos)+1)  # position
                # vertices = np.vstack((np.hstack((ex, ex[::-1], ex[0])), i+scale*np.hstack((ev,-ev[::-1], ev[0])))).T

                # if highlight_exons is not None:
                #     if (e.start_pos, e.end_pos) in highlight_exons:  # entire exon
                #         ax.add_patch(patches.PathPatch(mpath.Path(vertices, closed=True), fc=highlight_color, ec='none', lw=0, zorder=2, clip_on=clip_on))
                #     else:
                #         ax.add_patch(patches.PathPatch(mpath.Path(vertices, closed=True), ec='none', lw=0,
                #                                        fc=fc if t.type == 'protein_coding' else 'darkgray',
                #                                        zorder=2, clip_on=clip_on))
                #         # x = [x for x in highlight_exons if x[0] >= e.start_pos and x[1] <= e.end_pos]
                #         # if x:
                #         #     x =
                #         #     ax.add_patch(patches.PathPatch(mpath.Path(vertices, closed=True), fc=highlight_color, ec='none', lw=0, zorder=2, clip_on=clip_on))
                #
                # elif highlight_transcripts is not None and t.id in highlight_transcripts:
                #     ax.add_patch(patches.PathPatch(mpath.Path(vertices, closed=True), fc=highlight_color, ec='none', lw=0, zorder=2, clip_on=clip_on))
                # else:
                #     ax.add_patch(patches.PathPatch(mpath.Path(vertices, closed=True), ec='none', lw=0,
                #                                    fc=fc if t.type == 'protein_coding' else 'darkgray',
                #                                    zorder=2, clip_on=clip_on))

        if highlight_region is not None:
            s,e = highlight_region.split(':')[-1].split('-')
            s = self.map_pos(int(s))
            e = self.map_pos(int(e))
            patch = patches.Rectangle((s, -0.5), e-s, len(transcripts), fc=hsv_to_rgb([0, 0.8, 1]), alpha=0.5, zorder=-10, clip_on=clip_on)
            ax.add_patch(patch)

        ax.set_ylim([-0.6, i+0.6])

        #ax.set_xlim([self.ce[0][0], self.ce[-1][1]])
        # ax.set_xlim(self.map_pos([np.min([t.start_pos for t in transcripts]), np.max([t.end_pos for t in transcripts])]))
        if xlim is not None:
            ax.set_xlim(xlim)
        else:
            xlim = ax.get_xlim()
            if xlim[0] == 0 and xlim[1] == 1:
                xlim = self.map_pos(np.array([self.start_pos, self.end_pos]))
                if not axes_input:
                    xlim = [xlim[0]-150, xlim[1]+150]
                ax.set_xlim(xlim)

        if ylabels is not None:
            ax.set_yticks(range(len(transcripts)))
            yticklabels = [getattr(t, ylabels) for t in transcripts[::-1]]
            ax.set_yticklabels(yticklabels, fontsize=9)
            mane_transcript = self.get_mane_transcript()
            if mane_transcript is not None:
                ax.get_yticklabels()[yticklabels.index(getattr(mane_transcript, ylabels))].set_color('tab:red')

        if not axes_input:
            start_pos = np.min([t.start_pos for t in self.transcripts if t.type not in self.exclude_biotypes])
            end_pos = np.max([t.end_pos for t in self.transcripts if t.type not in self.exclude_biotypes])
            x = self.map_pos([start_pos, end_pos])
            ax.set_xticks(x)
            ax.set_xticklabels([f"{start_pos:,}", f"{end_pos:,}"], ha='center', fontsize=9)
        #     ax.set_xticks(self.map_pos(np.array([self.start_pos, self.end_pos])))
        #     ax.set_xticklabels([self.start_pos, self.end_pos], ha='center', fontsize=9)
            # add transcript type label
            ax2 = ax.twinx()
            ax2.set_ylim([-0.6, i+0.6])
            ax2.set_yticks(range(len(transcripts)))
            ax2.set_yticklabels([t.type.replace('_', ' ').capitalize() for t in transcripts[::-1]], ha='left', fontsize=9)
            format_plot(ax2, tick_length=4, hide=['top', 'left', 'right'])
            format_plot(ax, tick_length=4, hide=['top', 'left', 'right'])

        if coverage is not None:
            self.plot_coverage(coverage, ac, max_intron=max_intron)
            ac.set_title(f"{self.name} ({self.id})", fontsize=12)
        elif not axes_input:
            ax.set_title(f"{self.name} ({self.id})", fontsize=12)
            ax.set_xlabel(self.chr, fontsize=12)

        return ax

    def get_mane_transcript(self, tag='MANE_Select'):
        mane_transcript = [t for t in self.transcripts if 'tags' in t.attributes and tag in t.attributes['tags']]
        assert len(mane_transcript) <= 1
        if len(mane_transcript) == 1:
            return mane_transcript[0]
        else:
            return None

    def plot_coverage(self, coverage, ax, color=3*[0.66], max_intron=1000):
        # only plot first max_intron bases of introns
        # if not self.ce[-1][1] - self.ce[0][0] + 1 == len(coverage):
        #     raise ValueError(f'Coverage ({len(coverage)}) does not match gene length ({self.ce[-1][1]-self.ce[0][0]+1})')
        ax.margins(0)
        # coordinates:
        # pidx = [np.arange(self.ce[0][0], self.ce[0][1]+1)]
        # for i in range(1, self.ce.shape[0]):
        #     li = np.minimum(self.ce[i,0]-1 - self.ce[i-1,1], max_intron)
        #     ri = np.arange(self.ce[i-1,1]+1, self.ce[i-1,1]+1 + li)
        #     pidx.append(ri)
        #     pidx.append(np.arange(self.ce[i][0], self.ce[i][1]+1))
        # pidx = np.concatenate(pidx)
        # pidx = pidx-pidx[0]

        # x = np.arange(len(pidx))
        x = self.map_pos(np.arange(self.start_pos, self.end_pos+1))
        if len(coverage.shape) == 1:
            ax.fill_between(x, coverage, edgecolor='none', facecolor=color)
        else:
            ax.plot(x, coverage)
        format_plot(ax, tick_length=4, hide=['top', 'right'])
        plt.setp(ax.get_xticklabels(), visible=False)
        for line in ax.xaxis.get_ticklines():
            line.set_markersize(0)
            line.set_markeredgewidth(0)

    def plot_junctions(self, ax, junction_df, coverage_s, show_counts=True, count_col='count',
                       align='both', h=0.3, lw=3, lw_fct=np.sqrt, ec=[0.6]*3, clip_on=True):
        """
        Plot junctions ("sashimi" plot).

        junction_df : pd.DataFrame, must contain count_col and
                      'start' and 'end' columns corresponding to intron coordinates
        """

        # new axes
        fig = ax.get_figure()
        pos = ax.get_position()
        aw, ah = pos.bounds[2:] * fig.get_size_inches()
        ax0 = fig.add_axes(pos, fc='none')
        ax0.set_xlim([0, 1])
        ys = ah/aw  # correct for aspect ratio
        ax0.set_ylim([0, ys])
        for k in ax0.spines:
            ax0.spines[k].set_visible(False)
        ax0.set_xticks([])
        ax0.set_yticks([])
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()

        for _,r in junction_df.iterrows():
            x1 = np.array([self.map_pos(r['start']), coverage_s[r['start']-1]])
            x2 = np.array([self.map_pos(r['end']),   coverage_s[r['end']+1]])

            # relative position
            dx = xlim[1] - xlim[0]
            dy = ylim[1] - ylim[0]
            x1_t = np.array([(x1[0]-xlim[0])/dx, (x1[1]-ylim[0])/dy*ys])
            x2_t = np.array([(x2[0]-xlim[0])/dx, (x2[1]-ylim[0])/dy*ys])
            if align == 'minimum':
                ymin = np.minimum(x1_t[1], x2_t[1])
                x1_t[1] = ymin
                x2_t[1] = ymin

            _plot_arc(ax0, x1_t, x2_t, h*ys, lw=lw * lw_fct(r[count_col]/junction_df[count_col].max()), ec=ec, clip_on=clip_on)

            if show_counts:
                x0_t = (x1_t + x2_t) / 2
                txt = ax0.text(x0_t[0], x0_t[1]+h*ys/2, f"{r[count_col]:.2f}".rstrip('0').rstrip('.'), ha='center', va='center', clip_on=clip_on)
                txt.set_bbox(dict(facecolor='w', alpha=0.75, edgecolor='none', boxstyle="round,pad=0.1"))


def _plot_arc(ax, x1, x2, h, lw=1, ec='k', clip_on=True):
    x0 = (x1 + x2) / 2
    d = x2 - x1
    t = np.arctan(d[1]/d[0])
    ax.add_patch(patches.Arc(x0, d[0] / np.cos(t), h, lw=lw, ec=ec, clip_on=clip_on,
                             angle=t*180/np.pi, theta1=0, theta2=180))


def get_attributes(attr_str):
    attributes = defaultdict()
    for a in attr_str.replace('"', '').split(';')[:-1]:
        kv = a.strip().split(' ')
        if kv[0] != 'tag':
            attributes[kv[0]] = kv[1]
        else:
            attributes.setdefault('tags', []).append(kv[1])
    return attributes


def write_attributes(attr_dict):
    s = []
    for k,v in attr_dict.items():
        if k == 'tags':
            for t in v:
                s.append(f'tag "{t}";')
        elif k == 'level':
            s.append(f'{k} {v};')
        else:
            s.append(f'{k} "{v}";')
    return ' '.join(s)


class Annotation(object):

    def __init__(self, varin, verbose=True):
        """
        Parse annotation from GTF file and build gene/transcript/exon object hierarchy
        """

        self.gene_dict = defaultdict()
        self.genes = []
        self.transcript_dict = defaultdict()
        self.transcripts = []
        self.gene_ids = []
        self.gene_names = []
        self.header = []
        self.fasta_dict = None

        if isinstance(varin, list):
            self.genes = varin
        elif isinstance(varin, str):  # load from GTF
            gtfpath = varin

            if gtfpath.endswith('.gz'):
                opener = gzip.open(gtfpath, 'rt')
            else:
                opener = open(gtfpath, 'r')

            with opener as gtf:
                for row in gtf:
                    if row[0] == '#':
                        self.header.append(row.strip())
                        continue

                    row = row.strip().split('\t')
                    chrom = row[0]
                    # source = row[1]
                    annot_type = row[2]
                    start_pos = int(row[3])
                    end_pos = int(row[4])
                    # row[5] is always '.'
                    strand = row[6]
                    # phase = row[7]

                    attributes = get_attributes(row[8])

                    if annot_type == 'gene':
                        gene_id = attributes['gene_id']
                        g = Gene(gene_id, attributes['gene_name'], attributes['gene_type'], chrom, strand, start_pos, end_pos)
                        g.source = row[1]
                        g.phase = row[7]
                        g.attributes_string = row[8]
                        if 'havana_gene' in attributes.keys():
                            g.havana_id = attributes['havana_gene']
                        self.gene_dict[gene_id] = g
                        self.gene_ids.append(gene_id)
                        self.gene_names.append(attributes['gene_name'])
                        self.genes.append(g)

                    elif annot_type == 'transcript':
                        transcript_id = attributes['transcript_id']
                        if 'transcript_name' not in attributes:
                            attributes['transcript_name'] = attributes['transcript_id']
                        if 'transcript_type' not in attributes:
                            attributes['transcript_type'] = 'unknown'
                        t = Transcript(attributes.pop('transcript_id'),
                                       attributes.pop('transcript_name'),
                                       attributes.pop('transcript_type'),
                                       g, start_pos, end_pos)
                        t.source = row[1]
                        t.attributes = attributes
                        t.attributes_string = row[8]
                        g.transcripts.append(t)
                        self.transcript_dict[transcript_id] = t
                        self.transcripts.append(t)

                    elif annot_type == 'exon':
                        if 'exon_id' in attributes:
                            e = Exon(attributes['exon_id'], attributes['exon_number'], t, start_pos, end_pos)
                        else:
                            e = Exon(str(len(t.exons)+1), len(t.exons)+1, t, start_pos, end_pos)
                        e.attributes_string = row[8]
                        t.exons.append(e)

                    # UTRs may span multiple exons and are separately annotated for each
                    # The order of UTRs in the annotation is always 5'->3': increasing coordinates for +strand genes, decreasing for -strand
                    elif annot_type == 'UTR':
                        # cases:
                        #   - start of first exon -> 5' UTR
                        #   - start of an exon with preceding exon in 5' UTR -> 5' UTR
                        #   - else append to 3' UTR
                        if g.strand == '+':
                            if (start_pos == t.start_pos or
                                    (len(t.utr5) < len(t.exons) and start_pos == t.exons[len(t.utr5)].start_pos)):
                                t.utr5.append([start_pos, end_pos])
                            else:
                                t.utr3.append([start_pos, end_pos])
                        else:
                            if (end_pos == t.end_pos or
                                    (len(t.utr5) < len(t.exons) and end_pos == t.exons[len(t.utr5)].end_pos)):
                                t.utr5.append([start_pos, end_pos])
                            else:
                                t.utr3.append([start_pos, end_pos])

                    elif annot_type == 'CDS':
                        t.exons[int(attributes['exon_number'])-1].CDS = [start_pos, end_pos]

                    # start/stop codons may be split across exons -> store/append coordinates
                    elif annot_type == 'start_codon':
                        t.start_codon.extend(np.arange(start_pos, end_pos+1))

                    elif annot_type == 'stop_codon':
                        t.stop_codon.extend(np.arange(start_pos, end_pos+1))

                    elif annot_type == 'Selenocysteine':
                        pass

                    if len(self.genes) % 1000 == 0 and verbose:
                        print(f'\rGenes parsed: {len(self.genes)}', end='')
            if verbose:
                print(f'\rGenes parsed: {len(self.genes)}')

        self.gene_ids = np.array(self.gene_ids)
        self.gene_names = np.array(self.gene_names)

        self.genes = np.array(self.genes)
        self.transcripts = np.array(self.transcripts)

        self.transcripts_per_gene = np.array([len(g.transcripts) for g in self.genes])

        # dictionary of gene arrays by chromosome
        chrs = [g.chr for g in self.genes]
        chrs,sidx = np.unique(chrs, return_index=True) # sorted output (lex)
        i = np.argsort(sidx)
        chrs = chrs[i]
        # start/end index of chromosomes in annotation
        sidx = sidx[i]
        eidx = np.hstack((sidx[1:]-1, len(self.genes)-1))

        self.chr_list = chrs
        self.chr_index = dict([(chrs[i], [sidx[i],eidx[i]]) for i in range(len(chrs))])
        self.chr_genes = dict([(chrs[i], self.genes[sidx[i]:eidx[i]+1]) for i in range(len(chrs))])

        # interval trees with gene starts/ends for each chr
        self.gene_interval_trees = defaultdict(IntervalTree)
        for g in self.genes:
            self.gene_interval_trees[g.chr].add(g.start_pos, g.end_pos+1, g)
            # add transcript lenghts
            for t in g.transcripts:
                t.length = sum([e.length for e in t.exons])
            # add annotation
            # g.annotation = self

        self.add_biotype()


    def load_fasta(self, fasta_file):
        self.fasta_dict = load_fasta(fasta_file)


    def query_genes(self, region_str):
        """Find genes overlapping region defined as chr:start-end"""
        chrom, pos = region_str.split(':')
        pos = [int(i) for i in pos.split('-')]
        if len(pos) == 2:
            return self.gene_interval_trees[chrom].find(pos[0], pos[1])
        else:
            return self.gene_interval_trees[chrom].find(pos[0], pos[0])


    def add_biotype(self):
        """
        Add biotype annotation from http://useast.ensembl.org/Help/Glossary?id=275
        """
        biotype = np.array([g.type for g in self.genes])
        pc_type = [
            'IG_C_gene',
            'IG_D_gene',
            'IG_J_gene',
            'IG_LV_gene',
            'IG_M_gene',
            'IG_V_gene',
            'IG_Z_gene',
            'nonsense_mediated_decay',
            'nontranslating_CDS',
            'non_stop_decay',
            'polymorphic_pseudogene',
            'protein_coding',
            'TR_C_gene',
            'TR_D_gene',
            'TR_gene',
            'TR_J_gene',
            'TR_V_gene'
        ]
        biotype[np.isin(biotype, pc_type)] = 'protein_coding'

        pseudo_type = [
            'disrupted_domain',
            'IG_C_pseudogene',
            'IG_J_pseudogene',
            'IG_pseudogene',
            'IG_V_pseudogene',
            'processed_pseudogene',
            'pseudogene',
            'transcribed_processed_pseudogene',
            'transcribed_unprocessed_pseudogene',
            'translated_processed_pseudogene',
            'translated_unprocessed_pseudogene',
            'TR_J_pseudogene',
            'TR_V_pseudogene',
            'unitary_pseudogene',
            'unprocessed_pseudogene',
            'transcribed_unitary_pseudogene',  # not in ensembl
        ]
        biotype[np.isin(biotype, pseudo_type)] = 'pseudogene'

        lnc_type = [
            '3prime_overlapping_ncrna',
            '3prime_overlapping_ncRNA',
            'ambiguous_orf',
            'antisense',
            'lincRNA',
            'ncrna_host',
            'non_coding',
            'processed_transcript',
            'retained_intron',
            'sense_intronic',
            'sense_overlapping',
            'bidirectional_promoter_lncRNA',  # not in ensembl
            'macro_lncRNA',  # not in ensembl
        ]
        biotype[np.isin(biotype, lnc_type)] = 'long_noncoding'

        snc_type = [
            'miRNA',
            'miRNA_pseudogene',
            'misc_RNA',
            'misc_RNA_pseudogene',
            'Mt_rRNA',
            'Mt_tRNA',
            'Mt_tRNA_pseudogene',
            'ncRNA',
            'pre_miRNA',
            'RNase_MRP_RNA',
            'RNase_P_RNA',
            'rRNA',
            'rRNA_pseudogene',
            'scRNA_pseudogene',
            'snlRNA',
            'snoRNA',
            'snoRNA_pseudogene',
            'snRNA',
            'snRNA_pseudogene',
            'SRP_RNA',
            'tmRNA',
            'tRNA',
            'tRNA_pseudogene',
            'ribozyme',  # not in ensembl
            'sRNA',  # not in ensembl
            'scRNA',  # not in ensembl
            'scaRNA',  # not in ensembl
            'vaultRNA',  # not in ensembl
        ]
        biotype[np.isin(biotype, snc_type)] = 'short_noncoding'
        for (i,g) in enumerate(self.genes):
            g.biotype = biotype[i]


    def get_cassette_transcripts(self):
        """
        Return list of transcripts with a cassette exon

        Definition used: exon unique to transcript, flanking exons present in other transcripts
        """
        if not hasattr(self, 'cassette_transcripts') or len(self.cassette_transcripts) == 0:
            self.cassette_transcripts = []
            for g in self.genes:
                # proj = np.bincount(np.concatenate([np.arange(e.start_pos-g.start_pos,e.end_pos-g.start_pos+1) for t in g.transcripts for e in t.exons]))
                proj = np.zeros(g.end_pos-g.start_pos+1)
                for t in g.transcripts:
                    for e in t.exons:
                        proj[e.start_pos-g.start_pos:e.end_pos-g.start_pos+1] += 1

                for t in g.transcripts:
                    if len(t.exons) > 2:
                        cand = np.zeros(len(t.exons), dtype=bool)
                        for (i,e) in enumerate(t.exons[1:-1]):
                            cand[i] = all(proj[e.start_pos-g.start_pos:e.end_pos-g.start_pos+1] == 1)
                        for i in np.arange(1,len(cand)-1):
                            if cand[i] and (not cand[i-1]) and (not cand[i+1]):
                                e.iscassette = True
                                if e.transcript not in self.cassette_transcripts:
                                    self.cassette_transcripts.append(e.transcript)

            print('Number of cassette exons found: '+str(len(self.cassette_transcripts)))

        return self.cassette_transcripts


    def get_junctions(self, min_intron_length=0, exclude_biotypes=[]):
        """Return DataFrame with junction information: chr, intron_start, intron_end"""
        junctions = []
        for g in self.genes:
            for t in g.transcripts:
                if t.type not in exclude_biotypes:
                    for i in range(len(t.exons)-1):
                        if g.strand == '+':
                            junctions.append([g.chr, t.exons[i].end_pos+1, t.exons[i+1].start_pos-1, g.strand, g.id])
                        else:
                            junctions.append([g.chr, t.exons[i+1].end_pos+1, t.exons[i].start_pos-1, g.strand, g.id])
        df = pd.DataFrame(junctions, columns=['chr', 'intron_start', 'intron_end', 'strand', 'gene_id']).drop_duplicates()
        # sort within chrs
        df = df.groupby('chr', sort=False).apply(lambda x: x.sort_values(['intron_start', 'intron_end'])).reset_index(drop=True)
        return df


    def get_junction_ids(self, min_intron_length=0):
        """
        For each junction in the annotation construct identifier string:
        <chromosome>_<first base of intron>_<last base of intron>

        Coordinates are 1-based
        """

        id2gene = defaultdict()
        junction_ids = []
        for c in annot.chr_list:
            idset  = set()  # chr.-specific set to preserve chr. order
            for g in annot.chr_genes[c]:
                for t in g.transcripts:
                    if len(t.exons) > 1:
                        if g.strand == '+':
                            for i in range(len(t.exons)-1):
                                if t.exons[i+1].start_pos-1 - t.exons[i].end_pos >= min_intron_length:
                                    j = g.chr+'_'+str(t.exons[i].end_pos+1)+'_'+str(t.exons[i+1].start_pos-1)
                                    idset.add(j)
                                    id2gene.setdefault(j, set()).add(g.id)
                        else:
                            for i in range(len(t.exons)-1):
                                if t.exons[i].start_pos-1 - t.exons[i+1].end_pos >= min_intron_length:
                                    j = g.chr+'_'+str(t.exons[i+1].end_pos+1)+'_'+str(t.exons[i].start_pos-1)
                                    idset.add(j)
                                    id2gene.setdefault(j, set()).add(g.id)
            # sort by position
            idset = list(idset)
            idset.sort(key=lambda x: [int(i) for i in x.split('_')[1:]])
            junction_ids.extend(idset)

        return junction_ids, id2gene


    def export_junctions(self, dest_file, min_intron_length=0):
        """
        Write junctions to file, as:
          chromosome, intron start, intron end, gene_id(s)
        """
        junction_ids, id2gene = self.get_junction_ids(min_intron_length=min_intron_length)
        with open(dest_file, 'w') as f:
            f.write('chr\tintron_start\tintron_end\tgene_id\n')
            for i in junction_ids:
                f.write(i.replace('_','\t')+'\t'+', '.join(id2gene[i])+'\n')


    def get_gene_index(self, query):
        """Return gene index(es) corresponding to gene_id or gene_name"""
        if len(query) > 4 and query[:4] == 'ENSG':
            return np.nonzero(query == self.gene_ids)[0]
        else:
            return np.nonzero(query == self.gene_names)[0]


    def get_gene(self, query):
        """Return gene(s) corresponding to gene_id or gene_name"""
        if len(query) > 4 and query[:4] == 'ENSG':
            g = self.genes[np.where(query == self.gene_ids)[0]]
        else:
            g = self.genes[np.where(query == self.gene_names)[0]]
        if len(g) == 1:
            g = g[0]
        return g


    def get_genes_by_transcript_type(self, transcript_type):
        """Return subset of genes containing transcripts of a given type, i.e., protein_coding"""
        return [g for g in self.genes if transcript_type in [t.type for t in g.transcripts]]


    def map2transcripts(self, genevalues):
        """Maps gene vector to transcript vector"""
        return np.repeat(genevalues, self.transcripts_per_gene)


    def get_transcript_indexes(self, gene_index):
        """Index(es) in transcript array"""
        return np.sum(self.transcripts_per_gene[:gene_index]) + np.arange(self.transcripts_per_gene[gene_index])


    def get_g2tmap(self, sort=False):
        """Return array mapping gene_ids to transcript_ids"""
        g2tmap = []
        for g in self.genes:
            for t in g.transcripts:
                g2tmap.append([g.id, t.id])

        g2tmap = np.array(g2tmap)

        if sort: # sort by gene, then transcript id
            idx = np.lexsort((g2tmap[:,1], g2tmap[:,0]))
            g2tmap = gt2map[idx,:]

        return g2tmap


    def load_mappability(self, bigwig):
        """
        Add mappability to each gene, transcript and exon.
        Transcript values are averages over exons;
        Gene values are averages over transcripts
        """
        # ex = []
        bw = pyBigWig.open(bigwig)
        for i,g in enumerate(self.genes, 1):
            gm = 0
            for t in g.transcripts:
                tm = 0
                for e in t.exons:
                    if g.chr in bw.chroms():
                        m = bw.stats(g.chr, e.start_pos-1, e.end_pos, exact=True)[0]
                        # m = np.nanmean(bw.values(g.chr, e.start_pos-1, e.end_pos))
                    else:
                        m = np.nan
                    tm += m
                    e.mappability = m
                    # ex.append(m)
                tm /= len(t.exons)
                t.mappability = tm
                gm += tm
            gm /= len(g.transcripts)
            g.mappability = gm
            # ex.append(gm)
            if i % 100 == 0 or i == len(self.genes):
                print(f'\r  * Loading mappability. Genes parsed: {i:5d}/{len(self.genes)}', end='' if i < len(self.genes) else None)
        bw.close()


    def get_tss_bed(self):
        """Get DataFrame with [chr, TSS-1, TSS, gene_id] columns for each gene"""
        bed_df = []
        for g in self.genes:
            bed_df.append([g.chr, g.tss-1, g.tss, g.id])
        bed_df = pd.DataFrame(bed_df, columns=['chr', 'start', 'end', 'gene_id'])
        bed_df.index = bed_df['gene_id']
        # sort by start position
        bed_df = bed_df.groupby('chr', sort=False, group_keys=False).apply(lambda x: x.sort_values('start'))
        return bed_df


    def write_gtf(self, gtf_path):
        """Write to GTF file. Only gene/transcript/exon features are used."""
        with open(gtf_path, 'w') as gtf:
            if self.header:
                gtf.write('\n'.join(self.header)+'\n')
            for g in self.genes:
                gtf.write('\t'.join([g.chr, g.source, 'gene',
                                     str(g.start_pos), str(g.end_pos),
                                     '.', g.strand, '.', g.attributes_string])+'\n')

                for t in g.transcripts:
                    gtf.write('\t'.join([g.chr, t.source, 'transcript',
                                         str(t.start_pos), str(t.end_pos),
                                         '.', g.strand, '.', t.attributes_string])+'\n')
                    for e in t.exons:
                        gtf.write('\t'.join([g.chr, t.source, 'exon',
                                             str(e.start_pos), str(e.end_pos),
                                             '.', g.strand, '.', e.attributes_string])+'\n')


    def write_bed(self, bed_path, overwrite=False):
        """
        Write to BED format.
        Columns: chr, start, end, gff3_attributes (ID, name, biotype), score (1000), strand, thick_start, thick_end, ., #exons, sizes, starts
        """
        if not os.path.exists(bed_path) or overwrite:
            with open(bed_path, 'w') as bed:
                bed.write('#gffTags\n')
                for g in self.genes:
                    for t in g.transcripts:
                        # BED intervals: [...), 0-based
                        start = str(t.start_pos-1)
                        end = str(t.end_pos)
                        exon_lengths = [str(e.length) for e in t.exons]
                        if g.strand == '+':
                            exon_starts = [str(e.start_pos - t.exons[0].start_pos) for e in t.exons]
                        elif g.strand == '-':
                            exon_lengths = exon_lengths[::-1]
                            exon_starts = [str(e.start_pos - t.exons[-1].start_pos) for e in t.exons[::-1]]

                        cds_ranges = [e.CDS for e in t.exons if hasattr(e, 'CDS')]
                        if cds_ranges:
                            if t.stop_codon:
                                cds_ranges.append(t.stop_codon[::2])
                            if g.strand == '+':
                                thick_start = str(cds_ranges[0][0] - 1)
                                thick_end = str(cds_ranges[-1][1])
                            elif g.strand == '-':
                                thick_start = str(cds_ranges[-1][0] - 1)
                                thick_end = str(cds_ranges[0][1])
                        else:
                            thick_start = start
                            thick_end = end

                        attributes = f"ID={t.id};Name={t.name};Type={t.type};Parent={g.id}"

                        s = [g.chr, start, end, attributes, '1000', g.strand, thick_start, thick_end, '.',
                            str(len(t.exons)),
                            ','.join(exon_lengths) + ',',
                            ','.join(exon_starts) + ',',
                        ]
                        bed.write('\t'.join(s)+'\n')
