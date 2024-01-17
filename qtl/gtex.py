import numpy as np
import pandas as pd
import json
import urllib
import ssl
from collections.abc import Iterable
from matplotlib.colors import hsv_to_rgb, to_hex


def s2d(x):
    """Parse donor ID from sample ID"""
    if isinstance(x, str):
        return '-'.join(x.split('-')[:2])
    elif isinstance(x, Iterable):
        return ['-'.join(i.split('-')[:2]) for i in x]


def get_tissue_id(t):
    """Convert tissue name to tissue ID"""
    if isinstance(t, str):
        return t.replace('(','').replace(')','').replace(' - ', ' ').replace(' ', '_')
    elif isinstance(t, Iterable):
        return [i.replace('(','').replace(')','').replace(' - ', ' ').replace(' ', '_') for i in t]


def _get_api_data():
    context = ssl.create_default_context()
    context.check_hostname = False
    context.verify_mode = ssl.CERT_NONE
    tissues_json = json.loads(urllib.request.urlopen('https://gtexportal.org/api/v2/dataset/tissueSiteDetail',
                                                     context=context).read().decode())['data']
    return tissues_json


def get_colors_df(diff_brain=False):
    """Return pd.DataFrame mapping tissue IDs to colors"""
    tissues_json = _get_api_data()
    colors_df = pd.DataFrame(tissues_json).rename(columns={
        'tissueSiteDetailId':'tissue_id',
        'colorHex':'color_hex',
        'colorRgb':'color_rgb',
        'tissueSiteDetail':'tissue_site_detail',
        'tissueSiteDetailAbbr':'tissue_abbrv',
        'tissueSite':'tissue_site',
        'ontologyId':'ontology_id',
    }).set_index('tissue_id')
    colors_df = colors_df[['tissue_site', 'tissue_site_detail', 'tissue_abbrv', 'ontology_id', 'color_rgb', 'color_hex']]
    colors_df['color_hex'] = '#' + colors_df['color_hex']
    if diff_brain:
        rgb_s = pd.Series({
            'Brain_Amygdala':                        hsv_to_rgb([ 0.1,  1., 0.933]),
            'Brain_Anterior_cingulate_cortex_BA24':  hsv_to_rgb([ 0.11, 1., 0.933]),
            'Brain_Caudate_basal_ganglia':           hsv_to_rgb([ 0.12, 1., 0.933]),
            'Brain_Cerebellar_Hemisphere':           hsv_to_rgb([ 0.13, 1., 0.933]),
            'Brain_Cerebellum':                      hsv_to_rgb([ 0.13, 1., 0.933]),
            'Brain_Cortex':                          hsv_to_rgb([ 0.14, 1., 0.933]),
            'Brain_Frontal_Cortex_BA9':              hsv_to_rgb([ 0.14, 1., 0.933]),
            'Brain_Hippocampus':                     hsv_to_rgb([ 0.15, 1., 0.933]),
            'Brain_Hypothalamus':                    hsv_to_rgb([ 0.16, 1., 0.933]),
            'Brain_Nucleus_accumbens_basal_ganglia': hsv_to_rgb([ 0.17, 1., 0.933]),
            'Brain_Putamen_basal_ganglia':           hsv_to_rgb([ 0.18, 1., 0.933]),
            'Brain_Spinal_cord_cervical_c-1':        hsv_to_rgb([ 0.19, 1., 0.933]),
            'Brain_Substantia_nigra':                hsv_to_rgb([ 0.2,  1., 0.933]),
        })
        brain_tissues = [i for i in sorted(colors_df.index) if i.startswith('Brain')]
        colors_df.loc[brain_tissues, 'color_hex'] = rgb_s[brain_tissues].apply(lambda x: to_hex(x).upper())
        colors_df.loc[brain_tissues, 'color_rgb'] = rgb_s[brain_tissues].apply(
            lambda x: ','.join(np.round(x*255).astype(int).astype(str)))

    colors_df.index.name = 'tissue_id'
    colors_df.insert(3, 'tissue_title', colors_df['tissue_site_detail'].map(tissue_title_map))
    return colors_df


# Simplified tissue names for figures
tissue_title_map = {
    'Adipose - Subcutaneous': 'Subcutaneous adipose',
    'Adipose - Visceral (Omentum)': 'Visceral omentum',
    'Adrenal Gland': 'Adrenal gland',
    'Artery - Aorta': 'Aorta',
    'Artery - Coronary': 'Coronary artery',
    'Artery - Tibial': 'Tibial artery',
    'Bladder': 'Bladder',
    'Brain - Amygdala': 'Amygdala',
    'Brain - Anterior cingulate cortex (BA24)': 'Anterior cingulate cortex',
    'Brain - Caudate (basal ganglia)': 'Caudate (basal ganglia)',
    'Brain - Cerebellar Hemisphere': 'Cerebellar hemisphere',
    'Brain - Cerebellum': 'Cerebellum',
    'Brain - Cortex': 'Cortex',
    'Brain - Frontal Cortex (BA9)': 'Frontal cortex (BA9)',
    'Brain - Hippocampus': 'Hippocampus',
    'Brain - Hypothalamus': 'Hypothalamus',
    'Brain - Nucleus accumbens (basal ganglia)': 'Nucleus accumbens (basal ganglia)',
    'Brain - Putamen (basal ganglia)': 'Putamen (basal ganglia)',
    'Brain - Spinal cord (cervical c-1)': 'Spinal cord (cervical c-1)',
    'Brain - Substantia nigra': 'Substantia nigra',
    'Breast - Mammary Tissue': 'Breast mammary tissue',
    'Cells - EBV-transformed lymphocytes': 'EBV-transformed lymphocytes',
    'Cells - Cultured fibroblasts': 'Cultured fibroblasts',
    'Cervix - Ectocervix': 'Ectocervix',
    'Cervix - Endocervix': 'Endocervix',
    'Colon - Sigmoid': 'Sigmoid colon',
    'Colon - Transverse': 'Transverse colon',
    'Esophagus - Gastroesophageal Junction': 'Gastroesophageal junction',
    'Esophagus - Mucosa': 'Esophagus mucosa',
    'Esophagus - Muscularis': 'Esophagus muscularis',
    'Fallopian Tube': 'Fallopian tube',
    'Heart - Atrial Appendage': 'Atrial appendage',
    'Heart - Left Ventricle': 'Left ventricle',
    'Kidney - Cortex': 'Kidney cortex',
    'Kidney - Medulla': 'Kidney medulla',
    'Liver': 'Liver',
    'Lung': 'Lung',
    'Minor Salivary Gland': 'Minor salivary gland',
    'Muscle - Skeletal': 'Skeletal muscle',
    'Nerve - Tibial': 'Tibial nerve',
    'Ovary': 'Ovary',
    'Pancreas': 'Pancreas',
    'Pituitary': 'Pituitary',
    'Prostate': 'Prostate',
    'Skin - Not Sun Exposed (Suprapubic)': 'Not sun-exposed skin (suprapubic)',
    'Skin - Sun Exposed (Lower leg)': 'Sun-exposed skin (lower leg)',
    'Small Intestine - Terminal Ileum': 'Small intestine terminal ileum',
    'Spleen': 'Spleen',
    'Stomach': 'Stomach',
    'Testis': 'Testis',
    'Thyroid': 'Thyroid',
    'Uterus': 'Uterus',
    'Vagina': 'Vagina',
    'Whole Blood': 'Whole blood',
}


entex_tissue_map = {
    "Peyer's patch": 'Small Intestine - Terminal Ileum',
    'adrenal gland': 'Adrenal Gland',
    'ascending aorta': 'Artery - Aorta',  # correct mapping?
    'body of pancreas': 'Pancreas',
    'breast epithelium': 'Breast - Mammary Tissue',
    'coronary artery': 'Artery - Coronary',
    'esophagus muscularis mucosa': 'Esophagus - Muscularis',
    'esophagus squamous epithelium': 'Esophagus - Mucosa',
    'gastrocnemius medialis': 'Muscle - Skeletal',
    'gastroesophageal sphincter': 'Esophagus - Gastroesophageal Junction',
    'heart left ventricle': 'Heart - Left Ventricle',
    'lower leg skin': 'Skin - Sun Exposed (Lower leg)',
    'omental fat pad': 'Adipose - Visceral (Omentum)',
    'ovary': 'Ovary',
    'prostate gland': 'Prostate',
    'right atrium auricular region': 'Heart - Atrial Appendage',
    'right lobe of liver': 'Liver',
    'sigmoid colon': 'Colon - Sigmoid',
    'spleen': 'Spleen',
    'stomach': 'Stomach',
    'subcutaneous adipose tissue': 'Adipose - Subcutaneous',
    'suprapubic skin': 'Skin - Not Sun Exposed (Suprapubic)',
    'testis': 'Testis',
    'thoracic aorta': 'Artery - Aorta',  # correct mapping?
    'thyroid gland': 'Thyroid',
    'tibial artery': 'Artery - Tibial',
    'tibial nerve': 'Nerve - Tibial',
    'transverse colon': 'Colon - Transverse',
    'upper lobe of left lung': 'Lung',
    'uterus': 'Uterus',
    'vagina': 'Vagina'
}
