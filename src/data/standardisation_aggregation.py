# -*- coding: utf-8 -*-
"""
Created on Thu Jan 28 14:03:42 2021

@author: Moritz
"""
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import division

from rdkit import Chem

from molvs.metal import MetalDisconnector
from molvs.fragment import FragmentRemover
from molvs.normalize import Normalizer
from molvs.standardize import canonicalize_tautomer_smiles
from molvs.charge import Uncharger

import pandas as pd
import numpy as np
import argparse
import os
from rdkit import RDLogger

RDLogger.DisableLog('rdApp.warning')

def valid_smiles(smiles):
    # checks if valid rdkit mol can be obtained; rdkit constructs a valid mol from an empty string, mark them as invalid
    # return True or False
    m = Chem.MolFromSmiles(smiles)
    if smiles == '':
        return (False)
    elif m == None:
        return (False)
    elif m != None:
        return (True)


def disconnect_metal(smiles):
    # return smiles with disconnected metaks
    m = Chem.MolFromSmiles(smiles)
    md = MetalDisconnector()
    m = md.disconnect(m)
    return (Chem.MolToSmiles(m))


def is_organic(fragment):
    # check if mol is organic
    for a in fragment.GetAtoms():
        if a.GetAtomicNum() == 6:
            return True
    return False


def fragment_removal_smiles(smiles, leave_last=False):
    # Utility function that returns the result SMILES after FragmentRemover is applied to given a SMILES string
    # if only single organic fragment: don't remove anything
    # elif only one of the fragments organic: keep only organic
    # else: remove specified fragments
    mol = Chem.MolFromSmiles(smiles)
    fragments = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=False)
    fragment_count = len(fragments)
    if fragment_count == 1 and is_organic(fragments[0]) == True:
        return (smiles)

    else:
        organic_frags = []
        for i, fragment in enumerate(fragments):
            if is_organic(fragment) == True:
                organic_frags.append(i)

        organic_count = len(organic_frags)

        if organic_count == 1:
            return (Chem.MolToSmiles(fragments[organic_frags[0]]))

        else:
            mol = FragmentRemover(leave_last=leave_last).remove(mol)
            return (Chem.MolToSmiles(mol, isomericSmiles=True))


def normalize_smiles(smiles):
    """Utility function that runs normalization rules on a given a SMILES string."""
    mol = Chem.MolFromSmiles(smiles, sanitize=False)
    mol = Normalizer().normalize(mol)
    if mol:
        return Chem.MolToSmiles(mol, isomericSmiles=True)


def uncharge_smiles(smiles):
    """Utility function that returns the uncharged SMILES for a given SMILES string."""
    mol = Chem.MolFromSmiles(smiles)
    u = Uncharger()
    mol = u.uncharge(mol)
    if mol:
        return (Chem.MolToSmiles(mol, isomericSmiles=True))


def neutralizeRadicals(smiles):
    # neutralise mols that are charged because being radicals
    mol = Chem.MolFromSmiles(smiles)
    for a in mol.GetAtoms():
        if a.GetNumRadicalElectrons() >= 1 and a.GetFormalCharge() >= 1:
            a.SetNumRadicalElectrons(0)
            a.SetFormalCharge(0)
    return (Chem.MolToSmiles(mol))


def standardise(smiles):
    # remove Hs
    smiles = Chem.MolToSmiles(
        Chem.rdmolops.RemoveHs(Chem.MolFromSmiles(smiles)))
    # disconnect metals
    smiles = disconnect_metal(smiles)
    # remove fragments
    smiles = fragment_removal_smiles(smiles)
    # uncharge
    smiles = uncharge_smiles(smiles)
    # uncharge radicals
    smiles = neutralizeRadicals(smiles)
    # normalise chemotypes
    smiles = normalize_smiles(smiles)
    # canonicalise tautomers
    smiles = canonicalize_tautomer_smiles(smiles)
    # transform to rdkit mol and back to get rdkit canonical version
    smiles = Chem.MolToSmiles(Chem.MolFromSmiles(
        smiles, sanitize=True), canonical=True)
    # transform to inchi key and back to canonicalise tautomers not considered by molvs
    final_smiles = Chem.MolToSmiles(Chem.MolFromInchi(Chem.MolToInchi(
        Chem.MolFromSmiles(smiles, sanitize=True))), canonical=True)
    # if final_smiles different than smiles return smiles as well, return as a list
    if final_smiles == smiles:
        return ([final_smiles])
    else:
        return ([final_smiles, smiles])


def smiles_to_frags(smiles):
    # convert smiles to list of frags
    frags = Chem.GetMolFrags(Chem.MolFromSmiles(smiles), asMols=True)
    fraglist = [Chem.MolToSmiles(frag) for frag in frags]
    return (fraglist)


def drop_fragments(smiles):
    # handling of multiple fragments:
    # remove inorganic fragments
    # if identical/stereoisomers: keep only 1
    # if different organic components: drop

    # handling of single fragments
    # if not organic: return ''
    # else: return smiles

    if len(smiles_to_frags(smiles)) > 1:
        smileslist = smiles_to_frags(smiles)
        # remove all inorganic fragments, if list becomes empty: return ''
        organics = []
        for frag in smileslist:
            if is_organic(Chem.MolFromSmiles(frag)) == True:
                organics.append(frag)
        if len(organics) == 0:
            return ('')
        elif len(organics) == 1:
            return (organics[0])
        else:
            smileslist = organics
            # check if all frags are identical, if all identical, return first
            identical = True
            for fra in smileslist[1:]:
                if fra != smileslist[0]:
                    identical = False
            if identical == True:
                return (smileslist[0])
            else:
                return ('')

    else:
        if is_organic(Chem.MolFromSmiles(smiles)) == True:
            return (smiles)
        else:
            return ('')


# function that performs aggregation of several rows in the a df grouped by identical standardised SMILES
def majority_agg(df_grouped):
    # store aggregated rows in list
    df_records = []
    for smiles in df_grouped.groups.keys():
        df = df_grouped.get_group(smiles)
        if df.shape[0] == 1:
            df_records.append(df)
        else:
            d = {}
            for col in df.columns[:-1]:
                act = 0
                inact = 0
                for i in df[col]:
                    if i == 1:
                        act += 1
                    elif i == 0:
                        inact += 1

                if act > inact:
                    d[col] = 1
                elif act < inact:
                    d[col] = 0
                else:
                    d[col] = np.nan

            # df that contains agg values for single smile
            df_smi_agg = pd.DataFrame(d, index=[0])
            df_smi_agg['standardised_smiles'] = smiles
            # collect single-line dfs in list
            df_records.append(df_smi_agg)
    df_return = pd.concat(df_records)
    return (df_return)


def parse_arguments():
    parser = argparse.ArgumentParser(description='Process dataset')
    parser.add_argument('--data_name', type=str,
                        default='tox21', help='Name of the dataset')
    parser.add_argument('--input_path', type=str,
                        default='/home/yehlab/projs/project_ya/Project_mol_embedding/FeatureNet_mol_embedding/src/data/Original_dataset', help='Path to the input CSV file')
    parser.add_argument('--output_path', type=str,
                        default='/home/yehlab/projs/project_ya/Project_mol_embedding/FeatureNet_mol_embedding/src/data/Processed_dataset', help='Path to the output CSV file')
    return parser.parse_args()


def main():
    args = parse_arguments()
    df = pd.read_csv(os.path.join(args.input_path, f'{args.data_name}.csv'))

    # drop rows for which no mol can be generated in rdkit
    df = df.astype({'smiles': 'str'})
    valid_smis = []

    for smi in df['smiles']:
        valid_smis.append(valid_smiles(smi))

    df['valid_smiles'] = valid_smis
    df_valid = df[df['valid_smiles'] == True].copy()
    df_valid.drop('valid_smiles', axis=1, inplace=True)

    standardised_smiles = []
    failed = []
    inchi_tautomers = []
    for i, smi in enumerate(df_valid['smiles']):
        print(i)
        try:
            standardised = standardise(smi)
            standardised_smiles.append(standardised[0])
            if len(standardised) == 2:
                inchi_tautomers.append(standardised)
        except:
            standardised_smiles.append('')
            failed.append((i, smi))

    filtered_smiles = []
    for smi in standardised_smiles:
        filtered_smiles.append(drop_fragments(smi))

    df_valid['standardised_smiles'] = filtered_smiles
    df_filtered = df_valid[df_valid['standardised_smiles'] != ''].copy()

    # aggregate identical SMILES in df_filtered

    # keep track which original_smiles names map to standardises smiles
    dict_smi_index = {}
    for smi, idx in zip(df_filtered['standardised_smiles'], df_filtered.index):
        if smi not in dict_smi_index:
            dict_smi_index[smi] = [idx]
        else:
            dict_smi_index[smi].append(idx)

    # drop columns
    df_filtered.drop(labels=['smiles'], axis=1, inplace=True)

    # aggregate results consider as active if at least one of the instances is active
    df_grouped = df_filtered.groupby('standardised_smiles')

    df_agg = majority_agg(df_grouped=df_grouped)
    df_agg.reset_index(inplace=True, drop=True)

    # add column containing identifier
    df_agg['Identifier'] = [dict_smi_index[smi]
                            for smi in df_agg['standardised_smiles']]

    df_agg.to_csv(os.path.join(args.output_path,
                  f'{args.data_name}_aggregated.csv'), index=False)


if __name__ == '__main__':
    main()
