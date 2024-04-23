# Copyright 2023 Applied BioComputation Group, Stony Brook University
# Copyright 2021 AlQuraishi Laboratory
# Copyright 2021 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import collections
import dataclasses
import io
import os
import re
import shutil
import string
import subprocess
import sys
from typing import Dict, List, MutableMapping, Optional, Sequence, Tuple

import numpy as np
from Bio.PDB import PDBParser

from src import constants, kalign, msa_pairing

MAX_TEMPLATES = 1
MSA_CROP_SIZE = 2048

FeatureDict = MutableMapping[str, np.ndarray]
DeletionMatrix = Sequence[Sequence[int]]

_UNIPROT_PATTERN = re.compile(
    r"""
    ^
    # UniProtKB/TrEMBL or UniProtKB/Swiss-Prot
    (?:tr|sp)
    \|
    # A primary accession number of the UniProtKB entry.
    (?P<AccessionIdentifier>[A-Za-z0-9]{6,10})
    # Occasionally there is a _0 or _1 isoform suffix, which we ignore.
    (?:_\d)?
    \|
    # TREMBL repeats the accession ID here. Swiss-Prot has a mnemonic
    # protein ID code.
    (?:[A-Za-z0-9]+)
    _
    # A mnemonic species identification code.
    (?P<SpeciesIdentifier>([A-Za-z0-9]){1,5})
    # Small BFD uses a final value after an underscore, which we ignore.
    (?:_\d+)?
    $
    """,
    re.VERBOSE,
)

REQUIRED_FEATURES = frozenset(
    {
        "aatype",
        "all_atom_mask",
        "all_atom_positions",
        "all_chains_entity_ids",
        "all_crops_all_chains_mask",
        "all_crops_all_chains_positions",
        "all_crops_all_chains_residue_ids",
        "assembly_num_chains",
        "asym_id",
        "bert_mask",
        "cluster_bias_mask",
        "deletion_matrix",
        "deletion_mean",
        "entity_id",
        "entity_mask",
        "mem_peak",
        "msa",
        "msa_mask",
        "num_alignments",
        "num_templates",
        "queue_size",
        "residue_index",
        "resolution",
        "seq_length",
        "seq_mask",
        "sym_id",
        "template_aatype",
        "template_all_atom_mask",
        "template_all_atom_positions",
        "renum_mask",
    }
)

TEMPLATE_FEATURES = {
    "template_aatype": np.float32,
    "template_all_atom_masks": np.float32,
    "template_all_atom_positions": np.float32,
    "template_sequence": np.object_,
    "template_sum_probs": np.float32,
}


@dataclasses.dataclass(frozen=True)
class Protein:
    """Protein structure representation."""

    # Cartesian coordinates of atoms in angstroms. The atom types correspond to
    # residue_constants.atom_types, i.e. the first three are N, CA, CB.
    atom_positions: np.ndarray  # [num_res, num_atom_type, 3]

    # Amino-acid type for each residue represented as an integer between 0 and
    # 20, where 20 is 'X'.
    aatype: np.ndarray  # [num_res]

    # Binary float mask to indicate presence of a particular atom. 1.0 if an atom
    # is present and 0.0 if not. This should be used for loss masking.
    atom_mask: np.ndarray  # [num_res, num_atom_type]

    # Residue index as used in PDB. It is not necessarily continuous or 0-indexed.
    # residue_index: np.ndarray  # [num_res]

    # B-factors, or temperature factors, of each residue (in sq. angstroms units),
    # representing the displacement of the residue from its ground truth mean
    # value.
    b_factors: np.ndarray  # [num_res, num_atom_type]

    res_name: np.ndarray


@dataclasses.dataclass(frozen=True)
class Msa:
    """Class representing a parsed MSA file."""

    sequences: Sequence[str]
    deletion_matrix: DeletionMatrix
    descriptions: Sequence[str]

    def __post_init__(self):
        if not (
            len(self.sequences) == len(self.deletion_matrix) == len(self.descriptions)
        ):
            raise ValueError(
                "All fields for an MSA must have the same length. "
                f"Got {len(self.sequences)} sequences, "
                f"{len(self.deletion_matrix)} rows in the deletion matrix and "
                f"{len(self.descriptions)} descriptions."
            )

    def __len__(self):
        return len(self.sequences)

    def truncate(self, max_seqs: int):
        return Msa(
            sequences=self.sequences[:max_seqs],
            deletion_matrix=self.deletion_matrix[:max_seqs],
            descriptions=self.descriptions[:max_seqs],
        )


@dataclasses.dataclass(frozen=True)
class Identifiers:
    species_id: str = ""


"""
Preprocess the single sequence to get corresponding chain info.

Input:
    @chain_id: str; Default: 'A'.
    @in_a3m_string: str; Content of a3m file.
    @is_homomer_or_monomer: bool; 
    @is_protein: bool; Default: False.
    @pdb_template_object: Object; Default: None,
    @input_sequence: str; Default: None,
    @description: str; Default: None,
Output:
    Dictonary: chain features.
"""


def process_single_chain(
    chain_id,
    in_a3m_string,
    is_homomer_or_monomer,
    is_protein=False,
    pdb_template_object=None,
    input_sequence=None,
):
    chain_feat = make_sequence_features(
        sequence=input_sequence,
        description=chain_id,
        num_res=len(input_sequence),
    )

    msa = parse_a3m(in_a3m_string)
    msa_feat = make_msa_features((msa,))
    chain_feat.update(msa_feat)

    if pdb_template_object is not None:
        temp_feat = make_single_pdb_temp(
            chain_feat["sequence"][0].decode("utf-8"),
            pdb_template_object,
            is_protein,
        )
        chain_feat.update(temp_feat)

    if not is_homomer_or_monomer:
        all_seq_features = make_msa_features([msa])
        valid_feats = (
            "msa",
            "msa_mask",
            "deletion_matrix",
            "deletion_matrix_int",
            "msa_uniprot_accession_identifiers",
            "msa_species_identifiers",
        )
        feats = {
            f"{k}_all_seq": v for k, v in all_seq_features.items() if k in valid_feats
        }

        chain_feat.update(feats)
    return chain_feat


"""
Preprocess the given protein and peptide sequence and protein a3m file with template for inference.

Input:
    @protein_seq: str; Note that it should be a cropped sequence.
    @peptide_seq: str; 
    @protein_a3m_path: str; path of protein a3m file.
Output:
    Dictonary: features for inference.
"""


def preprocess_for_inference(protein_seq, peptide_seq, protein_a3m_path):
    single_dataset = {
        "protein_seq": protein_seq,
        "peptide_seq": peptide_seq,
    }
    assert (
        8 <= len(peptide_seq) <= 11
    ), "Sorry, we are currently only able to predict complexes with peptides ranging in length from 8 to 11."
    single_dataset = {**single_dataset, **constants.template_dict[len(peptide_seq)]}
    protein_chain, peptide_chain = constants.temp_chains
    is_homomer = False
    all_chain_features = {}

    if "template" in single_dataset:
        with open(single_dataset["template"], "r") as fp:
            pdb_template_string = fp.read()
    else:
        pdb_template_string = None

    print("Reading a3m file...")
    # preprocess protein chain
    if not os.path.exists(protein_a3m_path):
        print(f"No a3m file found for protein at {protein_a3m_path}")
        return None
    with open(protein_a3m_path, "r") as f:
        a3m_string_protein = f.read()

    print("Processing protein chain...")
    protein_template_object = from_pdb_string(
        pdb_template_string, single_dataset["protein_tmp_chain"]
    )
    chain_features = process_single_chain(
        protein_chain,
        a3m_string_protein,
        is_homomer,
        is_protein=True,
        pdb_template_object=protein_template_object,
        input_sequence=protein_seq,
    )
    chain_features = convert_monomer_features(chain_features, chain_id=protein_chain)
    all_chain_features[protein_chain] = chain_features

    # preprocess peptide chain
    a3m_string_peptide = f">{peptide_chain}\n{peptide_seq}\n"

    print("Processing peptide chain...")
    peptide_template_object = from_pdb_string(
        pdb_template_string, single_dataset["pep_tmp_chain"]
    )
    chain_features = process_single_chain(
        peptide_chain,
        a3m_string_peptide,
        is_homomer,
        is_protein=False,
        pdb_template_object=peptide_template_object,
        input_sequence=peptide_seq,
    )
    chain_features = convert_monomer_features(chain_features, chain_id=peptide_chain)
    all_chain_features[peptide_chain] = chain_features

    print("Mering features...")
    all_chain_features = add_assembly_features(all_chain_features)
    np_example = pair_and_merge(all_chain_features, is_homomer)
    np_example = pad_msa(np_example, 512)

    # specifically for this project
    np_example["loss_mask"] = (np_example["entity_id"] == 2) * 1.0

    return np_example


"""
Preprocess
"""


def process_unmerged_features(all_chain_features):
    """Postprocessing stage for per-chain features before merging."""
    num_chains = len(all_chain_features)
    for chain_features in all_chain_features.values():
        # Convert deletion matrices to float.
        chain_features["deletion_matrix"] = np.asarray(
            chain_features.pop("deletion_matrix_int"), dtype=np.float32
        )
        if "deletion_matrix_int_all_seq" in chain_features:
            chain_features["deletion_matrix_all_seq"] = np.asarray(
                chain_features.pop("deletion_matrix_int_all_seq"), dtype=np.float32
            )

        chain_features["deletion_mean"] = np.mean(
            chain_features["deletion_matrix"], axis=0
        )

        # Add assembly_num_chains.
        chain_features["assembly_num_chains"] = np.asarray(num_chains)

    # Add entity_mask.
    for chain_features in all_chain_features.values():
        chain_features["entity_mask"] = (chain_features["entity_id"] != 0).astype(
            np.int32
        )


def pair_and_merge(all_chain_features, is_homomer):
    """Runs processing on features to augment, pair and merge.

    Args:
      all_chain_features: A MutableMap of dictionaries of features for each chain.

    Returns:
      A dictionary of features.
    """

    process_unmerged_features(all_chain_features)

    np_chains_list = list(all_chain_features.values())

    pair_msa_sequences = not is_homomer

    if pair_msa_sequences:
        np_chains_list = msa_pairing.create_paired_features(chains=np_chains_list)
        np_chains_list = msa_pairing.deduplicate_unpaired_sequences(np_chains_list)
    np_chains_list = crop_chains(
        np_chains_list,
        msa_crop_size=MSA_CROP_SIZE,
        pair_msa_sequences=pair_msa_sequences,
        max_templates=MAX_TEMPLATES,
    )
    np_example = msa_pairing.merge_chain_features(
        np_chains_list=np_chains_list,
        pair_msa_sequences=pair_msa_sequences,
        max_templates=MAX_TEMPLATES,
    )
    np_example = process_final(np_example)

    return np_example


"""
Feature Processing
"""


def crop_chains(
    chains_list: List[FeatureDict],
    msa_crop_size: int,
    pair_msa_sequences: bool,
    max_templates: int,
) -> List[FeatureDict]:
    """Crops the MSAs for a set of chains.

    Args:
      chains_list: A list of chains to be cropped.
      msa_crop_size: The total number of sequences to crop from the MSA.
      pair_msa_sequences: Whether we are operating in sequence-pairing mode.
      max_templates: The maximum templates to use per chain.

    Returns:
      The chains cropped.
    """

    # Apply the cropping.
    cropped_chains = []
    for chain in chains_list:
        cropped_chain = _crop_single_chain(
            chain,
            msa_crop_size=msa_crop_size,
            pair_msa_sequences=pair_msa_sequences,
            max_templates=max_templates,
        )
        cropped_chains.append(cropped_chain)

    return cropped_chains


def process_final(np_example: FeatureDict) -> FeatureDict:
    """Final processing steps in data pipeline, after merging and pairing."""
    np_example = _correct_msa_restypes(np_example)
    np_example = _make_seq_mask(np_example)
    np_example = _make_msa_mask(np_example)
    np_example = _filter_features(np_example)
    return np_example


def _correct_msa_restypes(np_example):
    """Correct MSA restype to have the same order as residue_constants."""
    new_order_list = constants.MAP_HHBLITS_AATYPE_TO_OUR_AATYPE
    np_example["msa"] = np.take(new_order_list, np_example["msa"], axis=0)
    np_example["msa"] = np_example["msa"].astype(np.int32)
    return np_example


def _crop_single_chain(
    chain: FeatureDict, msa_crop_size: int, pair_msa_sequences: bool, max_templates: int
) -> FeatureDict:
    """Crops msa sequences to `msa_crop_size`."""
    msa_size = chain["num_alignments"]

    if pair_msa_sequences:
        msa_size_all_seq = chain["num_alignments_all_seq"]
        msa_crop_size_all_seq = np.minimum(msa_size_all_seq, msa_crop_size // 2)

        # We reduce the number of un-paired sequences, by the number of times a
        # sequence from this chain's MSA is included in the paired MSA.  This keeps
        # the MSA size for each chain roughly constant.
        msa_all_seq = chain["msa_all_seq"][:msa_crop_size_all_seq, :]
        num_non_gapped_pairs = np.sum(
            np.any(msa_all_seq != msa_pairing.MSA_GAP_IDX, axis=1)
        )
        num_non_gapped_pairs = np.minimum(num_non_gapped_pairs, msa_crop_size_all_seq)

        # Restrict the unpaired crop size so that paired+unpaired sequences do not
        # exceed msa_seqs_per_chain for each chain.
        max_msa_crop_size = np.maximum(msa_crop_size - num_non_gapped_pairs, 0)
        msa_crop_size = np.minimum(msa_size, max_msa_crop_size)
    else:
        msa_crop_size = np.minimum(msa_size, msa_crop_size)

    include_templates = "template_aatype" in chain and max_templates
    if include_templates:
        num_templates = chain["template_aatype"].shape[0]
        templates_crop_size = np.minimum(num_templates, max_templates)

    for k in chain:
        k_split = k.split("_all_seq")[0]
        if k_split in msa_pairing.TEMPLATE_FEATURES:
            chain[k] = chain[k][:templates_crop_size, :]
        elif k_split in msa_pairing.MSA_FEATURES:
            if "_all_seq" in k and pair_msa_sequences:
                chain[k] = chain[k][:msa_crop_size_all_seq, :]
            else:
                chain[k] = chain[k][:msa_crop_size, :]

    chain["num_alignments"] = np.asarray(msa_crop_size, dtype=np.int32)
    if include_templates:
        chain["num_templates"] = np.asarray(templates_crop_size, dtype=np.int32)
    if pair_msa_sequences:
        chain["num_alignments_all_seq"] = np.asarray(
            msa_crop_size_all_seq, dtype=np.int32
        )
    return chain


def _filter_features(np_example: FeatureDict) -> FeatureDict:
    """Filters features of example to only those requested."""
    return {k: v for (k, v) in np_example.items() if k in REQUIRED_FEATURES}


def _make_msa_mask(np_example):
    """Mask features are all ones, but will later be zero-padded."""

    np_example["msa_mask"] = np.ones_like(np_example["msa"], dtype=np.float32)

    seq_mask = (np_example["entity_id"] > 0).astype(np.float32)
    np_example["msa_mask"] *= seq_mask[None]

    return np_example


def _make_seq_mask(np_example):
    np_example["seq_mask"] = (np_example["entity_id"] > 0).astype(np.float32)
    return np_example


"""
Pipeline Multimer
"""


def add_assembly_features(
    all_chain_features: MutableMapping[str, FeatureDict],
) -> MutableMapping[str, FeatureDict]:
    """Add features to distinguish between chains.

    Args:
      all_chain_features: A dictionary which maps chain_id to a dictionary of
        features for each chain.

    Returns:
      all_chain_features: A dictionary which maps strings of the form
        `<seq_id>_<sym_id>` to the corresponding chain features. E.g. two
        chains from a homodimer would have keys A_1 and A_2. Two chains from a
        heterodimer would have keys A_1 and B_1.
    """
    # Group the chains by sequence
    seq_to_entity_id = {}
    grouped_chains = collections.defaultdict(list)
    for chain_id, chain_features in all_chain_features.items():
        seq = str(chain_features["sequence"])
        if seq not in seq_to_entity_id:
            seq_to_entity_id[seq] = len(seq_to_entity_id) + 1
        grouped_chains[seq_to_entity_id[seq]].append(chain_features)

    new_all_chain_features = {}
    chain_id = 1
    for entity_id, group_chain_features in grouped_chains.items():
        for sym_id, chain_features in enumerate(group_chain_features, start=1):
            new_all_chain_features[
                f"{int_id_to_str_id(entity_id)}_{sym_id}"
            ] = chain_features
            seq_length = chain_features["seq_length"]
            chain_features["asym_id"] = chain_id * np.ones(seq_length)
            chain_features["sym_id"] = sym_id * np.ones(seq_length)
            chain_features["entity_id"] = entity_id * np.ones(seq_length)
            chain_id += 1

    return new_all_chain_features


def convert_monomer_features(
    monomer_features: FeatureDict, chain_id: str
) -> FeatureDict:
    """Reshapes and modifies monomer features for multimer models."""
    converted = {}
    converted["auth_chain_id"] = np.asarray(chain_id, dtype=np.object_)
    unnecessary_leading_dim_feats = {
        "sequence",
        "domain_name",
        "num_alignments",
        "seq_length",
    }
    for feature_name, feature in monomer_features.items():
        if feature_name in unnecessary_leading_dim_feats:
            # asarray ensures it's a np.ndarray.
            feature = np.asarray(feature[0], dtype=feature.dtype)
        elif feature_name == "aatype":
            # The multimer model performs the one-hot operation itself.
            feature = np.argmax(feature, axis=-1).astype(np.int32)
        elif feature_name == "template_aatype":
            feature = np.argmax(feature, axis=-1).astype(np.int32)
            new_order_list = constants.MAP_HHBLITS_AATYPE_TO_OUR_AATYPE
            feature = np.take(new_order_list, feature.astype(np.int32), axis=0)
        elif feature_name == "template_all_atom_masks":
            feature_name = "template_all_atom_mask"
        converted[feature_name] = feature
    return converted


def get_identifiers(description: str) -> Identifiers:
    """Computes extra MSA features from the description."""
    sequence_identifier = _extract_sequence_identifier(description)
    if sequence_identifier is None:
        return Identifiers()
    else:
        return _parse_sequence_identifier(sequence_identifier)


def int_id_to_str_id(num: int) -> str:
    """Encodes a number as a string, using reverse spreadsheet style naming.

    Args:
      num: A positive integer.

    Returns:
      A string that encodes the positive integer using reverse spreadsheet style,
      naming e.g. 1 = A, 2 = B, ..., 27 = AA, 28 = BA, 29 = CA, ... This is the
      usual way to encode chain IDs in mmCIF files.
    """
    if num <= 0:
        raise ValueError(f"Only positive integers allowed, got {num}.")

    num = num - 1  # 1-based indexing.
    output = []
    while num >= 0:
        output.append(chr(num % 26 + ord("A")))
        num = num // 26 - 1
    return "".join(output)


def make_msa_features(msas: Sequence[Msa]) -> FeatureDict:
    """Constructs a feature dict of MSA features."""
    if not msas:
        raise ValueError("At least one MSA must be provided.")

    int_msa = []
    deletion_matrix = []
    species_ids = []
    seen_sequences = set()
    for msa_index, msa in enumerate(msas):
        if not msa:
            raise ValueError(f"MSA {msa_index} must contain at least one sequence.")
        for sequence_index, sequence in enumerate(msa.sequences):
            if sequence in seen_sequences:
                continue
            seen_sequences.add(sequence)
            int_msa.append([constants.HHBLITS_AA_TO_ID[res] for res in sequence])
            deletion_matrix.append(msa.deletion_matrix[sequence_index])
            identifiers = get_identifiers(msa.descriptions[sequence_index])
            species_ids.append(identifiers.species_id.encode("utf-8"))

    num_res = len(msas[0].sequences[0])
    num_alignments = len(int_msa)
    features = {}
    features["deletion_matrix_int"] = np.array(deletion_matrix, dtype=np.int32)
    features["msa"] = np.array(int_msa, dtype=np.int32)
    features["num_alignments"] = np.array([num_alignments] * num_res, dtype=np.int32)
    features["msa_species_identifiers"] = np.array(species_ids, dtype=np.object_)
    return features


def make_sequence_features(
    sequence: str, description: str, num_res: int
) -> FeatureDict:
    """Constructs a feature dict of sequence features."""
    features = {}
    features["aatype"] = constants.sequence_to_onehot(
        sequence=sequence, mapping=constants.restype_order_with_x, map_unknown_to_x=True
    )
    features["between_segment_residues"] = np.zeros((num_res,), dtype=np.int32)
    features["domain_name"] = np.array([description.encode("utf-8")], dtype=np.object_)
    features["residue_index"] = np.array(range(num_res), dtype=np.int32)
    features["seq_length"] = np.array([num_res] * num_res, dtype=np.int32)
    print(sequence)
    features["sequence"] = np.array([sequence.encode("utf-8")], dtype=np.object_)
    return features


def make_single_pdb_temp(
    query_seq,
    pdb_template_object,
    is_protein,
    kalign_binary_path=sys.prefix + '/bin/kalign',
):
    if not os.path.isfile(kalign_binary_path):
        kalign_binary_path=shutil.which('kalign')
    temp_seq = "".join(pdb_template_object.res_name)
    if is_protein:
        aligner = kalign.Kalign(binary_path=kalign_binary_path)
        out_align = parse_a3m(aligner.align([temp_seq, query_seq]))
        temp_align, query_align = out_align.sequences
        tmp_to_query_mapping = {}
        temp_index = -1
        query_index = -1
        for temp_aa, query_aa in zip(temp_align, query_align):
            if temp_aa != "-":
                temp_index += 1
            if query_aa != "-":
                query_index += 1
            if temp_aa != "-" and query_aa != "-":
                tmp_to_query_mapping[temp_index] = query_index
    # We require peptide from the sample has length equal to the template one
    else:
        assert len(query_seq) == len(temp_seq)
        tmp_to_query_mapping = {}
        for i in range(len(query_seq)):
            tmp_to_query_mapping[i] = i
    output_templates_sequence = []
    templates_all_atom_positions = []
    templates_all_atom_masks = []
    for _ in query_seq:
        templates_all_atom_positions.append(np.zeros((constants.atom_type_num, 3)))
        templates_all_atom_masks.append(np.zeros(constants.atom_type_num))
        output_templates_sequence.append("-")
    for i, j in tmp_to_query_mapping.items():
        if is_protein:
            templates_all_atom_positions[j] = pdb_template_object.atom_positions[i]
            templates_all_atom_masks[j] = pdb_template_object.atom_mask[i]
            output_templates_sequence[j] = pdb_template_object.res_name[i]
        else:
            templates_all_atom_positions[j][:5] = pdb_template_object.atom_positions[i][
                :5
            ]
            templates_all_atom_masks[j][:5] = pdb_template_object.atom_mask[i][:5]
            output_templates_sequence[j] = query_seq[j]

    output_templates_sequence = "".join(output_templates_sequence)
    templates_aatype = constants.sequence_to_onehot(
        output_templates_sequence, constants.HHBLITS_AA_TO_ID
    )
    temp_feat = {
        "template_all_atom_positions": np.array(templates_all_atom_positions),
        "template_all_atom_masks": np.array(templates_all_atom_masks),
        "template_sequence": output_templates_sequence.encode(),
        "template_aatype": np.array(templates_aatype),
        "template_sum_probs": 1.0,
    }
    template_features = {}
    for template_feature_name in TEMPLATE_FEATURES:
        template_features[template_feature_name] = []
    for k in template_features:
        template_features[k].append(temp_feat[k])
    for k in template_features:
        template_features[k] = np.stack(template_features[k], axis=0).astype(
            TEMPLATE_FEATURES[k]
        )
    return template_features


def pad_msa(np_example, min_num_seq):
    np_example = dict(np_example)
    num_seq = np_example["msa"].shape[0]
    if num_seq < min_num_seq:
        for feat in ("msa", "deletion_matrix", "bert_mask", "msa_mask"):
            np_example[feat] = np.pad(
                np_example[feat], ((0, min_num_seq - num_seq), (0, 0))
            )
        np_example["cluster_bias_mask"] = np.pad(
            np_example["cluster_bias_mask"], ((0, min_num_seq - num_seq),)
        )
    return np_example


def parse_a3m(a3m_string: str) -> Msa:
    """Parses sequences and deletion matrix from a3m format alignment.

    Args:
      a3m_string: The string contents of a a3m file. The first sequence in the
        file should be the query sequence.

    Returns:
      A tuple of:
        * A list of sequences that have been aligned to the query. These
          might contain duplicates.
        * The deletion matrix for the alignment as a list of lists. The element
          at `deletion_matrix[i][j]` is the number of residues deleted from
          the aligned sequence i at residue position j.
        * A list of descriptions, one per sequence, from the a3m file.
    """
    sequences, descriptions = parse_fasta(a3m_string)
    deletion_matrix = []
    for msa_sequence in sequences:
        deletion_vec = []
        deletion_count = 0
        for j in msa_sequence:
            if j.islower():
                deletion_count += 1
            else:
                deletion_vec.append(deletion_count)
                deletion_count = 0
        deletion_matrix.append(deletion_vec)

    # Make the MSA matrix out of aligned (deletion-free) sequences.
    deletion_table = str.maketrans("", "", string.ascii_lowercase)
    aligned_sequences = [s.translate(deletion_table) for s in sequences]
    return Msa(
        sequences=aligned_sequences,
        deletion_matrix=deletion_matrix,
        descriptions=descriptions,
    )


def parse_fasta(fasta_string: str) -> Tuple[Sequence[str], Sequence[str]]:
    """Parses FASTA string and returns list of strings with amino-acid sequences.

    Arguments:
      fasta_string: The string contents of a FASTA file.

    Returns:
      A tuple of two lists:
      * A list of sequences.
      * A list of sequence descriptions taken from the comment lines. In the
        same order as the sequences.
    """
    sequences = []
    descriptions = []
    index = -1
    for line in fasta_string.splitlines():
        line = line.strip()
        if line.startswith(">"):
            index += 1
            descriptions.append(line[1:])  # Remove the '>' at the beginning.
            sequences.append("")
            continue
        elif not line:
            continue  # Skip blank lines.
        sequences[index] += line

    return sequences, descriptions


def _extract_sequence_identifier(description: str) -> Optional[str]:
    """Extracts sequence identifier from description. Returns None if no match."""
    split_description = description.split()
    if split_description:
        return split_description[0].partition("/")[0]
    else:
        return None


def _parse_sequence_identifier(msa_sequence_identifier: str) -> Identifiers:
    """Gets species from an msa sequence identifier.

    The sequence identifier has the format specified by
    _UNIPROT_TREMBL_ENTRY_NAME_PATTERN or _UNIPROT_SWISSPROT_ENTRY_NAME_PATTERN.
    An example of a sequence identifier: `tr|A0A146SKV9|A0A146SKV9_FUNHE`

    Args:
      msa_sequence_identifier: a sequence identifier.

    Returns:
      An `Identifiers` instance with species_id. These
      can be empty in the case where no identifier was found.
    """
    matches = re.search(_UNIPROT_PATTERN, msa_sequence_identifier.strip())
    if matches:
        return Identifiers(species_id=matches.group("SpeciesIdentifier"))
    return Identifiers()


"""
PDB to Template
"""


def from_pdb_string(pdb_str: str, chain_id: Optional[str] = None) -> Protein:
    """Takes a PDB string and constructs a Protein object.

    WARNING: All non-standard residue types will be converted into UNK. All
      non-standard atoms will be ignored.

    Args:
      pdb_str: The contents of the pdb file
      chain_id: If None, then the pdb file must contain a single chain (which
        will be parsed). If chain_id is specified (e.g. A), then only that chain
        is parsed.

    Returns:
      A new `Protein` parsed from the pdb contents.
    """
    if pdb_str is None:
        return None
    pdb_fh = io.StringIO(pdb_str)
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("none", pdb_fh)
    models = list(structure.get_models())
    if len(models) != 1:
        raise ValueError(
            f"Only single model PDBs are supported. Found {len(models)} models."
        )
    model = models[0]

    if chain_id is not None:
        chain = model[chain_id]
    else:
        chains = list(model.get_chains())
        if len(chains) != 1:
            raise ValueError(
                "Only single chain PDBs are supported when chain_id not specified. "
                f"Found {len(chains)} chains."
            )
        else:
            chain = chains[0]

    atom_positions = []
    aatype = []
    atom_mask = []
    # residue_index = []
    b_factors = []
    res_name = []

    for res in chain:
        # if res.id[2] != " ":
        #    raise ValueError(
        #        f"PDB contains an insertion code at chain {chain.id} and residue "
        #        f"index {res.id[1]}. These are not supported."
        #    )

        # Exclude water and other solvents
        if res.resname not in constants.restype_1to3.values():
            continue
        res_shortname = constants.restype_3to1.get(res.resname, "X")
        restype_idx = constants.restype_order.get(res_shortname, constants.restype_num)
        pos = np.zeros((constants.atom_type_num, 3))
        mask = np.zeros((constants.atom_type_num,))
        res_b_factors = np.zeros((constants.atom_type_num,))
        for atom in res:
            if atom.name not in constants.atom_types:
                continue
            pos[constants.atom_order[atom.name]] = atom.coord
            mask[constants.atom_order[atom.name]] = 1.0
            res_b_factors[constants.atom_order[atom.name]] = atom.bfactor
        if np.sum(mask) < 0.5:
            # If no known atom positions are reported for the residue then skip it.
            continue
        res_name.append(res_shortname)
        aatype.append(restype_idx)
        atom_positions.append(pos)
        atom_mask.append(mask)
        # residue_index.append(res.id[1])
        b_factors.append(res_b_factors)

    return Protein(
        atom_positions=np.array(atom_positions),
        atom_mask=np.array(atom_mask),
        aatype=np.array(aatype),
        # residue_index=np.array(residue_index),
        b_factors=np.array(b_factors),
        res_name=np.array(res_name),
    )


def get_a3m(protein_sequence, a3m_path: str, unique_id: str):
    filename_query = os.path.join("a3m_generation", 'query_' + unique_id + '.fasta')
    with open(filename_query, "w") as file:
        file.write(">" + unique_id + "\n" + protein_sequence)
    os.makedirs(os.path.dirname(a3m_path), exist_ok=True)
    result = subprocess.run(
        f"./a3m_generation/msa_run --fasta_file {filename_query} --output_file {a3m_path}",
        shell=True,
        capture_output=True,
        text=True,
    )
    os.remove(filename_query)