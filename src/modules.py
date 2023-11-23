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

import math
import os
from functools import partial
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from src import config_multimer, constants, rigid
from src.chunk_utils import chunk_layer
from src.primitives import Attention, GlobalAttention
from src.tensor_utils import masked_mean

"""
Modules Multimer
"""


class DockerIteration(nn.Module):
    def __init__(self, global_config):
        super().__init__()

        with torch.set_grad_enabled(False):
            self.InputEmbedder = InputEmbedding(global_config)
            self.TemplateEmbedding1D = TemplateEmbedding1D(global_config)
            self.Evoformer = nn.ModuleList(
                [
                    EvoformerIteration(
                        global_config["model"]["embeddings_and_evoformer"]["evoformer"],
                        global_config["model"]["embeddings_and_evoformer"],
                        global_config["af_version"],
                    )
                    for _ in range(
                        global_config["model"]["embeddings_and_evoformer"][
                            "evoformer_num_block"
                        ]
                    )
                ]
            )
            self.EvoformerExtractSingleRec = nn.Linear(
                global_config["model"]["embeddings_and_evoformer"]["msa_channel"],
                global_config["model"]["embeddings_and_evoformer"]["seq_channel"],
            )
            self.StructureModule = StructureModule(
                global_config["model"]["heads"]["structure_module"],
                global_config["model"]["embeddings_and_evoformer"],
            )
            self.Distogram = Distogram(global_config)
            self.PredictedLddt = PredictedLddt(
                global_config,
                num_layers=global_config["model"]["heads"]["predicted_aligned_error"][
                    "num_layers"
                ],
            )
            self.PredictedAlignedError = PredictedAlignedError(global_config)
            self.ExperimentallyResolvedHead = ExperimentallyResolvedHead(global_config)
            self.MaskedMsaHead = MaskedMsaHead(global_config)
            self.masked_losses_factor = global_config["model"]["masked_losses_factor"]
            self.global_config = global_config

    def _preprocess_batch_msa(self, batch, index=None, msa_indices_prefix=None):
        batch["msa_profile"] = make_msa_profile(batch)
        batch = sample_msa(
            batch,
            self.global_config["model"]["embeddings_and_evoformer"]["num_msa"],
            index,
            msa_indices_prefix,
        )
        (
            batch["cluster_profile"],
            batch["cluster_deletion_mean"],
        ) = nearest_neighbor_clusters(batch)
        batch["msa_feat"] = create_msa_feat(batch)
        batch["extra_msa_feat"], batch["extra_msa_mask"] = create_extra_msa_feature(
            batch,
            self.global_config["model"]["embeddings_and_evoformer"]["num_extra_msa"],
        )
        if "all_atom_positions" in batch:
            batch["pseudo_beta"], batch["pseudo_beta_mask"] = pseudo_beta_fn(
                batch["aatype"], batch["all_atom_positions"], batch["all_atom_mask"]
            )
        return batch

    def iteration(self, batch, recycle=None):
        msa_activations, pair_activations, msa_mask, pair_mask = checkpoint(
            partial(self.InputEmbedder, recycle=recycle), batch
        )
        num_msa_seq = msa_activations.shape[1]
        if self.global_config["model"]["embeddings_and_evoformer"]["template"][
            "enabled"
        ]:
            template_features, template_masks = self.TemplateEmbedding1D(batch)
            msa_activations = torch.cat(
                (msa_activations, template_features), dim=1
            ).type(torch.float32)
            msa_mask = torch.cat((msa_mask, template_masks), dim=1).type(torch.float32)
            del template_features

        for evo_i, evo_iter in enumerate(self.Evoformer):
            msa_activations, pair_activations = checkpoint(
                evo_iter,
                msa_activations.clone(),
                pair_activations.clone(),
                msa_mask,
                pair_mask,
            )

        single_activations = self.EvoformerExtractSingleRec(msa_activations[:, 0])
        representations = {"single": single_activations, "pair": pair_activations}
        representations["msa"] = msa_activations[:, :num_msa_seq]
        struct_out = self.StructureModule(single_activations, pair_activations, batch)

        representations["structure_module"] = struct_out["act"]

        atom14_pred_positions = struct_out["atom_pos"][-1]
        atom37_pred_positions = atom14_to_atom37(
            atom14_pred_positions.squeeze(0), batch["aatype"].squeeze(0).long()
        )
        atom37_mask = atom_37_mask(batch["aatype"][0].long())

        representations["final_atom14_positions"] = atom14_pred_positions.squeeze(0)
        representations["final_all_atom"] = atom37_pred_positions
        representations["struct_out"] = struct_out
        representations["final_atom_mask"] = atom37_mask

        m_1_prev = msa_activations[:, 0]
        z_prev = pair_activations
        x_prev = atom37_pred_positions.unsqueeze(0)
        del struct_out

        return representations, m_1_prev, z_prev, x_prev

    def forward(self, init_batch, msa_indices_prefix=None):
        index_init = 0
        batch = self._preprocess_batch_msa(init_batch, index_init, msa_indices_prefix)

        recycles = None

        min_num_recycle = self.global_config["model"]["min_num_recycle_eval"]
        confident_plddt = self.global_config["model"]["confident_plddt"]
        num_recycle = self.global_config["model"]["max_num_recycle_eval"]
        for recycle_iter in range(num_recycle):
            out, m_1_prev, z_prev, x_prev = self.iteration(batch, recycles)
            plddt = compute_plddt(self.PredictedLddt(out))
            mean_masked_plddt = (plddt * batch["loss_mask"]).sum() / batch[
                "loss_mask"
            ].sum()
            if recycle_iter >= min_num_recycle and mean_masked_plddt >= confident_plddt:
                break
            elif recycle_iter < (num_recycle - 1):
                recycles = {
                    "prev_msa_first_row": m_1_prev,
                    "prev_pair": z_prev,
                    "prev_pos": x_prev,
                }
                del out, m_1_prev, z_prev, x_prev
            recycle_iter += 1

        del recycles

        distogram_logits, distogram_bin_edges = self.Distogram(out)
        pred_lddt = self.PredictedLddt(out)
        pae_logits, pae_breaks = self.PredictedAlignedError(out)
        resovled_logits = self.ExperimentallyResolvedHead(out)
        masked_msa_logits = self.MaskedMsaHead(out)

        out["distogram"] = {}
        out["predicted_lddt"] = {}
        out["predicted_aligned_error"] = {}
        out["distogram"]["logits"] = distogram_logits
        out["distogram"]["bin_edges"] = distogram_bin_edges
        out["predicted_lddt"]["logits"] = pred_lddt
        out["predicted_aligned_error"]["logits"] = pae_logits
        out["predicted_aligned_error"]["breaks"] = pae_breaks
        out["experimentally_resolved"] = resovled_logits
        out["msa_head"] = masked_msa_logits

        plddt = compute_plddt(pred_lddt)
        return out, plddt


class Distogram(nn.Module):
    def __init__(self, global_config):
        super().__init__()
        self.half_logits = nn.Linear(
            global_config["model"]["embeddings_and_evoformer"]["pair_channel"],
            global_config["model"]["heads"]["distogram"]["num_bins"],
        )
        self.first_break = global_config["model"]["heads"]["distogram"]["first_break"]
        self.last_break = global_config["model"]["heads"]["distogram"]["last_break"]
        self.num_bins = global_config["model"]["heads"]["distogram"]["num_bins"]

    def forward(self, representations):
        pair = representations["pair"]
        half_logits = self.half_logits(pair)
        logits = half_logits + half_logits.transpose(-2, -3)
        breaks = torch.linspace(
            self.first_break, self.last_break, self.num_bins - 1, device=pair.device
        )
        return logits, breaks


class EvoformerIteration(nn.Module):
    def __init__(self, config, global_config, af_version):
        super().__init__()
        self.OuterProductMean = OuterProductMean(
            config["outer_product_mean"], global_config
        )
        self.RowAttentionWithPairBias = RowAttentionWithPairBias(
            config["msa_row_attention_with_pair_bias"], global_config
        )
        self.LigColumnAttention = LigColumnAttention(
            config["msa_column_attention"], global_config
        )
        self.RecTransition = Transition(config["msa_transition"], global_config)
        self.TriangleMultiplicationOutgoing = TriangleMultiplicationOutgoing(
            config["triangle_multiplication_outgoing"], global_config, af_version
        )
        self.TriangleMultiplicationIngoing = TriangleMultiplicationIngoing(
            config["triangle_multiplication_incoming"], global_config, af_version
        )
        self.TriangleAttentionStartingNode = TriangleAttentionStartingNode(
            config["triangle_attention_starting_node"], global_config
        )
        self.TriangleAttentionEndingNode = TriangleAttentionEndingNode(
            config["triangle_attention_ending_node"], global_config
        )
        self.PairTransition = Transition(config["pair_transition"], global_config)

    def forward(self, msa_act, pair_act, msa_mask, pair_mask):
        pair_act = pair_act + self.OuterProductMean(msa_act, msa_mask)
        msa_act = msa_act + self.RowAttentionWithPairBias(msa_act, pair_act, msa_mask)
        msa_act = msa_act + self.LigColumnAttention(msa_act, msa_mask)
        msa_act = msa_act + self.RecTransition(msa_act, msa_mask)
        pair_act = pair_act + self.TriangleMultiplicationOutgoing(pair_act, pair_mask)
        pair_act = pair_act + self.TriangleMultiplicationIngoing(pair_act, pair_mask)
        pair_act = pair_act + self.TriangleAttentionStartingNode(pair_act, pair_mask)
        pair_act = pair_act + self.TriangleAttentionEndingNode(pair_act, pair_mask)
        pair_act = pair_act + self.PairTransition(pair_act, pair_mask)
        return msa_act, pair_act


class ExperimentallyResolvedHead(nn.Module):
    def __init__(self, global_config):
        super().__init__()
        self.logits = nn.Linear(
            global_config["model"]["embeddings_and_evoformer"]["seq_channel"], 37
        )

    def forward(self, representations):
        return self.logits(representations["single"])


class ExtraColumnGlobalAttention(nn.Module):
    def __init__(self, config, global_config, af_version):
        super().__init__()
        self.af_version = af_version
        self.attn_num_c = config["attention_channel"]
        self.num_heads = config["num_head"]
        self.global_config = global_config
        self.query_norm = nn.LayerNorm(global_config["extra_msa_channel"])
        if af_version == 2:
            self.q = nn.Linear(
                global_config["extra_msa_channel"],
                self.attn_num_c * self.num_heads,
                bias=False,
            )
            self.k = nn.Linear(
                global_config["extra_msa_channel"], self.attn_num_c, bias=False
            )
            self.v = nn.Linear(
                global_config["extra_msa_channel"], self.attn_num_c, bias=False
            )
            self.gate = nn.Linear(
                global_config["extra_msa_channel"], self.attn_num_c * self.num_heads
            )
            self.output = nn.Linear(
                self.attn_num_c * self.num_heads, global_config["extra_msa_channel"]
            )
        elif af_version == 3:
            in_num_c = global_config["extra_msa_channel"]
            self.mha = GlobalAttention(
                in_num_c, self.attn_num_c, self.num_heads, 1e9, 1e-10
            )

    @torch.jit.ignore
    def _chunk(
        self,
        m: torch.Tensor,
        mask: torch.Tensor,
        chunk_size: int,
    ) -> torch.Tensor:
        def fn(m, mask):
            m = self.query_norm(m)
            return self.mha(
                m,
                mask,
            )

        return chunk_layer(
            fn,
            {
                "m": m,
                "mask": mask,
            },
            chunk_size=chunk_size,
            no_batch_dims=len(m.shape[:-2]),
        )

    def forward(self, msa_act, msa_mask):
        msa_act = msa_act.transpose(-2, -3)
        msa_mask = msa_mask.transpose(-1, -2)
        if self.af_version == 2:
            msa_act = self.query_norm(msa_act)
            q_avg = torch.sum(msa_act, dim=-2) / msa_act.shape[-2]
            q = self.q(q_avg).view(*q_avg.shape[:-1], self.num_heads, self.attn_num_c)
            q = q * (self.attn_num_c ** (-0.5))
            k = self.k(msa_act)
            v = self.v(msa_act)
            gate = torch.sigmoid(
                self.gate(msa_act).view(
                    *msa_act.shape[:-1], self.num_heads, self.attn_num_c
                )
            )
            w = torch.softmax(torch.einsum("bihc,bikc->bihk", q, k), dim=-1)
            out_1d = torch.einsum("bmhk,bmkc->bmhc", w, v)
            out_1d = out_1d.unsqueeze(-3) * gate
            out = self.output(
                out_1d.view(*out_1d.shape[:-2], self.attn_num_c * self.num_heads)
            )
        elif self.af_version == 3:
            out = self._chunk(msa_act, msa_mask, 64)
        return out.transpose(-2, -3)


class FragExtraStack(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                FragExtraStackIteration(
                    config["model"]["embeddings_and_evoformer"]["extra_msa"],
                    config["model"]["embeddings_and_evoformer"],
                    config["af_version"],
                )
                for _ in range(
                    config["model"]["embeddings_and_evoformer"][
                        "extra_msa_stack_num_block"
                    ]
                )
            ]
        )

    def forward(self, msa_act, pair_act, extra_mask_msa, extra_mask_pair):
        for l in self.layers:
            msa_act, pair_act = checkpoint(
                l, msa_act.clone(), pair_act.clone(), extra_mask_msa, extra_mask_pair
            )
        return pair_act


class FragExtraStackIteration(torch.nn.Module):
    def __init__(self, config, global_config, af_version):
        super().__init__()
        self.RowAttentionWithPairBias = RowAttentionWithPairBias(
            config["msa_row_attention_with_pair_bias"], global_config
        )
        self.ExtraColumnGlobalAttention = ExtraColumnGlobalAttention(
            config["msa_column_attention"], global_config, af_version
        )
        self.RecTransition = Transition(config["msa_transition"], global_config)
        self.OuterProductMean = OuterProductMean(
            config["outer_product_mean"], global_config
        )
        self.TriangleMultiplicationOutgoing = TriangleMultiplicationOutgoing(
            config["triangle_multiplication_outgoing"], global_config, af_version
        )
        self.TriangleMultiplicationIngoing = TriangleMultiplicationIngoing(
            config["triangle_multiplication_incoming"], global_config, af_version
        )
        self.TriangleAttentionStartingNode = TriangleAttentionStartingNode(
            config["triangle_attention_starting_node"], global_config
        )
        self.TriangleAttentionEndingNode = TriangleAttentionEndingNode(
            config["triangle_attention_ending_node"], global_config
        )
        self.PairTransition = Transition(config["pair_transition"], global_config)

    def forward(self, msa_act, pair_act, msa_mask, pair_mask):
        pair_act = pair_act + self.OuterProductMean(msa_act, msa_mask)
        msa_act = msa_act + self.RowAttentionWithPairBias(msa_act, pair_act, msa_mask)
        msa_act = msa_act + self.ExtraColumnGlobalAttention(msa_act, msa_mask)
        msa_act = msa_act + self.RecTransition(msa_act, msa_mask)
        pair_act = pair_act + self.TriangleMultiplicationOutgoing(pair_act, pair_mask)
        pair_act = pair_act + self.TriangleMultiplicationIngoing(pair_act, pair_mask)
        pair_act = pair_act + self.TriangleAttentionStartingNode(pair_act, pair_mask)
        pair_act = pair_act + self.TriangleAttentionEndingNode(pair_act, pair_mask)
        pair_act = pair_act + self.PairTransition(pair_act, pair_mask)
        return msa_act, pair_act


class InputEmbedding(nn.Module):
    def __init__(self, global_config):
        super().__init__()
        self.preprocessing_1d = nn.Linear(
            global_config["aatype"],
            global_config["model"]["embeddings_and_evoformer"]["msa_channel"],
        )
        self.left_single = nn.Linear(
            global_config["aatype"],
            global_config["model"]["embeddings_and_evoformer"]["pair_channel"],
        )
        self.right_single = nn.Linear(
            global_config["aatype"],
            global_config["model"]["embeddings_and_evoformer"]["pair_channel"],
        )
        self.preprocess_msa = nn.Linear(
            global_config["msa"],
            global_config["model"]["embeddings_and_evoformer"]["msa_channel"],
        )
        self.max_seq = global_config["model"]["embeddings_and_evoformer"]["num_msa"]
        self.msa_channel = global_config["model"]["embeddings_and_evoformer"][
            "msa_channel"
        ]
        self.pair_channel = global_config["model"]["embeddings_and_evoformer"][
            "pair_channel"
        ]
        self.num_extra_msa = global_config["model"]["embeddings_and_evoformer"][
            "num_extra_msa"
        ]
        self.global_config = global_config

        self.TemplateEmbedding = TemplateEmbedding(
            global_config["model"]["embeddings_and_evoformer"]["template"],
            global_config,
        )
        self.RecyclingEmbedder = RecyclingEmbedder(global_config)
        self.extra_msa_activations = nn.Linear(
            global_config["extra_msa_act"],
            global_config["model"]["embeddings_and_evoformer"]["extra_msa_channel"],
        )
        self.FragExtraStack = FragExtraStack(global_config)

    def forward(self, batch, recycle):
        num_batch, num_res = batch["aatype"].shape[0], batch["aatype"].shape[1]
        target_feat = nn.functional.one_hot(batch["aatype"].long(), 21).float()
        preprocessed_1d = self.preprocessing_1d(target_feat)
        left_single = self.left_single(target_feat)
        right_single = self.right_single(target_feat)
        pair_activations = left_single.unsqueeze(2) + right_single.unsqueeze(1)
        preprocess_msa = self.preprocess_msa(batch["msa_feat"])
        msa_activations = preprocess_msa + preprocessed_1d
        mask_2d = batch["seq_mask"][..., None] * batch["seq_mask"][..., None, :]
        mask_2d = mask_2d.type(torch.float32)
        if self.global_config["recycle"] and recycle == None:
            recycle = {
                "prev_pos": torch.zeros(num_batch, num_res, 37, 3).to(
                    batch["aatype"].device
                ),
                "prev_msa_first_row": torch.zeros(
                    num_batch, num_res, self.msa_channel
                ).to(batch["aatype"].device),
                "prev_pair": torch.zeros(
                    num_batch, num_res, num_res, self.pair_channel
                ).to(batch["aatype"].device),
            }

        if recycle is not None:
            prev_msa_first_row, pair_activation_update = self.RecyclingEmbedder(
                batch, recycle
            )
            pair_activations = pair_activations + pair_activation_update
            msa_activations[:, 0] += prev_msa_first_row
            del recycle

        if self.global_config["model"]["embeddings_and_evoformer"]["template"][
            "enabled"
        ]:
            template_batch = {
                "template_aatype": batch["template_aatype"][0],
                "template_all_atom_positions": batch["template_all_atom_positions"][0],
                "template_all_atom_mask": batch["template_all_atom_mask"][0],
            }
            multichain_mask = (
                batch["asym_id"][..., None] == batch["asym_id"][:, None, ...]
            )
            template_act = self.TemplateEmbedding(
                pair_activations[0], template_batch, mask_2d[0], multichain_mask[0]
            )
            pair_activations = pair_activations + template_act
            del template_batch

        extra_msa_activations = self.extra_msa_activations(batch["extra_msa_feat"])

        pair_activations = self.FragExtraStack(
            extra_msa_activations,
            pair_activations,
            batch["extra_msa_mask"].type(torch.float32),
            mask_2d,
        )
        msa_mask = batch["msa_mask"]
        del target_feat
        return msa_activations, pair_activations, msa_mask, mask_2d


class LigColumnAttention(nn.Module):
    def __init__(self, config, global_config):
        super().__init__()

        attn_num_c = config["attention_channel"]
        num_heads = config["num_head"]
        in_num_c = global_config["msa_channel"]

        self.query_norm = nn.LayerNorm(in_num_c)
        self.attn_num_c = attn_num_c
        self.num_heads = num_heads
        self.mha = Attention(in_num_c, in_num_c, in_num_c, attn_num_c, num_heads)

    @torch.jit.ignore
    def _chunk(
        self,
        m: torch.Tensor,
        biases: List[torch.Tensor],
        chunk_size: int,
    ) -> torch.Tensor:
        def fn(m, biases):
            m = self.query_norm(m)
            return self.mha(q_x=m, kv_x=m, biases=biases)

        return chunk_layer(
            fn,
            {
                "m": m,
                "biases": biases,
            },
            chunk_size=chunk_size,
            no_batch_dims=len(m.shape[:-2]),
        )

    def forward(self, msa_act, msa_mask):
        msa_act = msa_act.transpose(-2, -3)
        msa_mask = msa_mask.transpose(-1, -2)
        bias = (1e9 * (msa_mask - 1.0))[..., :, None, None, :]
        biases = [bias]
        out_1d = self._chunk(
            msa_act,
            biases,
            64,
        )

        out_1d = out_1d.transpose(-2, -3)

        return out_1d


class MaskedMsaHead(nn.Module):
    def __init__(self, global_config):
        super().__init__()
        self.logits = nn.Linear(
            global_config["model"]["embeddings_and_evoformer"]["msa_channel"], 22
        )

    def forward(self, representations):
        return self.logits(representations["msa"])


class OuterProductMean(nn.Module):
    def __init__(self, config, global_config):
        super().__init__()
        in_c = config["norm_channel"]
        out_c = config["num_output_channel"]
        mid_c = config["num_outer_channel"]
        self.layer_norm_input = nn.LayerNorm(in_c)
        self.left_projection = nn.Linear(in_c, mid_c)
        self.right_projection = nn.Linear(in_c, mid_c)
        self.output = nn.Linear(mid_c * mid_c, out_c)
        self.mid_c = mid_c
        self.out_c = out_c

    def forward(self, act, mask):
        act = self.layer_norm_input(act)
        mask = mask[..., None]
        left_act = mask * self.left_projection(act)
        right_act = mask * self.right_projection(act)
        x2d = torch.einsum("bmix,bmjy->bjixy", left_act, right_act)  # / x1d.shape[1]
        out = self.output(x2d.flatten(start_dim=-2)).transpose(-2, -3)
        norm = torch.einsum("...abc,...adc->...bdc", mask, mask)
        out = out / (norm + 1e-3)
        return out


class PredictedAlignedError(nn.Module):
    def __init__(self, global_config):
        super().__init__()
        self.logits = nn.Linear(
            global_config["model"]["embeddings_and_evoformer"]["pair_channel"],
            global_config["model"]["heads"]["predicted_aligned_error"]["num_bins"],
        )
        self.max_error_bin = global_config["model"]["heads"]["predicted_aligned_error"][
            "max_error_bin"
        ]
        self.num_bins = global_config["model"]["heads"]["predicted_aligned_error"][
            "num_bins"
        ]

    def forward(self, representations):
        act = representations["pair"]
        logits = self.logits(act)
        breaks = torch.linspace(
            0.0, self.max_error_bin, self.num_bins - 1, device=act.device
        )
        return logits, breaks


class PredictedLddt(nn.Module):
    def __init__(self, global_config, num_layers=0):
        super().__init__()
        self.input_layer_norm = nn.LayerNorm(
            global_config["model"]["embeddings_and_evoformer"]["seq_channel"]
        )
        self.act_0 = nn.Linear(
            global_config["model"]["embeddings_and_evoformer"]["seq_channel"],
            global_config["model"]["heads"]["predicted_lddt"]["num_channels"],
        )
        self.act_1 = nn.Linear(
            global_config["model"]["heads"]["predicted_lddt"]["num_channels"],
            global_config["model"]["heads"]["predicted_lddt"]["num_channels"],
        )
        self.linear_layers = nn.ModuleList()
        for i in range(num_layers):
            self.linear_layers.append(
                nn.Linear(
                    global_config["model"]["heads"]["predicted_lddt"]["num_channels"],
                    global_config["model"]["heads"]["predicted_lddt"]["num_channels"],
                )
            )
        self.logits = nn.Linear(
            global_config["model"]["heads"]["predicted_lddt"]["num_channels"],
            global_config["model"]["heads"]["predicted_lddt"]["num_bins"],
        )

    def forward(self, representations):
        act = representations["structure_module"]
        act = self.input_layer_norm(act)
        act = self.act_0(act).relu_()
        act = self.act_1(act).relu_()

        for linear_layer in self.linear_layers:
            act = linear_layer(act).relu_()

        logits = self.logits(act)
        return logits


class RecyclingEmbedder(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.prev_pos_linear = nn.Linear(
            config["model"]["embeddings_and_evoformer"]["prev_pos"]["num_bins"],
            config["model"]["embeddings_and_evoformer"]["pair_channel"],
        )
        self.max_bin = config["model"]["embeddings_and_evoformer"]["prev_pos"][
            "max_bin"
        ]
        self.min_bin = config["model"]["embeddings_and_evoformer"]["prev_pos"][
            "min_bin"
        ]
        self.num_bins = config["model"]["embeddings_and_evoformer"]["prev_pos"][
            "num_bins"
        ]
        self.config = config
        self.prev_pair_norm = nn.LayerNorm(
            config["model"]["embeddings_and_evoformer"]["pair_channel"]
        )
        self.prev_msa_first_row_norm = nn.LayerNorm(
            config["model"]["embeddings_and_evoformer"]["msa_channel"]
        )
        self.position_activations = nn.Linear(
            config["rel_feat"],
            config["model"]["embeddings_and_evoformer"]["pair_channel"],
        )

    def _relative_encoding(self, batch):
        c = self.config["model"]["embeddings_and_evoformer"]
        rel_feats = []
        pos = batch["residue_index"]
        asym_id = batch["asym_id"]
        asym_id_same = torch.eq(asym_id[..., None], asym_id[..., None, :])
        offset = pos[..., None] - pos[..., None, :]

        clipped_offset = torch.clip(
            offset + c["max_relative_idx"], min=0, max=2 * c["max_relative_idx"]
        )

        if c["use_chain_relative"]:
            final_offset = torch.where(
                asym_id_same,
                clipped_offset,
                (2 * c["max_relative_idx"] + 1) * torch.ones_like(clipped_offset),
            )

            rel_pos = torch.nn.functional.one_hot(
                final_offset.long(), 2 * c["max_relative_idx"] + 2
            )

            rel_feats.append(rel_pos)

            entity_id = batch["entity_id"]
            entity_id_same = torch.eq(entity_id[..., None], entity_id[..., None, :])
            rel_feats.append(entity_id_same.type(rel_pos.dtype)[..., None])

            sym_id = batch["sym_id"]
            rel_sym_id = sym_id[..., None] - sym_id[..., None, :]

            max_rel_chain = c["max_relative_chain"]

            clipped_rel_chain = torch.clip(
                rel_sym_id + max_rel_chain, min=0, max=2 * max_rel_chain
            )

            final_rel_chain = torch.where(
                entity_id_same,
                clipped_rel_chain,
                (2 * max_rel_chain + 1) * torch.ones_like(clipped_rel_chain),
            )
            rel_chain = torch.nn.functional.one_hot(
                final_rel_chain.long(), 2 * c["max_relative_chain"] + 2
            )

            rel_feats.append(rel_chain)

        else:
            rel_pos = torch.nn.functional.one_hot(
                clipped_offset.long(), 2 * c["max_relative_idx"] + 1
            )
            rel_feats.append(rel_pos)

        rel_feat = torch.cat(rel_feats, -1)
        return rel_feat

    def forward(self, batch, recycle):
        prev_pseudo_beta = pseudo_beta_fn(batch["aatype"], recycle["prev_pos"], None)
        dgram = torch.sum(
            (prev_pseudo_beta[..., None, :] - prev_pseudo_beta[..., None, :, :]) ** 2,
            dim=-1,
            keepdim=True,
        )
        lower = (
            torch.linspace(
                self.min_bin,
                self.max_bin,
                self.num_bins,
                device=prev_pseudo_beta.device,
            )
            ** 2
        )
        upper = torch.cat([lower[1:], lower.new_tensor([1e8])], dim=-1)
        dgram = ((dgram > lower) * (dgram < upper)).type(dgram.dtype)
        prev_pos_linear = self.prev_pos_linear(dgram)
        pair_activation_update = prev_pos_linear + self.prev_pair_norm(
            recycle["prev_pair"]
        )
        rel_feat = self._relative_encoding(batch)
        pair_activation_update = pair_activation_update + self.position_activations(
            rel_feat.float()
        )
        prev_msa_first_row = self.prev_msa_first_row_norm(recycle["prev_msa_first_row"])
        del dgram, prev_pseudo_beta

        return prev_msa_first_row, pair_activation_update


class RowAttentionWithPairBias(nn.Module):
    def __init__(self, config, global_config):
        super().__init__()

        attn_num_c = config["attention_channel"]
        num_heads = config["num_head"]
        in_num_c = config["norm_channel"]
        pair_rep_num_c = global_config["pair_channel"]

        self.query_norm = nn.LayerNorm(in_num_c)
        self.feat_2d_norm = nn.LayerNorm(pair_rep_num_c)
        self.feat_2d_weights = nn.Linear(pair_rep_num_c, num_heads, bias=False)
        self.attn_num_c = attn_num_c
        self.num_heads = num_heads
        self.mha = Attention(in_num_c, in_num_c, in_num_c, attn_num_c, num_heads)

    @torch.jit.ignore
    def _chunk(
        self,
        m: torch.Tensor,
        biases: List[torch.Tensor],
        chunk_size: int,
    ) -> torch.Tensor:
        def fn(m, biases):
            m = self.query_norm(m)
            return self.mha(
                q_x=m,
                kv_x=m,
                biases=biases,
            )

        return chunk_layer(
            fn,
            {
                "m": m,
                "biases": biases,
            },
            chunk_size=chunk_size,
            no_batch_dims=len(m.shape[:-2]),
        )

    def forward(self, msa_act, pair_act, msa_mask):
        chunks = []

        for i in range(0, pair_act.shape[-3], 256):
            z_chunk = pair_act[..., i : i + 256, :, :]

            # [*, N_res, N_res, C_z]
            z_chunk = self.feat_2d_norm(z_chunk)

            # [*, N_res, N_res, no_heads]
            z_chunk = self.feat_2d_weights(z_chunk)

            chunks.append(z_chunk)

        z = torch.cat(chunks, dim=-3)

        z = z.permute(0, 3, 1, 2).unsqueeze(-4)
        bias = (1e9 * (msa_mask - 1.0))[..., :, None, None, :]
        biases = [bias, z]
        out_1d = self._chunk(
            msa_act,
            biases,
            64,
        )

        return out_1d


class SingleTemplateEmbedding(nn.Module):
    def __init__(self, config, global_config):
        super().__init__()
        self.query_embedding_norm = nn.LayerNorm(
            global_config["model"]["embeddings_and_evoformer"]["pair_channel"]
        )
        self.TemplateEmbeddingIteration = nn.ModuleList(
            [
                TemplateEmbeddingIteration(config["template_pair_stack"], global_config)
                for _ in range(config["template_pair_stack"]["num_block"])
            ]
        )
        self.output_layer_norm = nn.LayerNorm(config["num_channels"])
        self.template_pair_emb_0 = nn.Linear(
            config["dgram_features"]["num_bins"], config["num_channels"]
        )
        self.template_pair_emb_1 = nn.Linear(1, config["num_channels"])
        self.template_pair_emb_2 = nn.Linear(22, config["num_channels"])
        self.template_pair_emb_3 = nn.Linear(22, config["num_channels"])
        self.template_pair_emb_4 = nn.Linear(1, config["num_channels"])
        self.template_pair_emb_5 = nn.Linear(1, config["num_channels"])
        self.template_pair_emb_6 = nn.Linear(1, config["num_channels"])
        self.template_pair_emb_7 = nn.Linear(1, config["num_channels"])
        self.template_pair_emb_8 = nn.Linear(
            global_config["model"]["embeddings_and_evoformer"]["pair_channel"],
            config["num_channels"],
        )

        self.max_bin = config["dgram_features"]["max_bin"]
        self.min_bin = config["dgram_features"]["min_bin"]
        self.num_bins = config["dgram_features"]["num_bins"]
        self.interchain_enabled = global_config["model"]["embeddings_and_evoformer"][
            "template"
        ]["interchain_enabled"]

    def forward(
        self,
        query_embedding,
        template_aatype,
        template_all_atom_positions,
        template_all_atom_mask,
        padding_mask_2d,
        multichain_mask_2d,
    ):
        query_embedding = query_embedding.clone()
        template_positions, pseudo_beta_mask = pseudo_beta_fn(
            template_aatype, template_all_atom_positions, template_all_atom_mask
        )
        pseudo_beta_mask_2d = pseudo_beta_mask[:, None] * pseudo_beta_mask[None, :]
        # Enable interchain contacts
        if self.interchain_enabled:
            multichain_mask_2d = torch.ones_like(multichain_mask_2d, dtype=torch.bool)
        pseudo_beta_mask_2d *= multichain_mask_2d

        pseudo_beta_mask_2d *= multichain_mask_2d

        dgram = torch.sum(
            (template_positions[..., None, :] - template_positions[..., None, :, :])
            ** 2,
            dim=-1,
            keepdim=True,
        )
        lower = (
            torch.linspace(
                self.min_bin,
                self.max_bin,
                self.num_bins,
                device=template_positions.device,
            )
            ** 2
        )
        upper = torch.cat([lower[1:], lower.new_tensor([1e8])], dim=-1)
        template_dgram = ((dgram > lower) * (dgram < upper)).type(dgram.dtype)
        template_dgram *= pseudo_beta_mask_2d[..., None]
        template_dgram = template_dgram.type(query_embedding.dtype)
        pseudo_beta_mask_2d = pseudo_beta_mask_2d.type(query_embedding.dtype)
        aatype = nn.functional.one_hot(template_aatype, 22).type(query_embedding.dtype)
        raw_atom_pos = template_all_atom_positions
        n, ca, c = [constants.atom_order[a] for a in ["N", "CA", "C"]]
        rigids = rigid.Rigid.make_transform_from_reference(
            n_xyz=raw_atom_pos[..., n, :],
            ca_xyz=raw_atom_pos[..., ca, :],
            c_xyz=raw_atom_pos[..., c, :],
            eps=1e-20,
        )
        backbone_mask = (
            template_all_atom_mask[:, n]
            * template_all_atom_mask[:, ca]
            * template_all_atom_mask[:, c]
        ).float()
        points = rigids.get_trans()[..., None, :, :]
        rigid_vec = rigids[..., None].invert_apply(points)
        inv_distance_scalar = torch.rsqrt(1e-20 + torch.sum(rigid_vec**2, dim=-1))
        backbone_mask_2d = backbone_mask[:, None] * backbone_mask[None, :]
        backbone_mask_2d *= multichain_mask_2d
        unit_vector = rigid_vec * inv_distance_scalar[..., None]
        unit_vector = unit_vector * backbone_mask_2d[..., None]
        unbind_unit_vector = torch.unbind(unit_vector[..., None, :], dim=-1)
        query_embedding = self.query_embedding_norm(query_embedding)
        act = self.template_pair_emb_0(template_dgram)
        act = act + self.template_pair_emb_1(pseudo_beta_mask_2d[..., None])
        act = act + self.template_pair_emb_2(aatype[None, :, :])
        act = act + self.template_pair_emb_3(aatype[:, None, :])
        act = act + self.template_pair_emb_4(unbind_unit_vector[0])
        act = act + self.template_pair_emb_5(unbind_unit_vector[1])
        act = act + self.template_pair_emb_6(unbind_unit_vector[2])
        act = act + self.template_pair_emb_7(backbone_mask_2d[..., None])
        act = act + self.template_pair_emb_8(query_embedding)
        act = torch.unsqueeze(act, dim=0)
        for iter_temp in self.TemplateEmbeddingIteration:
            act = iter_temp(act, torch.unsqueeze(padding_mask_2d, dim=0))
        act = torch.squeeze(act)
        act = self.output_layer_norm(act)
        return act


class TemplateEmbedding(nn.Module):
    def __init__(self, config, global_config):
        super().__init__()
        self.SingleTemplateEmbedding = SingleTemplateEmbedding(config, global_config)
        self.relu = nn.ReLU()
        self.output_linear = nn.Linear(
            config["num_channels"],
            global_config["model"]["embeddings_and_evoformer"]["pair_channel"],
        )
        self.num_channels = config["num_channels"]

    def forward(
        self, query_embedding, template_batch, padding_mask_2d, multichain_mask_2d
    ):
        num_templates = template_batch["template_aatype"].shape[0]
        num_res, _, query_num_channels = query_embedding.shape
        scan_init = torch.zeros(
            (num_res, num_res, self.num_channels),
            device=query_embedding.device,
            dtype=query_embedding.dtype,
        )
        for i in range(num_templates):
            partial_emb = self.SingleTemplateEmbedding(
                query_embedding,
                template_batch["template_aatype"][i],
                template_batch["template_all_atom_positions"][i],
                template_batch["template_all_atom_mask"][i],
                padding_mask_2d,
                multichain_mask_2d,
            )
            scan_init = scan_init + partial_emb
        embedding = scan_init / num_templates
        embedding = self.relu(embedding)
        embedding = self.output_linear(embedding)
        embedding = torch.unsqueeze(embedding, dim=0)
        return embedding


class TemplateEmbedding1D(torch.nn.Module):
    def __init__(self, global_config):
        super().__init__()
        self.template_single_embedding = nn.Linear(
            34, global_config["model"]["embeddings_and_evoformer"]["msa_channel"]
        )
        self.template_projection = nn.Linear(
            global_config["model"]["embeddings_and_evoformer"]["msa_channel"],
            global_config["model"]["embeddings_and_evoformer"]["msa_channel"],
        )
        self.relu = nn.ReLU()

    def forward(self, batch):
        aatype_one_hot = nn.functional.one_hot(batch["template_aatype"], 22)

        num_templates = batch["template_aatype"].shape[1]
        all_chi_angles = []
        all_chi_masks = []
        for i in range(num_templates):
            template_chi_angles, template_chi_mask = compute_chi_angles(
                batch["template_all_atom_positions"][0][i, :, :, :],
                batch["template_all_atom_mask"][0][i, :, :],
                batch["template_aatype"][0][i, :],
            )
            all_chi_angles.append(template_chi_angles)
            all_chi_masks.append(template_chi_mask)
        chi_angles = torch.stack(all_chi_angles, dim=0)
        chi_angles = torch.unsqueeze(chi_angles, dim=0)
        chi_mask = torch.stack(all_chi_masks, dim=0)
        chi_mask = torch.unsqueeze(chi_mask, dim=0)

        template_features = torch.cat(
            (
                aatype_one_hot,
                torch.sin(chi_angles) * chi_mask,
                torch.cos(chi_angles) * chi_mask,
                chi_mask,
            ),
            dim=-1,
        ).type(torch.float32)
        template_mask = chi_mask[..., 0]

        template_activations = self.template_single_embedding(template_features)
        template_activations = self.relu(template_activations)
        template_activations = self.template_projection(template_activations)

        return template_activations, template_mask


class TemplateEmbeddingIteration(nn.Module):
    def __init__(self, config, global_config):
        super().__init__()
        self.TriangleMultiplicationOutgoing = TriangleMultiplicationOutgoing(
            config["triangle_multiplication_outgoing"],
            global_config,
            global_config["af_version"],
        )
        self.TriangleMultiplicationIngoing = TriangleMultiplicationIngoing(
            config["triangle_multiplication_incoming"],
            global_config,
            global_config["af_version"],
        )
        self.TriangleAttentionStartingNode = TriangleAttentionStartingNode(
            config["triangle_attention_starting_node"], global_config
        )
        self.TriangleAttentionEndingNode = TriangleAttentionEndingNode(
            config["triangle_attention_ending_node"], global_config
        )
        self.PairTransition = Transition(config["pair_transition"], global_config)

    def forward(self, act, pair_mask):
        act = act + self.TriangleMultiplicationOutgoing(act, pair_mask)
        act = act + self.TriangleMultiplicationIngoing(act, pair_mask)
        act = act + self.TriangleAttentionStartingNode(act, pair_mask)
        act = act + self.TriangleAttentionEndingNode(act, pair_mask)
        act = act + self.PairTransition(act, pair_mask)
        return act


class Transition(nn.Module):
    def __init__(self, config, global_config):
        super().__init__()
        self.input_layer_norm = nn.LayerNorm(config["norm_channel"])
        self.transition1 = nn.Linear(
            config["norm_channel"],
            config["norm_channel"] * config["num_intermediate_factor"],
        )
        self.transition2 = nn.Linear(
            config["norm_channel"] * config["num_intermediate_factor"],
            config["norm_channel"],
        )

    def forward(self, act, mask):
        act = self.input_layer_norm(act)
        act = self.transition1(act).relu_()
        act = self.transition2(act)
        return act


class TriangleMultiplicationOutgoing(nn.Module):
    def __init__(self, config, global_config, af_version):
        super().__init__()
        in_c = config["norm_channel"]
        mid_c = config["num_intermediate_channel"]
        self.fused_projection = config["fuse_projection_weights"]

        if af_version == 3:
            self.mid_c = mid_c
        if config["fuse_projection_weights"]:
            self.left_norm_input = nn.LayerNorm(in_c)
            self.projection = nn.Linear(in_c, 2 * mid_c)
            self.gate = nn.Linear(in_c, 2 * mid_c)
        else:
            self.layer_norm_input = nn.LayerNorm(in_c)
            self.left_projection = nn.Linear(in_c, mid_c)
            self.right_projection = nn.Linear(in_c, mid_c)
            self.left_gate = nn.Linear(in_c, mid_c)
            self.right_gate = nn.Linear(in_c, mid_c)
        self.center_layer_norm = nn.LayerNorm(mid_c)
        self.output_projection = nn.Linear(mid_c, in_c)
        self.gating_linear = nn.Linear(in_c, in_c)

    def forward(self, act, mask):
        if self.fused_projection:
            left_act = self.left_norm_input(act)
            mask = mask[..., None]
            proj_act = (
                mask * self.projection(left_act) * torch.sigmoid(self.gate(left_act))
            )
            left_proj = proj_act[..., : self.mid_c]
            right_proj = proj_act[..., self.mid_c :]
            gate_values = self.gating_linear(left_act)
        else:
            act = self.layer_norm_input(act)
            mask = mask[..., None]
            left_proj = (
                mask * self.left_projection(act) * torch.sigmoid(self.left_gate(act))
            )
            right_proj = (
                mask * self.right_projection(act) * torch.sigmoid(self.right_gate(act))
            )
            gate_values = self.gating_linear(act)
        out = torch.einsum("bikc,bjkc->bijc", left_proj, right_proj)
        out = self.center_layer_norm(out)
        out = self.output_projection(out)
        out = out * torch.sigmoid(gate_values)
        return out


class TriangleMultiplicationIngoing(nn.Module):
    def __init__(self, config, global_config, af_version):
        super().__init__()
        in_c = config["norm_channel"]
        mid_c = config["num_intermediate_channel"]
        self.fused_projection = config["fuse_projection_weights"]

        if af_version == 3:
            self.mid_c = mid_c

        if config["fuse_projection_weights"]:
            self.left_norm_input = nn.LayerNorm(in_c)
            self.projection = nn.Linear(in_c, 2 * mid_c)
            self.gate = nn.Linear(in_c, 2 * mid_c)
        else:
            self.layer_norm_input = nn.LayerNorm(in_c)
            self.left_projection = nn.Linear(in_c, mid_c)
            self.right_projection = nn.Linear(in_c, mid_c)
            self.left_gate = nn.Linear(in_c, mid_c)
            self.right_gate = nn.Linear(in_c, mid_c)
        self.center_layer_norm = nn.LayerNorm(mid_c)
        self.output_projection = nn.Linear(mid_c, in_c)
        self.gating_linear = nn.Linear(in_c, in_c)

    def forward(self, act, mask):
        if self.fused_projection:
            left_act = self.left_norm_input(act)
            mask = mask[..., None]
            proj_act = (
                mask * self.projection(left_act) * torch.sigmoid(self.gate(left_act))
            )
            left_proj = proj_act[..., : self.mid_c]
            right_proj = proj_act[..., self.mid_c :]
            gate_values = self.gating_linear(left_act)
        else:
            act = self.layer_norm_input(act)
            mask = mask[..., None]
            left_proj = (
                mask * self.left_projection(act) * torch.sigmoid(self.left_gate(act))
            )
            right_proj = (
                mask * self.right_projection(act) * torch.sigmoid(self.right_gate(act))
            )
            gate_values = self.gating_linear(act)
        out = torch.einsum("bkjc,bkic->bijc", left_proj, right_proj)
        out = self.center_layer_norm(out)
        out = self.output_projection(out)
        out = out * torch.sigmoid(gate_values)
        return out


class TriangleAttentionStartingNode(nn.Module):
    def __init__(self, config, global_config):
        super().__init__()
        attn_num_c = config["attention_channel"]
        num_heads = config["num_head"]
        num_in_c = config["norm_channel"]
        self.attn_num_c = attn_num_c
        self.num_heads = num_heads

        self.query_norm = nn.LayerNorm(num_in_c)
        self.feat_2d_weights = nn.Linear(num_in_c, num_heads, bias=False)

        self.mha = Attention(num_in_c, num_in_c, num_in_c, attn_num_c, num_heads)

    @torch.jit.ignore
    def _chunk(
        self,
        x: torch.Tensor,
        biases: List[torch.Tensor],
        chunk_size: int,
        inplace_safe: bool = False,
    ) -> torch.Tensor:
        "triangle! triangle!"
        mha_inputs = {
            "q_x": x,
            "kv_x": x,
            "biases": biases,
        }

        return chunk_layer(
            partial(
                self.mha,
            ),
            mha_inputs,
            chunk_size=chunk_size,
            no_batch_dims=len(x.shape[:-2]),
            _out=x if inplace_safe else None,
        )

    def forward(self, act, mask):
        act = self.query_norm(act)
        bias = (1e9 * (mask - 1.0))[..., :, None, None, :]
        nonbatched_bias = self.feat_2d_weights(act)
        nonbatched_bias = nonbatched_bias.permute(0, 3, 1, 2)
        nonbatched_bias = nonbatched_bias.unsqueeze(-4)
        biases = [bias, nonbatched_bias]
        out = self._chunk(
            act,
            biases,
            4,
            inplace_safe=False,
        )

        return out


class TriangleAttentionEndingNode(nn.Module):
    def __init__(self, config, global_config):
        super().__init__()
        attention_num_c = config["attention_channel"]
        num_heads = config["num_head"]
        num_in_c = config["norm_channel"]

        self.attention_num_c = attention_num_c
        self.num_heads = num_heads

        self.query_norm = nn.LayerNorm(num_in_c)
        self.feat_2d_weights = nn.Linear(num_in_c, num_heads, bias=False)

        self.mha = Attention(num_in_c, num_in_c, num_in_c, attention_num_c, num_heads)

    @torch.jit.ignore
    def _chunk(
        self,
        x: torch.Tensor,
        biases: List[torch.Tensor],
        chunk_size: int,
        inplace_safe: bool = False,
    ) -> torch.Tensor:
        "triangle! triangle!"
        mha_inputs = {
            "q_x": x,
            "kv_x": x,
            "biases": biases,
        }

        return chunk_layer(
            partial(
                self.mha,
            ),
            mha_inputs,
            chunk_size=chunk_size,
            no_batch_dims=len(x.shape[:-2]),
            _out=x if inplace_safe else None,
        )

    def forward(self, act, mask):
        act = act.transpose(-2, -3)
        act = self.query_norm(act)
        mask = mask.transpose(-1, -2)
        bias = (1e9 * (mask - 1.0))[..., :, None, None, :]
        nonbatched_bias = self.feat_2d_weights(act)
        nonbatched_bias = nonbatched_bias.permute(0, 3, 1, 2)
        nonbatched_bias = nonbatched_bias.unsqueeze(-4)
        biases = [bias, nonbatched_bias]
        out = self._chunk(
            act,
            biases,
            4,
            inplace_safe=False,
        )
        out = out.transpose(-2, -3)

        return out


def create_extra_msa_feature(batch, num_extra_msa):
    extra_msa = batch["extra_msa"][:, :num_extra_msa]
    deletion_matrix = batch["extra_deletion_matrix"][:, :num_extra_msa]
    msa_1hot = torch.nn.functional.one_hot(extra_msa.long(), 23)
    has_deletion = torch.clip(deletion_matrix, 0.0, 1.0)[..., None]
    deletion_value = (torch.arctan(deletion_matrix / 3.0) * (2.0 / torch.pi))[..., None]
    extra_msa_mask = batch["extra_msa_mask"][:, :num_extra_msa]
    return torch.cat([msa_1hot, has_deletion, deletion_value], -1), extra_msa_mask


def create_msa_feat(batch):
    """Create and concatenate MSA features."""
    msa_1hot = torch.nn.functional.one_hot(batch["msa"].long(), 23)
    deletion_matrix = batch["deletion_matrix"]
    has_deletion = torch.clip(deletion_matrix, 0.0, 1.0)[..., None]
    deletion_value = (torch.arctan(deletion_matrix / 3.0) * (2.0 / torch.pi))[..., None]

    deletion_mean_value = (
        torch.arctan(batch["cluster_deletion_mean"] / 3.0) * (2.0 / torch.pi)
    )[..., None]

    msa_feat = [
        msa_1hot,
        has_deletion,
        deletion_value,
        batch["cluster_profile"],
        deletion_mean_value,
    ]

    return torch.cat(msa_feat, -1)


def make_msa_profile(batch):
    """Compute the MSA profile."""

    # Compute the profile for every residue (over all MSA sequences).
    return masked_mean(
        batch["msa_mask"][..., :, :, None],
        torch.nn.functional.one_hot(batch["msa"].long(), 22),
        dim=1,
    )


def nearest_neighbor_clusters(batch, gap_agreement_weight=0.0):
    """Assign each extra MSA sequence to its nearest neighbor in sampled MSA."""

    # Determine how much weight we assign to each agreement.  In theory, we could
    # use a full blosum matrix here, but right now let's just down-weight gap
    # agreement because it could be spurious.
    # Never put weight on agreeing on BERT mask.

    weights = torch.tensor(
        [1.0] * 21 + [gap_agreement_weight] + [0.0],
        dtype=torch.float32,
        device=batch["aatype"].device,
    )

    msa_mask = batch["msa_mask"]
    msa_one_hot = torch.nn.functional.one_hot(batch["msa"].long(), 23)

    extra_mask = batch["extra_msa_mask"]
    extra_one_hot = torch.nn.functional.one_hot(batch["extra_msa"].long(), 23)

    msa_one_hot_masked = msa_mask[..., None] * msa_one_hot
    extra_one_hot_masked = extra_mask[..., None] * extra_one_hot

    agreement = torch.einsum(
        "...mrc, ...nrc->...nm", extra_one_hot_masked, weights * msa_one_hot_masked
    )

    cluster_assignment = torch.nn.functional.softmax(1e3 * agreement, 1)
    cluster_assignment *= torch.einsum("...mr, ...nr->...mn", msa_mask, extra_mask)

    cluster_count = torch.sum(cluster_assignment, dim=-1)
    cluster_count += 1.0  # We always include the sequence itself.

    msa_sum = torch.einsum(
        "...nm, ...mrc->...nrc", cluster_assignment, extra_one_hot_masked
    )
    msa_sum += msa_one_hot_masked

    cluster_profile = msa_sum / cluster_count[..., None, None]

    extra_deletion_matrix = batch["extra_deletion_matrix"]
    deletion_matrix = batch["deletion_matrix"]

    del_sum = torch.einsum(
        "...nm, ...mc->...nc", cluster_assignment, extra_mask * extra_deletion_matrix
    )
    del_sum += deletion_matrix  # Original sequence.
    cluster_deletion_mean = del_sum / cluster_count[..., None]

    return cluster_profile, cluster_deletion_mean


def pseudo_beta_fn(aatype, all_atom_positions, all_atom_mask):
    """Create pseudo beta features."""
    is_gly = torch.eq(aatype, constants.restype_order["G"])
    ca_idx = constants.atom_order["CA"]
    cb_idx = constants.atom_order["CB"]
    pseudo_beta = torch.where(
        torch.tile(is_gly[..., None], [1] * len(is_gly.shape) + [3]),
        all_atom_positions[..., ca_idx, :],
        all_atom_positions[..., cb_idx, :],
    )

    if all_atom_mask is not None:
        pseudo_beta_mask = torch.where(
            is_gly, all_atom_mask[..., ca_idx], all_atom_mask[..., cb_idx]
        )
        return pseudo_beta, pseudo_beta_mask
    else:
        return pseudo_beta


def sample_msa(batch, max_seq, index=None, msa_indices_prefix=None):
    logits = (torch.clip(torch.sum(batch["msa_mask"], -1), 0.0, 1.0) - 1.0) * 1e6
    # The cluster_bias_mask can be used to preserve the first row (target
    # sequence) for each chain, for example.
    if "cluster_bias_mask" not in batch:
        cluster_bias_mask = nn.functional.pad(
            torch.zeros(batch["msa"].shape[1] - 1), (1, 0), "constant", 1.0
        )
    else:
        cluster_bias_mask = batch["cluster_bias_mask"]

    if msa_indices_prefix != None:
        with open(msa_indices_prefix + str(index) + ".txt", "r") as f:
            index_order = [int(l.strip()) for l in f.readlines()]
            index_order = torch.tensor(index_order, device=batch["msa"].device)
    else:
        rand_ind = torch.randperm(logits.shape[-1] - 1) + 1
        index_order = torch.cat((torch.tensor([0]), rand_ind))

    sel_idx = index_order[:max_seq]
    extra_idx = index_order[max_seq:]
    batch_sp = {k: v.clone() for k, v in batch.items()}
    for k in ["msa", "deletion_matrix", "msa_mask", "bert_mask"]:
        if k in batch_sp:
            batch_sp["extra_" + k] = batch_sp[k][:, extra_idx]
            batch_sp[k] = batch_sp[k][:, sel_idx]
    return batch_sp


"""
Test Multimer
"""


def compute_plddt(logits):
    """Computes per-residue pLDDT from logits.

    Args:
      logits: [num_res, num_bins] output from the PredictedLDDTHead.

    Returns:
      plddt: [num_res] per-residue pLDDT.
    """
    num_bins = logits.shape[-1]
    bin_width = 1.0 / num_bins
    bin_centers = torch.arange(0.5 * bin_width, 1.0, bin_width, device=logits.device)
    probs = torch.nn.functional.softmax(logits, dim=-1)
    predicted_lddt_ca = torch.sum(probs * bin_centers[None, ...], dim=-1)
    return predicted_lddt_ca * 100


def compute_predicted_aligned_error(
    logits: torch.Tensor,
    breaks: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    aligned_confidence_probs = torch.nn.functional.softmax(logits, dim=-1)
    (
        predicted_aligned_error,
        max_predicted_aligned_error,
    ) = _calculate_expected_aligned_error(
        alignment_confidence_breaks=breaks,
        aligned_distance_error_probs=aligned_confidence_probs,
    )

    return {
        "aligned_confidence_probs": aligned_confidence_probs,
        "predicted_aligned_error": predicted_aligned_error,
        "max_predicted_aligned_error": max_predicted_aligned_error,
    }


def get_confidence_metrics(prediction_result, multimer_mode: bool):
    confidence_metrics = {}
    confidence_metrics["plddt"] = compute_plddt(
        prediction_result["predicted_lddt"]["logits"]
    )
    if "predicted_aligned_error" in prediction_result:
        confidence_metrics.update(
            compute_predicted_aligned_error(
                logits=prediction_result["predicted_aligned_error"]["logits"],
                breaks=prediction_result["predicted_aligned_error"]["breaks"],
            )
        )
        confidence_metrics["ptm"] = predicted_tm_score(
            logits=prediction_result["predicted_aligned_error"]["logits"],
            breaks=prediction_result["predicted_aligned_error"]["breaks"],
            asym_id=None,
        )
        if multimer_mode:
            # Compute the ipTM only for the multimer model.
            confidence_metrics["iptm"] = predicted_tm_score(
                logits=prediction_result["predicted_aligned_error"]["logits"],
                breaks=prediction_result["predicted_aligned_error"]["breaks"],
                asym_id=prediction_result["predicted_aligned_error"]["asym_id"],
                interface=True,
            )
            confidence_metrics["ranking_confidence"] = (
                0.8 * confidence_metrics["iptm"] + 0.2 * confidence_metrics["ptm"]
            )

    if not multimer_mode:
        # Monomer models use mean pLDDT for model ranking.
        confidence_metrics["ranking_confidence"] = torch.mean(
            confidence_metrics["plddt"]
        )

    return confidence_metrics


def predicted_tm_score(
    logits: torch.Tensor,
    breaks: torch.Tensor,
    residue_weights: Optional[torch.Tensor] = None,
    asym_id: Optional[torch.Tensor] = None,
    interface: bool = False,
) -> torch.Tensor:
    if residue_weights is None:
        residue_weights = logits.new_ones(logits.shape[-2])
    bin_centers = _calculate_bin_centers(breaks)

    num_res = logits.shape[-2]
    clipped_num_res = max(num_res, 19)
    d0 = 1.24 * (clipped_num_res - 15) ** (1.0 / 3) - 1.8

    probs = torch.nn.functional.softmax(logits, dim=-1)

    tm_per_bin = 1.0 / (1 + (bin_centers**2) / (d0**2))
    predicted_tm_term = torch.sum(probs * tm_per_bin, dim=-1)

    pair_mask = torch.ones((num_res, num_res), dtype=bool, device=logits.device)
    if interface:
        pair_mask *= asym_id[..., None] != asym_id[None, ...]

    predicted_tm_term *= pair_mask
    pair_residue_weights = pair_mask * (
        residue_weights[None, :] * residue_weights[:, None]
    )
    normed_residue_mask = pair_residue_weights / (
        1e-8 + torch.sum(pair_residue_weights, dim=-1, keepdim=True)
    )
    per_alignment = torch.sum(predicted_tm_term * normed_residue_mask, dim=-1)
    weighted = per_alignment * residue_weights
    argmax = (weighted == torch.max(weighted)).nonzero()[0]
    return per_alignment[tuple(argmax)]


def protein_to_pdb(
    aatype,
    atom_positions,
    residue_index,
    chain_index,
    atom_mask,
    b_factors,
    out_mask=None,
    chain_ids_list=None,
):
    restypes = constants.restypes + ["X"]
    res_1to3 = lambda r: constants.restype_1to3.get(restypes[r], "UNK")
    atom_types = constants.atom_types

    pdb_lines = []
    residue_index = residue_index.astype(np.int32)
    chain_index = chain_index.astype(np.int32)
    chain_ids = {}
    for i in np.unique(chain_index):  # np.unique gives sorted output.
        if i >= constants.PDB_MAX_CHAINS:
            raise ValueError(
                f"The PDB format supports at most {constants.PDB_MAX_CHAINS} chains."
            )
        if chain_ids_list is not None:
            chain_ids[i] = chain_ids_list[i - 1]
        else:
            chain_ids[i] = constants.PDB_CHAIN_IDS[i]

    pdb_lines.append("MODEL     1")
    atom_index = 1
    last_chain_index = chain_index[0]
    for i in range(aatype.shape[0]):
        if out_mask is not None and out_mask[i] == 0:
            continue
        if last_chain_index != chain_index[i]:
            pdb_lines.append(
                _chain_end(
                    atom_index,
                    res_1to3(aatype[i - 1]),
                    chain_ids[chain_index[i - 1]],
                    residue_index[i - 1],
                )
            )
            last_chain_index = chain_index[i]
            atom_index += 1

        res_name_3 = res_1to3(aatype[i])
        for atom_name, pos, mask, b_factor in zip(
            atom_types, atom_positions[i], atom_mask[i], b_factors[i]
        ):
            if mask < 0.5:
                continue

            record_type = "ATOM"
            name = atom_name if len(atom_name) == 4 else f" {atom_name}"
            alt_loc = ""
            insertion_code = ""
            occupancy = 1.00
            element = atom_name[0]  # Protein supports only C, N, O, S, this works.
            charge = ""
            # PDB is a columnar format, every space matters here!
            atom_line = (
                f"{record_type:<6}{atom_index:>5} {name:<4}{alt_loc:>1}"
                f"{res_name_3:>3} {chain_ids[chain_index[i]]:>1}"
                f"{residue_index[i]:>4}{insertion_code:>1}   "
                f"{pos[0]:>8.3f}{pos[1]:>8.3f}{pos[2]:>8.3f}"
                f"{occupancy:>6.2f}{b_factor:>6.2f}          "
                f"{element:>2}{charge:>2}"
            )
            pdb_lines.append(atom_line)
            atom_index += 1
    pdb_lines.append(
        _chain_end(
            atom_index,
            res_1to3(aatype[-1]),
            chain_ids[chain_index[-1]],
            residue_index[-1],
        )
    )
    pdb_lines.append("ENDMDL")
    pdb_lines.append("END")

    # Pad all lines to 80 characters.
    pdb_lines = [line.ljust(80) for line in pdb_lines]
    return "\n".join(pdb_lines) + "\n"


def _calculate_bin_centers(boundaries: torch.Tensor):
    step = boundaries[1] - boundaries[0]
    bin_centers = boundaries + step / 2
    bin_centers = torch.cat(
        [bin_centers, (bin_centers[-1] + step).unsqueeze(-1)], dim=0
    )
    return bin_centers


def _calculate_expected_aligned_error(
    alignment_confidence_breaks: torch.Tensor,
    aligned_distance_error_probs: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    bin_centers = _calculate_bin_centers(alignment_confidence_breaks)
    return (
        torch.sum(aligned_distance_error_probs * bin_centers, dim=-1),
        bin_centers[-1],
    )


def _chain_end(atom_index, end_resname, chain_name, residue_index) -> str:
    chain_end = "TER"
    return (
        f"{chain_end:<6}{atom_index:>5}      {end_resname:>3} "
        f"{chain_name:>1}{residue_index:>4}"
    )


"""
All Atom Multimer
"""


def atom14_to_atom37(atom14_data, aatype):
    assert atom14_data.shape[1] == 14
    assert aatype.ndim == 1
    assert aatype.shape[0] == atom14_data.shape[0]

    residx_atom37_to_atom14 = torch.tensor(
        constants.restype_name_to_atom14_ids, device=aatype.device, dtype=aatype.dtype
    )[aatype]
    atom14_data_flat = atom14_data.reshape(*atom14_data.shape[:2], -1)
    # add 15th field used as placeholder in restype_name_to_atom14_ids
    atom14_data_flat = torch.cat(
        [atom14_data_flat, torch.zeros_like(atom14_data_flat[:, :1])], dim=1
    )
    out = torch.gather(
        atom14_data_flat,
        1,
        residx_atom37_to_atom14[..., None].repeat(1, 1, atom14_data_flat.shape[-1]),
    )
    return out.reshape(atom14_data.shape[0], 37, *atom14_data.shape[2:])


def atom_37_mask(aatype):
    restype_atom37_mask = torch.zeros(
        [21, 37], dtype=torch.float32, device=aatype.device
    )
    for restype, restype_letter in enumerate(constants.restypes):
        restype_name = constants.restype_1to3[restype_letter]
        atom_names = constants.residue_atoms[restype_name]
        for atom_name in atom_names:
            atom_type = constants.atom_order[atom_name]
            restype_atom37_mask[restype, atom_type] = 1

    residx_atom37_mask = restype_atom37_mask[aatype]
    return residx_atom37_mask


def compute_chi_angles(positions, mask, aatype):
    assert positions.shape[-2] == constants.atom_type_num
    assert mask.shape[-1] == constants.atom_type_num

    chi_atom_indices = torch.cat(
        (
            torch.tensor(get_chi_atom_indices(), device=positions.device),
            torch.zeros((1, 4, 4), device=positions.device, dtype=torch.int),
        ),
        dim=0,
    )

    atom_indices = chi_atom_indices[aatype.long(), ...]

    atom_indices_flattern = atom_indices.reshape(*atom_indices.shape[:-2], -1)
    positions_unbind = torch.unbind(positions, dim=-1)
    positions_gather = [
        torch.gather(x, -1, atom_indices_flattern) for x in positions_unbind
    ]
    chi_angle_atoms = [x.reshape(-1, 4, 4, 1) for x in positions_gather]
    chi_angle_atoms = torch.cat(chi_angle_atoms, dim=-1)
    a, b, c, d = [chi_angle_atoms[..., i, :] for i in range(4)]
    v1 = a - b
    v2 = b - c
    v3 = d - c

    c1 = torch.cross(v1, v2, dim=-1)
    c2 = torch.cross(v3, v2, dim=-1)
    c3 = torch.cross(c2, c1, dim=-1)

    v2_mag = torch.sqrt(torch.sum(v2**2, dim=-1))
    c3_v2 = torch.sum(c3 * v2, dim=-1)
    c1_c2 = torch.sum(c1 * c2, dim=-1)
    chi_angles = torch.atan2(c3_v2, v2_mag * c1_c2)

    chi_angles_mask = list(constants.chi_angles_mask)
    chi_angles_mask.append([0.0, 0.0, 0.0, 0.0])
    chi_angles_mask.append([0.0, 0.0, 0.0, 0.0])
    chi_angles_mask = torch.tensor(np.asarray(chi_angles_mask), device=positions.device)

    chi_mask = chi_angles_mask[aatype.long(), ...]
    chi_angle_atoms_mask = torch.gather(mask, -1, atom_indices_flattern)
    chi_angle_atoms_mask = chi_angle_atoms_mask.reshape(-1, 4, 4)
    chi_angle_atoms_mask = torch.prod(chi_angle_atoms_mask, -1)
    chi_mask = chi_mask * chi_angle_atoms_mask.type(positions.dtype)
    return chi_angles, chi_mask


def frames_and_literature_positions_to_atom14_pos(aatype, all_frames_to_global):
    residx_to_group_idx = torch.tensor(
        constants.restype_atom14_to_rigid_group,
        device=all_frames_to_global.get_rots().device,
        requires_grad=False,
    )
    group_mask = residx_to_group_idx[aatype.long(), ...]
    group_mask = torch.nn.functional.one_hot(group_mask, num_classes=8)
    map_atoms_to_global = all_frames_to_global[..., None, :] * group_mask

    map_atoms_to_global = map_atoms_to_global.map_tensor_fn(
        lambda x: torch.sum(x, dim=-1)
    )

    lit_positions = torch.tensor(
        constants.restype_atom14_rigid_group_positions,
        dtype=all_frames_to_global.get_rots().dtype,
        device=all_frames_to_global.get_rots().device,
        requires_grad=False,
    )
    lit_positions = lit_positions[aatype.long(), ...]

    mask = torch.tensor(
        constants.restype_atom14_mask,
        dtype=all_frames_to_global.get_rots().dtype,
        device=all_frames_to_global.get_rots().device,
        requires_grad=False,
    )
    mask = mask[aatype.long(), ...].unsqueeze(-1)
    pred_positions = map_atoms_to_global.apply(lit_positions)
    pred_positions = pred_positions * mask
    return pred_positions


def get_chi_atom_indices():
    """Returns atom indices needed to compute chi angles for all residue types.

    Returns:
      A tensor of shape [residue_types=21, chis=4, atoms=4]. The residue types are
      in the order specified in constants.restypes + unknown residue type
      at the end. For chi angles which are not defined on the residue, the
      positions indices are by default set to 0.
    """
    chi_atom_indices = []
    for residue_name in constants.restypes:
        residue_name = constants.restype_1to3[residue_name]
        residue_chi_angles = constants.chi_angles_atoms[residue_name]
        atom_indices = []
        for chi_angle in residue_chi_angles:
            atom_indices.append([constants.atom_order[atom] for atom in chi_angle])
        for _ in range(4 - len(atom_indices)):
            atom_indices.append([0, 0, 0, 0])  # For chi angles not defined on the AA.
        chi_atom_indices.append(atom_indices)

    chi_atom_indices.append([[0, 0, 0, 0]] * 4)  # For UNKNOWN residue.

    return np.asarray(chi_atom_indices)


def torsion_angles_to_frames(in_rigid, angle, aatype):
    m = torch.tensor(
        constants.restype_rigid_group_default_frame,
        dtype=angle.dtype,
        device=angle.device,
        requires_grad=False,
    )
    default_frames = m[aatype.long(), ...]
    default_rot = in_rigid.from_tensor_4x4(default_frames)
    backbone_rot = angle.new_zeros((*((1,) * len(angle.shape[:-1])), 2))
    backbone_rot[..., 1] = 1
    angle = torch.cat([backbone_rot.expand(*angle.shape[:-2], -1, -1), angle], dim=-2)
    all_rots = angle.new_zeros(default_rot.get_rots().get_rot_mats().shape)
    all_rots[..., 0, 0] = 1
    all_rots[..., 1, 1] = angle[..., 1]
    all_rots[..., 1, 2] = -angle[..., 0]
    all_rots[..., 2, 1:] = angle
    all_rots = rigid.Rigid(rigid.Rotation(rot_mats=all_rots), None)
    all_frames = default_rot.compose(all_rots)
    chi2_frame_to_frame = all_frames[..., 5]
    chi3_frame_to_frame = all_frames[..., 6]
    chi4_frame_to_frame = all_frames[..., 7]

    chi1_frame_to_bb = all_frames[..., 4]
    chi2_frame_to_bb = chi1_frame_to_bb.compose(chi2_frame_to_frame)
    chi3_frame_to_bb = chi2_frame_to_bb.compose(chi3_frame_to_frame)
    chi4_frame_to_bb = chi3_frame_to_bb.compose(chi4_frame_to_frame)

    all_frames_to_bb = rigid.Rigid.cat(
        [
            all_frames[..., :5],
            chi2_frame_to_bb.unsqueeze(-1),
            chi3_frame_to_bb.unsqueeze(-1),
            chi4_frame_to_bb.unsqueeze(-1),
        ],
        dim=-1,
    )

    all_frames_to_global = in_rigid[..., None].compose(all_frames_to_bb)
    return all_frames_to_global


"""
Structure Multimer
"""


class InvariantPointAttention(torch.nn.Module):
    def __init__(self, config, global_config):
        super().__init__()

        self.num_head = config["num_head"]
        self.num_scalar_qk = config["num_scalar_qk"]
        self.num_scalar_v = config["num_scalar_v"]
        self.num_point_qk = config["num_point_qk"]
        self.num_point_v = config["num_point_v"]

        self.num_output_c = config["num_channel"]
        self.rep_1d_num_c = config["num_channel"]
        self.rep_2d_num_c = global_config["pair_channel"]

        self.q = nn.Linear(
            self.rep_1d_num_c, self.num_scalar_qk * self.num_head, bias=False
        )
        self.k = nn.Linear(
            self.rep_1d_num_c, (self.num_scalar_qk) * self.num_head, bias=False
        )
        self.v = nn.Linear(
            self.rep_1d_num_c, (self.num_scalar_v) * self.num_head, bias=False
        )
        self.q_points = nn.Linear(
            self.rep_1d_num_c, self.num_point_qk * self.num_head * 3
        )
        self.k_points = nn.Linear(
            self.rep_1d_num_c, (self.num_point_qk) * self.num_head * 3
        )
        self.v_points = nn.Linear(
            self.rep_1d_num_c, (self.num_point_v) * self.num_head * 3
        )
        self.rr_kqv_2d = nn.Linear(self.rep_2d_num_c, self.num_head)
        self.final_r = nn.Linear(
            self.num_head
            * (self.rep_2d_num_c + self.num_scalar_v + 4 * self.num_point_v),
            self.num_output_c,
        )
        self.trainable_w = nn.Parameter(torch.zeros((self.num_head)))
        self.softplus = nn.Softplus()

    def forward(self, act, act_2d, sequence_mask, rigid):

        q_scalar = checkpoint(self.q, act)
        q_scalar = q_scalar.view(*q_scalar.shape[:-1], self.num_head, -1)
        k_scalar = checkpoint(self.k, act)
        k_scalar = k_scalar.view(*k_scalar.shape[:-1], self.num_head, -1)
        v_scalar = checkpoint(self.v, act)
        v_scalar = v_scalar.view(*v_scalar.shape[:-1], self.num_head, -1)

        q_point = checkpoint(self.q_points, act)
        q_point = q_point.view(*q_point.shape[:-1], self.num_head, -1)
        q_point = torch.split(q_point, q_point.shape[-1] // 3, dim=-1)
        q_point = torch.stack(q_point, dim=-1)
        q_point = q_point.view(*q_point.shape[:-3], -1, q_point.shape[-1])
        q_point_global = rigid[..., None].apply(q_point)

        k_point = checkpoint(self.k_points, act)
        k_point = k_point.view(*k_point.shape[:-1], self.num_head, -1)
        k_point = torch.split(k_point, k_point.shape[-1] // 3, dim=-1)
        k_point = torch.stack(k_point, dim=-1)
        k_point = k_point.view(*k_point.shape[:-3], -1, k_point.shape[-1])
        k_point_global = rigid[..., None].apply(k_point)

        v_point = checkpoint(self.v_points, act)
        v_point = v_point.view(*v_point.shape[:-1], self.num_head, -1)
        v_point = torch.split(v_point, v_point.shape[-1] // 3, dim=-1)
        v_point = torch.stack(v_point, dim=-1)
        v_point = v_point.view(*v_point.shape[:-3], -1, v_point.shape[-1])
        v_point_global = rigid[..., None].apply(v_point)

        attn_logits = 0.0
        num_point_qk = self.num_point_qk
        point_variance = max(num_point_qk, 1) * 9.0 / 2
        point_weights = math.sqrt(1.0 / point_variance)
        trainable_point_weights = self.softplus(self.trainable_w)
        point_weights = point_weights * torch.unsqueeze(trainable_point_weights, dim=-1)
        dist2 = torch.sum(
            torch.square(
                torch.unsqueeze(q_point_global, -3)
                - torch.unsqueeze(k_point_global, -4)
            ),
            -1,
        )
        dist2 = dist2.view(*dist2.shape[:-1], self.num_head, -1)
        attn_qk_point = -0.5 * torch.sum(
            point_weights[None, None, None, ...] * dist2, -1
        )
        attn_logits += attn_qk_point
        num_scalar_qk = self.num_scalar_qk
        scalar_variance = max(num_scalar_qk, 1) * 1.0
        scalar_weights = math.sqrt(1.0 / scalar_variance)
        q_scalar *= scalar_weights

        attn_logits += torch.einsum("...qhc,...khc->...qkh", q_scalar, k_scalar)

        attention_2d = checkpoint(self.rr_kqv_2d, act_2d)
        attn_logits += attention_2d

        mask_2d = sequence_mask * torch.transpose(sequence_mask, -1, -2)
        attn_logits -= 1e5 * (1.0 - mask_2d[..., None])
        attn_logits *= math.sqrt(1.0 / 3)
        attn = torch.softmax(attn_logits, -2)
        result_scalar = torch.einsum("...qkh, ...khc->...qhc", attn, v_scalar)
        v_point_global = v_point_global.view(
            *v_point_global.shape[:-2], self.num_head, -1, 3
        )
        result_point_global = torch.sum(
            attn[..., None, None] * v_point_global[:, None, ...], -4
        )
        output_features = []
        num_query_residues = act.shape[1]
        result_scalar = result_scalar.reshape(*result_scalar.shape[:-2], -1)
        output_features.append(result_scalar)
        result_point_global = result_point_global.view(
            *result_point_global.shape[:-3], -1, 3
        )
        result_point_local = rigid[..., None].invert_apply(result_point_global)
        result_point_local_x, result_point_local_y, result_point_local_z = torch.split(
            result_point_local, 1, dim=-1
        )

        output_features.extend(
            [
                torch.squeeze(result_point_local_x, -1),
                torch.squeeze(result_point_local_y, -1),
                torch.squeeze(result_point_local_z, -1),
            ]
        )
        point_norms = torch.linalg.norm(result_point_local, dim=(-1))
        output_features.append(point_norms)
        result_attention_over_2d = torch.einsum("...ijh, ...ijc->...ihc", attn, act_2d)
        result_attention_over_2d = result_attention_over_2d.reshape(
            *result_attention_over_2d.shape[:-2], -1
        )
        output_features.append(result_attention_over_2d)
        final_act = torch.cat(output_features, -1)

        out = self.final_r(final_act)

        return out


class PredictSidechains(torch.nn.Module):
    def __init__(self, config, global_config):
        super().__init__()
        num_in_c = config["num_channel"]
        num_c = config["sidechain"]["num_channel"]
        self.num_torsions = 7

        self.s_cur = nn.Linear(num_in_c, num_c)
        self.s_ini = nn.Linear(num_in_c, num_c)
        self.relu = nn.ReLU()

        self.res1 = nn.Sequential(
            nn.ReLU(), nn.Linear(num_c, num_c), nn.ReLU(), nn.Linear(num_c, num_c)
        )

        self.res2 = nn.Sequential(
            nn.ReLU(), nn.Linear(num_c, num_c), nn.ReLU(), nn.Linear(num_c, num_c)
        )

        self.final = nn.Sequential(nn.ReLU(), nn.Linear(num_c, self.num_torsions * 2))

    def forward(self, s_cur, s_ini):
        a = self.s_cur(self.relu(s_cur.clone())) + self.s_ini(self.relu(s_ini))
        a += self.res1(a.clone())
        a += self.res2(a.clone())
        unnormalized_angles = self.final(a).reshape(*a.shape[:-1], self.num_torsions, 2)
        norm = torch.sqrt(
            torch.clamp(
                torch.sum(unnormalized_angles**2, dim=-1, keepdim=True),
                min=1e-12,
            )
        )
        normalized_angles = unnormalized_angles / norm
        return normalized_angles, unnormalized_angles


class StructureModule(torch.nn.Module):
    def __init__(self, config, global_config):
        super().__init__()
        self.num_iter = config["num_layer"]
        num_1dc = config["num_channel"]

        self.StructureModuleIteration = StructureModuleIteration(config, global_config)
        self.layers = [self.StructureModuleIteration for _ in range(self.num_iter)]
        self.single_layer_norm = nn.LayerNorm(num_1dc)
        self.pair_layer_norm = nn.LayerNorm(global_config["pair_channel"])
        self.initial_projection = nn.Linear(num_1dc, num_1dc)
        self.config = config
        self.global_config = global_config

    def forward(self, single_representation, pair_representation, batch):
        sequence_mask = batch["seq_mask"][..., None]
        act = self.single_layer_norm(single_representation)
        initial_act = act.clone()
        act = self.initial_projection(act)
        act_2d = self.pair_layer_norm(pair_representation)
        rigids = rigid.Rigid.identity(
            act.shape[:-1], act.dtype, act.device, False, fmt="quat"
        )

        out = []
        for l in self.layers:
            l.to(act.device)
            act, rigids, sc = l(
                act, rigids, initial_act, act_2d, batch["aatype"], sequence_mask
            )
            out.append(sc)
        outputs = dict_multimap(torch.stack, out)
        outputs["act"] = act
        return outputs


class StructureModuleIteration(torch.nn.Module):
    def __init__(self, config, global_config):
        super().__init__()
        self.InvariantPointAttention = InvariantPointAttention(config, global_config)
        self.drop = nn.Dropout(0.1)
        self.rec_norm = nn.LayerNorm(config["num_channel"])
        self.rec_norm2 = nn.LayerNorm(config["num_channel"])
        self.position_scale = config["position_scale"]

        num_1dc = config["num_channel"]
        self.transition_r = nn.Sequential(
            nn.Linear(num_1dc, num_1dc),
            nn.ReLU(),
            nn.Linear(num_1dc, num_1dc),
            nn.ReLU(),
            nn.Linear(num_1dc, num_1dc),
        )

        self.backbone_update = nn.Linear(num_1dc, 6)
        self.PredictSidechains = PredictSidechains(config, global_config)

    def forward(self, act, in_rigid, initial_act, act_2d, aatype, sequence_mask):
        self.InvariantPointAttention.to(act.device)

        act = act.clone()
        rec_1d_update = self.InvariantPointAttention(
            act, act_2d, sequence_mask, in_rigid
        )
        act = act + rec_1d_update
        act = self.rec_norm(act)
        input_act = act.clone()
        act = self.transition_r(act)
        act = act + input_act
        act = self.rec_norm2(act)

        rigid_flat = self.backbone_update(act)
        rigids = in_rigid.compose_q_update_vec(rigid_flat)
        norm_sc, unnorm_sc = checkpoint(self.PredictSidechains, act, initial_act)

        bb_to_global = rigid.Rigid(
            rigid.Rotation(rot_mats=rigids.get_rots().get_rot_mats(), quats=None),
            rigids.get_trans(),
        )

        bb_to_global = bb_to_global.scale_translation(self.position_scale)
        all_frames_to_global = torsion_angles_to_frames(bb_to_global, norm_sc, aatype)
        pred_positions = frames_and_literature_positions_to_atom14_pos(
            aatype, all_frames_to_global
        )
        scaled_rigids = rigids.scale_translation(self.position_scale)
        sc = {
            "angles_sin_cos": norm_sc,
            "unnormalized_angles_sin_cos": unnorm_sc,
            "atom_pos": pred_positions,
            "sc_frames": all_frames_to_global.to_tensor_4x4(),
            "frames": scaled_rigids.to_tensor_7(),
        }

        rigids = rigids.stop_rot_gradient()

        return act, rigids, sc


def dict_multimap(fn, dicts):
    first = dicts[0]
    new_dict = {}
    for k, v in first.items():
        all_v = [d[k] for d in dicts]
        if type(v) is dict:
            new_dict[k] = dict_multimap(fn, all_v)
        else:
            new_dict[k] = fn(all_v)

    return new_dict
