# Copyright 2023 Applied BioComputation Group, Stony Brook University
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

import os
import random
import warnings

import numpy as np
import torch

from src import config_multimer, constants, modules

warnings.filterwarnings("ignore")


class Model:
    """The MHC-Fine model for inference.

    Attributes:
        model: The pretrained model.
    """

    def __init__(self, model_path=constants.path_saved_state_dict):
        """Load the pretrained model for inference."""
        assert os.path.exists(
            model_path
        ), "Please download the model by following the instructions provided in the README."
        with torch.set_grad_enabled(False):
            self.model = self.get_model_from_dict(
                model_path, config_multimer.config_multimer
            ).to("cuda")

    def inference(self, np_sample, in_unique_id):
        """Inference on given model with fixed seed.

        Args:
            np_sample (dict): Features we preprocessed.
            in_unique_id (str): The name to save generated structure.

        Returns:
            Dictionary: inference metrics of plddt and masked plddt.
        """

        torch.manual_seed(constants.random_seed)
        random.seed(constants.random_seed)
        np.random.seed(constants.random_seed)

        os.makedirs(constants.DIR_OUTPUT, exist_ok=True)
        pdb_filename = os.path.join(constants.DIR_OUTPUT, in_unique_id + ".pdb")
        features = self.get_feature(np_sample)
        print("Running inference...")
        model_output, model_plddt = self.model(features)

        print("Writing predicted structure: ", pdb_filename)
        self.get_predicted_structure(features, model_output, pdb_filename)

        model_plddt = model_plddt.cpu().detach().numpy()[0]
        mask = features["loss_mask"].cpu().detach().numpy()[0]
        self.metrics = self.get_metric(model_plddt, mask)

        return self.metrics

    def get_feature(self, in_np_example):
        """Load the features to GPU. Note that the unsqueeze(0) is used to add extra dimension.

        Args:
            in_np_example (dict): Features we preprocessed.

        Returns:
            Dictionary: features and their corresponding values.
        """
        features = {
            k: torch.tensor(v).unsqueeze(0).to("cuda") for k, v in in_np_example.items()
        }
        return features

    def get_model_from_dict(self, in_path_saved_state_dict, global_config):
        """Load the model with given weights. Note that the model here is AF version 2.

        Args:
            in_path_saved_state_dict (str): Path of state dictionary file.
            global_config (dictionary): Model configuration.

        Returns:
            DockerIteration: Pretrained model.
        """
        model = modules.DockerIteration(global_config)
        model.load_state_dict(torch.load(in_path_saved_state_dict))
        return model

    def get_metric(self, plddt, mask):
        """Get metrics of the model.

        Args:
            plddt (ndarray): The ndarray for plddt score.
            mask (ndarray): The ndarray for mask.

        Returns:
            Dictionary: Path of saved metrics.
        """
        metrics = {
            "mean_plddt": float(plddt.mean()),
            "mean_masked_plddt": float((plddt * mask).sum() / mask.sum()),
        }
        return metrics

    def get_predicted_structure(self, features, output, pdb_filename):
        """Save predicted structure to pdb file.

        Args:
            features (dict): Dictionary of features.
            output (Tensor): Output tensor of the model.
            pdb_filename (str): Path to save pdb structure.
        """
        output["predicted_aligned_error"]["asym_id"] = features["asym_id"][0]
        confidences = modules.get_confidence_metrics(output, True)
        plddt = confidences["plddt"].detach().cpu().numpy()
        plddt_b_factors = np.repeat(plddt[..., None], constants.atom_type_num, axis=-1)
        mask = output["final_atom_mask"].cpu().numpy()
        pdb_out = modules.protein_to_pdb(
            features["aatype"][0].cpu().numpy(),
            output["final_all_atom"].detach().cpu().numpy(),
            features["residue_index"][0].cpu().numpy() + 1,
            features["asym_id"][0].cpu().numpy(),
            mask,
            plddt_b_factors[0],
            chain_ids_list=constants.temp_chains,
        )
        with open(pdb_filename, "w") as f:
            f.write(pdb_out)
        return
