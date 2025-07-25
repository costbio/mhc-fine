from abc import ABC, abstractmethod
import torch

class StrategyFactory(ABC):
    @abstractmethod
    def create_strategy(self, model):
        pass

class MsaRemovalStrategyFactory(StrategyFactory):
    def create_strategy(self, model):
        return MsaRemovalStrategy() if model.global_config['model']['msa_noising']['remove_msa'] else MsaNonRemovalStrategy()

class MsaRemovalStrategy:
    def get_preprocess_msa(self, model, batch, target_feat):
        return torch.zeros(1, model.global_config['model']['embeddings_and_evoformer']['num_msa'], batch['aatype'].shape[1],
            model.global_config['model']['embeddings_and_evoformer']['msa_channel'], dtype=target_feat.dtype, device=target_feat.device,
            requires_grad=target_feat.requires_grad)

    def get_extra_msa(self, model, batch, msa_activations):
        extra_msa_activations = torch.zeros(1, model.num_extra_msa - model.max_seq, batch['aatype'].shape[1],
            model.global_config['model']['embeddings_and_evoformer']['extra_msa_channel'], dtype=msa_activations.dtype,
            device=msa_activations.device, requires_grad=msa_activations.requires_grad)
        extra_msa_mask = torch.ones(1, model.num_extra_msa - model.max_seq, batch['aatype'].shape[1],
            dtype=msa_activations.dtype, device=msa_activations.device, requires_grad=msa_activations.requires_grad)
        msa_mask = torch.ones(1, model.global_config['model']['embeddings_and_evoformer']['num_msa'], batch['aatype'].shape[1],
            dtype=msa_activations.dtype, device=msa_activations.device, requires_grad=msa_activations.requires_grad)
        return extra_msa_activations, extra_msa_mask, msa_mask

class MsaNonRemovalStrategy:
    def get_preprocess_msa(self, model, batch, target_feat):
        return model.preprocess_msa(batch['msa_feat'])

    def get_extra_msa(self, model, batch, msa_activations):
        extra_msa_activations = model.extra_msa_activations(batch['extra_msa_feat'])
        extra_msa_mask = batch['extra_msa_mask'].type(torch.float32)
        msa_mask = batch['msa_mask']
        return extra_msa_activations, extra_msa_mask, msa_mask
