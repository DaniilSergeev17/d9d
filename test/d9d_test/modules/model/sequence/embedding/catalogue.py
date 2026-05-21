import transformers as tr
from d9d.module.block.hidden_states_aggregator import HiddenStatesAggregationMode
from d9d.module.model.qwen3_dense import (
    Qwen3DenseForEmbedding,
    Qwen3DenseForEmbeddingParameters,
    mapper_from_huggingface_qwen3_dense_for_embedding,
    mapper_to_huggingface_qwen3_dense_for_embedding,
)
from d9d.module.model.qwen3_moe import (
    Qwen3MoEExpertsFormat,
    Qwen3MoEForEmbedding,
    Qwen3MoEForEmbeddingParameters,
    mapper_from_huggingface_qwen3_moe_for_embedding,
    mapper_to_huggingface_qwen3_moe_for_embedding,
)
from d9d.module.parallelism.model.qwen3_dense import parallelize_qwen3_dense_for_embedding
from d9d.module.parallelism.model.qwen3_moe import parallelize_qwen3_moe_for_embedding

from d9d_test.modules.model.sequence.catalogue import (
    D9D_MODEL_PARAMETERS,
    HF_MODEL_PARAMETERS,
    ModelCatalogue,
    d9d_model_factory,
    hf_model_factory,
)

HF_MODEL_FACTORY_EMBEDDING = {
    ModelCatalogue.QWEN3_MOE: hf_model_factory(
        tr.Qwen3MoeModel,
        config=HF_MODEL_PARAMETERS[ModelCatalogue.QWEN3_MOE],
        bf16_layers=["embed_tokens", "layers", "norm"],
    ),
    ModelCatalogue.QWEN3_DENSE: hf_model_factory(
        tr.Qwen3Model,
        config=HF_MODEL_PARAMETERS[ModelCatalogue.QWEN3_DENSE],
        bf16_layers=["embed_tokens", "layers", "norm"],
    ),
}


_D9D_PARAMS = {
    ModelCatalogue.QWEN3_MOE: Qwen3MoEForEmbeddingParameters(
        model=D9D_MODEL_PARAMETERS[ModelCatalogue.QWEN3_MOE],
        embedding_dim=None,
        normalize=False,
    ),
    ModelCatalogue.QWEN3_DENSE: Qwen3DenseForEmbeddingParameters(
        model=D9D_MODEL_PARAMETERS[ModelCatalogue.QWEN3_DENSE],
        embedding_dim=None,
        normalize=False,
    ),
}


D9D_MODEL_FACTORIES_EMBEDDING = {
    ModelCatalogue.QWEN3_MOE: [
        d9d_model_factory(
            Qwen3MoEForEmbedding,
            params=_D9D_PARAMS[ModelCatalogue.QWEN3_MOE],
            hidden_states_snapshot_mode=HiddenStatesAggregationMode.no,
            enable_checkpointing=enable_checkpointing,
        )
        for enable_checkpointing in (True, False)
    ],
    ModelCatalogue.QWEN3_DENSE: [
        d9d_model_factory(
            Qwen3DenseForEmbedding,
            params=_D9D_PARAMS[ModelCatalogue.QWEN3_DENSE],
            hidden_states_snapshot_mode=HiddenStatesAggregationMode.no,
            enable_checkpointing=enable_checkpointing,
        )
        for enable_checkpointing in (True, False)
    ],
}


HF_TO_D9D_MAPPER_EMBEDDING = {
    ModelCatalogue.QWEN3_MOE: mapper_from_huggingface_qwen3_moe_for_embedding(
        _D9D_PARAMS[ModelCatalogue.QWEN3_MOE],
        experts_format=Qwen3MoEExpertsFormat.FUSED,
    ),
    ModelCatalogue.QWEN3_DENSE: mapper_from_huggingface_qwen3_dense_for_embedding(
        _D9D_PARAMS[ModelCatalogue.QWEN3_DENSE]
    ),
}


D9D_TO_HF_MAPPER_EMBEDDING = {
    ModelCatalogue.QWEN3_MOE: mapper_to_huggingface_qwen3_moe_for_embedding(
        _D9D_PARAMS[ModelCatalogue.QWEN3_MOE],
        experts_format=Qwen3MoEExpertsFormat.FUSED,
    ),
    ModelCatalogue.QWEN3_DENSE: mapper_to_huggingface_qwen3_dense_for_embedding(
        _D9D_PARAMS[ModelCatalogue.QWEN3_DENSE]
    ),
}


D9D_PARALLELIZE_FN = {
    ModelCatalogue.QWEN3_MOE: parallelize_qwen3_moe_for_embedding,
    ModelCatalogue.QWEN3_DENSE: parallelize_qwen3_dense_for_embedding,
}
