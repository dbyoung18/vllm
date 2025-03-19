# SPDX-License-Identifier: Apache-2.0

from typing import Optional

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)

try:
    import intel_extension_for_pytorch as ipex
except ImportError as e:
    logger.warning("Import error msg: %s", e.msg)


class ipex_ops:

    @staticmethod
    def _reshape_activation_tensor(
            x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        num = x.size(0)
        d = x.size(1) // 2
        x = x.reshape(num, 2, d)
        x1, x2 = torch.chunk(x, chunks=2, dim=1)
        x1 = x1.reshape(num, d)
        x2 = x2.reshape(num, d)
        return x1, x2

    @staticmethod
    def silu_and_mul(out: torch.Tensor, x: torch.Tensor) -> None:
        ipex.llm.functional.silu_and_mul(x, out)

    @staticmethod
    def gelu_and_mul(out: torch.Tensor, x: torch.Tensor) -> None:
        ipex.llm.functional.gelu_and_mul(x, out)

    @staticmethod
    def gelu_tanh_and_mul(out: torch.Tensor, x: torch.Tensor) -> None:
        ipex.llm.functional.gelu_and_mul(x, out)

    @staticmethod
    def gelu_fast(x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.gelu(x)

    @staticmethod
    def gelu_new(x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.gelu(x)

    @staticmethod
    def gelu_quick(out: torch.Tensor, x: torch.Tensor) -> None:
        ipex.llm.functional.gelu_quick(x, out)

    @staticmethod
    def paged_attention_v1(
        out: torch.Tensor,
        query: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        num_kv_heads: int,
        scale: float,
        block_tables: torch.Tensor,
        context_lens: torch.Tensor,
        block_size: int,
        max_context_len: int,
        alibi_slopes: Optional[torch.Tensor],
        kv_cache_dtype: str,
        k_scale: float,
        v_scale: float,
        tp_rank: int = 0,
        blocksparse_local_blocks: int = 0,
        blocksparse_vert_stride: int = 0,
        blocksparse_block_size: int = 64,
        blocksparse_head_sliding_step: int = 0,
    ) -> None:
        assert kv_cache_dtype == "auto"
        num_heads = out.size(1)
        num_queries_per_tokens = num_heads // num_kv_heads
        ipex.llm.modules.PagedAttention.single_query_kv_attention(
            out,
            query.contiguous(),
            key_cache.view_as(value_cache),
            value_cache,
            num_queries_per_tokens,
            scale,
            block_tables,
            context_lens,
            block_size,
            max_context_len,
            alibi_slopes,
        )

    @staticmethod
    def paged_attention_v2(
        out: torch.Tensor,
        exp_sum: torch.Tensor,
        max_logits: torch.Tensor,
        tmp_out: torch.Tensor,
        query: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        num_kv_heads: int,
        scale: float,
        block_tables: torch.Tensor,
        context_lens: torch.Tensor,
        block_size: int,
        max_context_len: int,
        alibi_slopes: Optional[torch.Tensor],
        kv_cache_dtype: str,
        k_scale: float,
        v_scale: float,
        tp_rank: int = 0,
        blocksparse_local_blocks: int = 0,
        blocksparse_vert_stride: int = 0,
        blocksparse_block_size: int = 64,
        blocksparse_head_sliding_step: int = 0,
    ) -> None:
        assert kv_cache_dtype == "auto"
        num_heads = out.size(1)
        num_queries_per_tokens = num_heads // num_kv_heads
        ipex.llm.modules.PagedAttention.single_query_kv_attention(
            out,
            query.contiguous(),
            key_cache.view_as(value_cache),
            value_cache,
            num_queries_per_tokens,
            scale,
            block_tables,
            context_lens,
            block_size,
            max_context_len,
            alibi_slopes,
        )

    @staticmethod
    def rotary_embedding(
        positions: torch.Tensor,  # [batch_size, seq_len]
        query: torch.Tensor,  # [batch_size, seq_len, num_heads*head_size]
        key: torch.Tensor,  # [batch_size, seq_len, num_kv_heads*head_size]
        head_size: int,
        cos_sin_cache: torch.Tensor,  # [cos_sin_dim, rot_dim]
        is_neox: bool,
    ) -> None:
        rot_dim = cos_sin_cache.size(1)
        ipex.llm.functional.rotary_embedding_batched(positions, query, key,
                                                     head_size, cos_sin_cache,
                                                     is_neox, rot_dim)

    @staticmethod
    def batched_rotary_embedding(positions: torch.Tensor, query: torch.Tensor,
                                 key: torch.Tensor, head_size: int,
                                 cos_sin_cache: torch.Tensor, is_neox: bool,
                                 rot_dim: int,
                                 cos_sin_cache_offsets: torch.Tensor) -> None:
        ipex.llm.functional.rotary_embedding_batched(positions, query, key,
                                                     head_size, cos_sin_cache,
                                                     is_neox, rot_dim,
                                                     cos_sin_cache_offsets)

    @staticmethod
    def rms_norm(input: torch.Tensor, weight: torch.Tensor,
                 epsilon: float) -> torch.Tensor:
        return ipex.llm.functional.rms_norm(input, weight, epsilon)

    @staticmethod
    def fused_add_rms_norm(input: torch.Tensor, residual: torch.Tensor,
                           weight: torch.Tensor, epsilon: float) -> None:
        tmp = ipex.llm.functional.add_rms_norm(residual, input, weight, None,
                                               epsilon, True)
        input.copy_(tmp)

    @staticmethod
    def varlen_attention(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        out: torch.Tensor,
        seqlen_q: torch.Tensor,
        seqlen_k: torch.Tensor,
        max_seqlen_q: int,
        max_seqlen_k: int,
        pdropout: float,
        softmax_scale: float,
        zero_tensors: bool,
        is_causal: bool,
        return_softmax: bool,
        gen_: torch.Generator,
        logits_soft_cap: float,
    ) -> None:
        ipex.llm.functional.varlen_attention(query.contiguous(),
                                             key.contiguous(),
                                             value.contiguous(), out,
                                             seqlen_q.int(), seqlen_k.int(),
                                             max_seqlen_q, max_seqlen_k,
                                             pdropout, softmax_scale,
                                             zero_tensors, is_causal,
                                             return_softmax, gen_,
                                             logits_soft_cap)

    @staticmethod
    def reshape_and_cache_flash(
        key: torch.Tensor,
        value: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
        kv_cache_dtype: str,
        k_scale: float,
        v_scale: float,
    ) -> None:
        assert kv_cache_dtype == "auto"
        ipex.llm.modules.PagedAttention.reshape_and_cache_flash(
            key, value, key_cache, value_cache, slot_mapping)

    @staticmethod
    def chunked_prefill(
        query: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        output: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        cu_seqlens_k: torch.Tensor,
        seq_used_k: Optional[torch.Tensor],
        block_table: torch.Tensor,
        alibi_slopes: Optional[torch.Tensor],
        max_seqlen_q: int,
        max_seqlen_k: int,
        p_dropout: float,
        softmax_scale: float,
        zero_tensors: bool,
        is_casual: bool,
        return_softmax: bool,
        gen_: Optional[torch.Generator],
    ):
        return ipex.llm.modules.PagedAttention.flash_attn_varlen_func(
            output,
            query.contiguous(),
            key_cache,
            value_cache,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            softmax_scale,
            is_casual,
            block_table,
            alibi_slopes,
            k_scale=1.0,
            v_scale=1.0,
        )

    @staticmethod
    def chunked_prefill_legacy(
            query: torch.Tensor,
            key_cache: torch.Tensor,
            value_cache: torch.Tensor,
            output: torch.Tensor,
            cu_seqlens_q: torch.Tensor,
            cu_seqlens_k: torch.Tensor,
            seq_used_k: Optional[torch.Tensor],
            block_table: torch.Tensor,
            alibi_slopes: Optional[torch.Tensor],
            max_seqlen_q: int,
            max_seqlen_k: int,
            p_dropout: float,
            softmax_scale: float,
            zero_tensors: bool,
            is_casual: bool,
            return_softmax: bool,
            gen_: Optional[torch.Generator],
    ):
         return torch.ops.torch_ipex.chunked_prefill(
             query.contiguous(),
             key_cache,
             value_cache,
             output,
             cu_seqlens_q,
             cu_seqlens_k,
             seq_used_k,
             block_table,
             alibi_slopes,
             max_seqlen_q,
             max_seqlen_k,
             p_dropout,
             softmax_scale,
             zero_tensors,
             -1,
             -1,
             is_casual,
             return_softmax,
             gen_,
             -1.0
         )

    @staticmethod
    def copy_blocks(key_caches: list[torch.Tensor],
                    value_caches: list[torch.Tensor],
                    block_mapping: torch.Tensor) -> None:
        torch.xpu.copy_blocks(  # type: ignore
            key_caches,
            value_caches,
            block_mapping,
        )

    @staticmethod
    def swap_blocks(src: torch.Tensor, dst: torch.Tensor,
                    block_mapping: torch.Tensor) -> None:
        torch.xpu.swap_blocks(src, dst, block_mapping)  # type: ignore

    @staticmethod
    def bgmv_shrink(inputs: torch.Tensor,
                    lora_a_weights: torch.Tensor,
                    output_tensor: torch.Tensor,
                    lora_indices_tensor: torch.Tensor,
                    scaling: float = 1.0) -> None:
        ipex.llm.functional.bgmv_shrink(inputs, lora_a_weights, output_tensor,
                                        lora_indices_tensor, scaling)

    @staticmethod
    def bgmv_expand(inputs: torch.Tensor,
                    lora_b_weights: torch.Tensor,
                    output_tensor: torch.Tensor,
                    lora_indices_tensor: torch.Tensor,
                    add_inputs: bool = True) -> None:
        ipex.llm.functional.bgmv_expand(inputs, lora_b_weights, output_tensor,
                                        lora_indices_tensor, add_inputs)

    @staticmethod
    def bgmv_expand_slice(inputs: torch.Tensor,
                          lora_b_weights: torch.Tensor,
                          output_tensor: torch.Tensor,
                          lora_indices_tensor: torch.Tensor,
                          slice_offset: int,
                          slice_size: int,
                          add_inputs: bool = True) -> None:
        ipex.llm.functional.bgmv_expand_slice(inputs, lora_b_weights,
                                              output_tensor,
                                              lora_indices_tensor,
                                              slice_offset, slice_size,
                                              add_inputs)

    @staticmethod
    def sgmv_shrink(inputs: torch.Tensor,
                    lora_a_weights: torch.Tensor,
                    output_tensor: torch.Tensor,
                    b_seq_start_loc: torch.Tensor,
                    seq_len_tensor: torch.Tensor,
                    lora_indices_tensor: torch.Tensor,
                    batches: int,
                    max_seq_length: int,
                    token_nums: int,
                    scaling: float = 1.0) -> None:
        assert inputs.size(0) == token_nums
        ipex.llm.functional.sgmv_shrink(inputs, lora_a_weights, output_tensor,
                                        b_seq_start_loc, seq_len_tensor,
                                        lora_indices_tensor, batches,
                                        max_seq_length, scaling)

    @staticmethod
    def sgmv_expand(inputs: torch.Tensor,
                    lora_b_weights: torch.Tensor,
                    output_tensor: torch.Tensor,
                    b_seq_start_loc: torch.Tensor,
                    seq_len_tensor: torch.Tensor,
                    lora_indices_tensor: torch.Tensor,
                    batches: int,
                    max_seq_length: int,
                    token_nums: int,
                    add_inputs: bool = False) -> None:
        assert inputs.size(0) == token_nums
        ipex.llm.functional.sgmv_expand(inputs, lora_b_weights, output_tensor,
                                        b_seq_start_loc, seq_len_tensor,
                                        lora_indices_tensor, batches,
                                        max_seq_length, add_inputs)

    @staticmethod
    def sgmv_expand_slice(inputs: torch.Tensor,
                          lora_b_weights: torch.Tensor,
                          output_tensor: torch.Tensor,
                          b_seq_start_loc: torch.Tensor,
                          seq_len_tensor: torch.Tensor,
                          lora_indices_tensor: torch.Tensor,
                          batches: int,
                          max_seq_length: int,
                          token_nums: int,
                          slice_offset: int,
                          slice_size: int,
                          add_inputs: bool = False) -> None:
        assert inputs.size(0) == token_nums
        ipex.llm.functional.sgmv_expand_slice(inputs, lora_b_weights,
                                              output_tensor, b_seq_start_loc,
                                              seq_len_tensor,
                                              lora_indices_tensor, batches,
                                              max_seq_length, slice_offset,
                                              slice_size, add_inputs)
