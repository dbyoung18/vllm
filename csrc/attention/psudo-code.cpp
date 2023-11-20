
(num_seqs, num_heads)
template <typename scalar_t>
void paged_attention_kernel(
    scalar_t* out,              // [num_seqs, num_heads, head_size]
    scalar_t* query,            // [num_seqs, num_heads, head_size]
    scalar_t* key_cache,        // [num_blocks, block_size, num_kv_heads, head_size]
    scalar_t* value_cache,      // [num_blocks, block_size, num_kv_heads, head_size]
    int* metadata,              // [num_seq * (max_num_blocks_per_seq + 1) + num_heads * 2]
    float scale,
    int max_num_blocks_per_seq,
    int num_seqs,
    int num_heads,
    int head_size,
    int num_kv_heads,
    int block_size) {
        seq_idx = item_idx_x
        head_idx = item_idx_y
        q = load(query, (seq_idx, block_idx))                // [1, head_size]
        for i in range(max_num_blocks_per_seq) {
            block_idx = seq_idx * (max_num_blocks_per_seq + 1) + i
            seq_len_idx = seq_idx * (max_num_blocks_per_seq + 1) + max_num_blocks_per_seq;
            // alibi_slopes_idx = num_seq * (max_num_blocks_per_seq + 1) + head_idx * 2 + 1
            head_map_idx = num_seq * (max_num_blocks_per_seq + 1) + head_idx * 2
            head_map, alibi_slopes = load2(metadata + head_map_idx)
            k_cache_block = load(key_cache, (block_idx, head_map), seq_len)     // [block_size, head_size]
            v_cache_block = load(value_cache, (block_idx, head_map), seq_len)   // [block_size, head_size]
            qk = q@k_cache_block.transpose()                // [1, block_size]
            qk = reduce_alone_head(qk)
            qk *= scale
            qk = softmax(qk)
            qkv = qk@v                                      // [1, head_size]
            store(out, (seq_idx, head_idx), qkv)
        }
    }
