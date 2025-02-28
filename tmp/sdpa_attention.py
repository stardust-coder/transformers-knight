from typing import Optional, Tuple
import torch

def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
###共通部分

new = True

if new == False:
    ##もともとのと比較する用
    def sdpa_attention_forward(
        inner_state: torch.Tensor, 
        inner_state_normalize: torch.Tensor, 
        module: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        dropout: float = 0.0,
        scaling: Optional[float] = None,
        is_causal: Optional[bool] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, None]:
        if hasattr(module, "num_key_value_groups"):
            key = repeat_kv(key, module.num_key_value_groups)
            value = repeat_kv(value, module.num_key_value_groups)

        causal_mask = attention_mask
        if attention_mask is not None:
            causal_mask = causal_mask[:, :, :, : key.shape[-2]]

        # SDPA with memory-efficient backend is bugged with non-contiguous itorchuts and custom attn_mask for some torch versions
        # Reference: https://github.com/pytorch/pytorch/issues/112577.
        query = query.contiguous()
        key = key.contiguous()
        value = value.contiguous()

        # We dispatch to SDPA's Flash Attention or Efficient kernels via this `is_causal` if statement instead of an inline conditional assignment
        # in SDPA to support both torch.compile's dynamic shapes and full graph options. An inline conditional prevents dynamic shapes from compiling.
        if is_causal is None:
            is_causal = causal_mask is None and query.shape[2] > 1

        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=causal_mask,
            dropout_p=dropout,
            scale=scaling,
            is_causal=is_causal,
        )
        attn_output = attn_output.transpose(1, 2).contiguous()

        import pdb; pdb.set_trace()
        return attn_output, None, None, None
        
else:
    ###自作
    def sdpa_attention_forward(
        inner_state: torch.Tensor, 
        inner_state_normalize: torch.Tensor, 
        module: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        dropout: float = 0.0,
        scaling: Optional[float] = None,
        is_causal: Optional[bool] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, None]:
        if hasattr(module, "num_key_value_groups"):
            key = repeat_kv(key, module.num_key_value_groups)
            value = repeat_kv(value, module.num_key_value_groups)

        causal_mask = attention_mask
        if attention_mask is not None:
            causal_mask = causal_mask[:, :, :, : key.shape[-2]]

        # SDPA with memory-efficient backend is bugged with non-contiguous itorchuts and custom attn_mask for some torch versions
        # Reference: https://github.com/pytorch/pytorch/issues/112577.
        query = query.contiguous()
        key = key.contiguous()
        value = value.contiguous()

        # We dispatch to SDPA's Flash Attention or Efficient kernels via this `is_causal` if statement instead of an inline conditional assignment
        # in SDPA to support both torch.compile's dynamic shapes and full graph options. An inline conditional prevents dynamic shapes from compiling.
        if is_causal is None:
            is_causal = causal_mask is None and query.shape[2] > 1

        # Efficient implementation equivalent to the following:
        # def scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=0.0,
        #         is_causal=False, scale=None, enable_gqa=False) -> torch.Tensor:

        #     L, S = query.size(-2), key.size(-2)
        #     scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
        #     attn_bias = torch.zeros(L, S, dtype=query.dtype)
        #     if is_causal:
        #         assert attn_mask is None
        #         temp_mask = torch.ones(L, S, dtype=torch.bool).tril(diagonal=0)
        #         attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
        #         attn_bias.to(query.dtype)

        #     if attn_mask is not None:
        #         if attn_mask.dtype == torch.bool:
        #             attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
        #         else:
        #             attn_bias += attn_mask

        #     if enable_gqa:
        #         key = key.repeat_interleave(query.size(-3)//key.size(-3), -3)
        #         value = value.repeat_interleave(query.size(-3)//value.size(-3), -3)

        #     attn_weight = query @ key.transpose(-2, -1) * scale_factor
        #     attn_weight += attn_bias.to("cuda:0")
        #     attn_weight = torch.softmax(attn_weight, dim=-1)
        #     attn_weight = torch.dropout(attn_weight, dropout_p, train=True)
            
        #     return attn_weight @ value

    
        def random_feature(x):
            T = x.shape[2]
            d = 128
            batch_size = x.shape[0]
            D = 100 # ランダム特徴量の次元
            omega = torch.randn(D, d).to("cuda").to(torch.bfloat16)  # ランダム特徴量の重み
            x_flat = x.view(-1, d).to("cuda").to(torch.bfloat16)  # # xを (batch_size * 32 * T, d) にリシェイプして omega と内積計算, (batch_size * 32 * T, d)

            rx = (# omega と x の内積計算: x と omega の形状を合わせる
                torch.exp(-torch.sum(x_flat * x_flat, dim=-1, keepdim=True) / 2) *
                torch.cat([torch.exp(torch.matmul(x_flat, omega.T)), torch.exp(-torch.matmul(x_flat, omega.T))], dim=-1) *
                torch.sqrt(torch.tensor(1 / (2 * D), dtype=torch.float32))
            )  # ランダム特徴量の計算
            rx = rx.view(batch_size, 32, T, 2 * D) # rxの形状を (batch_size, 32, T, 2 * D) に変換
            import pdb; pdb.set_trace()
            return rx

        print("Q shape:",query.shape) #torch.Size([1, 32, 11, 128])
        print("K shape:",key.shape)
        print("V shape :",value.shape)    
        inner_state = inner_state.to("cuda")
        print("inner state shape=", inner_state.shape)


        v = value[:,:,-2:-1,:].transpose(-1,-2)
        phi_k = random_feature(key[:,:,-2:-1,:])
        print(v.shape)
        print(phi_k.shape)
        increment = torch.matmul(v,phi_k)
        inner_state += increment #temporary
        print("increment shape:", increment.shape) #torch.Size([1, 32, 128, 200])

        increment_normalize = phi_k
        inner_state_normalize += increment_normalize #temporary
        print("increment normalization shape:", increment_normalize.shape) #torch.Size([1, 32, 1, 200])


        phi_q = random_feature(query).transpose(-1, -2)
        print("phi_q shape:", phi_q.shape) #torch.Size([1, 32, 200, 11])
        attn_output = torch.matmul(inner_state, phi_q)
        normalization = torch.matmul(inner_state_normalize, phi_q)
        # attn_output = attn_output/normalization
        print("normalization output shape:", normalization.shape)
        print("attn_output shape (before transpose):", attn_output.shape)
        attn_output = attn_output.transpose(-2, -1).contiguous()
        attn_output = attn_output.transpose(1, 2).contiguous()
        print("attn_output shape (after transpose):", attn_output.shape)  #torch.Size([1, 1, 32, T])となっておりおかしい　#originally torch.Size([1, 11, 32, 128])

        import pdb; pdb.set_trace()

        return attn_output, None, increment, increment_normalize

