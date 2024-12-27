#!/usr/bin/env python

# reference: https://github.com/karpathy/nanoGPT/blob/master/transformer_sizing.ipynb

from collections import OrderedDict
from typing import Dict


def embedding_params(embed_dim, vocab_size, sequence_length=None, rope: bool = False) -> Dict:
    out = OrderedDict()

    out['embedding/position'] = 0 if rope else embed_dim * sequence_length
    out['embedding/token'] = embed_dim * vocab_size
    out['embedding'] = out['embedding/position'] + out['embedding/token']
    return out


def attn_params(embed_dim, num_heads, attn_type, groups=None) -> Dict:
    """
    Estimates the number of parameters in attenation blocks.
    If layernorm, we assume that bias is False for simplicity.
    If RMSNorm, there is no bias.
    """
    assert attn_type in ["mha", "mqa", "gqa"]

    out = OrderedDict()
    out['attention/norm'] = embed_dim  # suppose that bias is False if LayerNorm

    if attn_type == "mha":
        out['attention/kqv'] = embed_dim * (3 * embed_dim)
    else:
        if attn_type == "gqa":
            assert groups is not None, f"groups must be provided for gqa, got {groups}"
        else:
            groups = 1
        kv_dim = (embed_dim // num_heads) * groups
        out['attention/kqv'] = embed_dim * embed_dim + 2 * embed_dim * kv_dim

    out['attention/proj'] = embed_dim * embed_dim
    out['attention'] = out['attention/norm'] + out['attention/kqv'] + out['attention/proj']
    return out


def mlp_params(embed_dim, ffw_size=None) -> Dict:
    out = OrderedDict()

    if ffw_size is None:  # feed forward size
        ffw_size = 4 * embed_dim  # gpt, llama etc.
    out['mlp/norm'] = embed_dim
    out['mlp/ffw'] = embed_dim * ffw_size
    out['mlp/proj'] = ffw_size * embed_dim
    out['mlp/gate'] = embed_dim * ffw_size
    out['mlp'] = out['mlp/norm'] + out['mlp/ffw'] + out['mlp/proj'] + out['mlp/gate']
    return out


def params(
    num_layers: int,
    vocab_size: int,
    sequence_length: int,
    embed_dim: int,
    num_heads: int,
    attn_type: str = "mha",
    groups: int = None,
    ffw_size: int = None,
    tie_weight: bool = True,
    rope: bool = False,
):
    """Estimates the number of parameters in the model, bias is False for simplicity"""
    out = OrderedDict()

    embed_out = embedding_params(embed_dim, vocab_size, sequence_length, rope)
    out.update(**embed_out)

    attn_out = attn_params(embed_dim, num_heads, attn_type, groups)
    out.update(**attn_out)

    mlp_out = mlp_params(embed_dim, ffw_size)
    out.update(**mlp_out)

    # the transformer and the rest of it
    out['block'] = out['attention'] + out['mlp']
    out['transformer'] = num_layers * out['block']
    out['final_norm'] = embed_dim
    out['dense'] = 0 if tie_weight else embed_dim * vocab_size

    out['total'] = out['embedding'] + out['transformer'] + out['final_norm'] + out['dense']

    return out


def estimate_checkpoint_size(params_total: int, optimizer: str = 'adamw'):
    params_bytes = params_total * 4  # 4 bytes per float32

    scale = 2 if optimizer in ("adam", "adamw") else 1  # adam/adamw store 2nd order ema
    params_and_buffers_bytes = params_bytes + scale * params_bytes
    return params_and_buffers_bytes


def display_param_with_ratio(p):
    params_total = p['total']
    print(f"{'name':20s} {'params':10s} {'ratio (%)':10s}")
    for k, v in p.items():
        if "total" in k:
            print(f"{k:20s} {v:15,d} {v/params_total:10.2%}")
        else:
            print(f"{k:20s} {v:15d} {v/params_total:10.2%}")


def attn_flops(embed_dim, num_heads, sequence_length, attn_type="mha", groups=None):
    assert attn_type in ["mha", "mqa", "gqa"]

    out = OrderedDict()
    head_size = embed_dim // num_heads

    if attn_type == "mha":
        kv_dim = embed_dim
    else:
        if attn_type == "gqa":
            assert groups is not None, f"groups must be provided for gqa, got {groups}"
        else:
            groups = 1
        kv_dim = head_size * groups

    kqv_numel = embed_dim * embed_dim + 2 * embed_dim * kv_dim

    # 1) the projection to key, query, values
    out['attention/kqv'] = 2 * sequence_length * kqv_numel

    # 2) calculating the attention scores
    # (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
    out['attention/scores'] = 2 * sequence_length * sequence_length * embed_dim
    # 3) the reduction of the values
    # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
    out['attention/reduce'] = 2 * num_heads * (sequence_length * sequence_length * head_size)
    # 4) the final linear projection
    out['attention/proj'] = 2 * sequence_length * (embed_dim * embed_dim)
    out['attention'] = sum(out['attention/'+k] for k in ['kqv', 'scores', 'reduce', 'proj'])
    return out


def kv_cache_attn_flops(embed_dim, num_heads, sequence_length, attn_type="mha", groups=None):
    assert attn_type in ["mha", "mqa", "gqa"]

    out = OrderedDict()
    head_size = embed_dim // num_heads

    if attn_type == "mha":
        kv_dim = embed_dim
    else:
        if attn_type == "gqa":
            assert groups is not None, f"groups must be provided for gqa, got {groups}"
        else:
            groups = 1
        kv_dim = head_size * groups

    kqv_numel = embed_dim * embed_dim + 2 * embed_dim * kv_dim

    # 1) the projection to key, query, values
    # 1/seq of normal attn/kqv
    out['attention/kqv'] = 2 * 1 * kqv_numel

    # 2) calculating the attention scores
    # (B, nh, 1, hs) x (B, nh, hs, T) -> (B, nh, 1, T)
    # 1/seq of normal attn/scores
    out['attention/scores'] = 2 * 1 * sequence_length * embed_dim
    # 3) the reduction of the values
    # (B, nh, 1, T) x (B, nh, T, hs) -> (B, nh, 1, hs)
    # 1/seq of normal attn/reduce
    out['attention/reduce'] = 2 * num_heads * 1 * sequence_length * head_size
    # 4) the final linear projection
    # 1/seq of normal attn/proj
    out['attention/proj'] = 2 * 1 * (embed_dim * embed_dim)
    out['attention'] = sum(out['attention/'+k] for k in ['kqv', 'scores', 'reduce', 'proj'])
    return out


def mlp_flops(embed_dim, sequence_length, ffw_size=None):
    out = OrderedDict()

    if ffw_size is None:
        ffw_size = 4 * embed_dim  # feed forward size
    out['mlp/ffw1'] = 2 * sequence_length * (embed_dim * ffw_size)
    out['mlp/ffw2'] = 2 * sequence_length * (ffw_size * embed_dim)
    out['mlp'] = out['mlp/ffw1'] + out['mlp/ffw2']
    return out


def flops(
    sequence_length,
    vocab_size,
    num_layers,
    num_heads,
    embed_dim,
    ffw_size=None,
    attn_type="mha",
    groups=None,
):
    # for matmu A @ B, with shape (BxC) @ (CxD) -> (BxD), the flops value is 2*B*C*D
    out = OrderedDict()

    flops_out = attn_flops(
        embed_dim, num_heads, sequence_length,
        attn_type=attn_type, groups=groups,
    )
    out.update(**flops_out)

    flops_out = mlp_flops(embed_dim, sequence_length, ffw_size)
    out.update(**flops_out)

    # the transformer and the rest of it
    out['block'] = out['attention'] + out['mlp']
    out['transformer'] = num_layers * out['block']
    out['dense'] = 2 * sequence_length * (embed_dim * vocab_size)

    # forward,backward,total
    out['forward_total'] = out['transformer'] + out['dense']
    out['backward_total'] = 2 * out['forward_total']  # use common estimate of bwd = 2*fwd
    out['total'] = out['forward_total'] + out['backward_total']

    return out


def kv_cache_flops(
    sequence_length,
    vocab_size,
    num_layers,
    num_heads,
    embed_dim,
    ffw_size=None,
    attn_type="mha",
    groups=None,
):
    out = OrderedDict()

    flops_out = kv_cache_attn_flops(
        embed_dim, num_heads, sequence_length,
        attn_type=attn_type, groups=groups,
    )
    out.update(**flops_out)

    flops_out = mlp_flops(embed_dim, 1, ffw_size)
    out.update(**flops_out)

    # the transformer and the rest of it
    out['block'] = out['attention'] + out['mlp']
    out['transformer'] = num_layers * out['block']
    out['dense'] = 2 * 1 * (embed_dim * vocab_size)

    # forward, backward, total
    out['forward_total'] = out['transformer'] + out['dense']
    out['backward_total'] = 2 * out['forward_total']  # use common estimate of bwd = 2*fwd
    out['total'] = out['forward_total'] + out['backward_total']

    return out


def palm_flops(
    sequence_length, vocab_size,
    num_layers, num_heads, embed_dim,
    attn_type="mha", groups=None, ffw_size=None,
    tie_weight=True, rope=False,
):
    """estimate of the model flops following PaLM paper formula"""
    param_dict = params(
        num_layers=num_layers,
        vocab_size=vocab_size,
        sequence_length=sequence_length,
        embed_dim=embed_dim,
        num_heads=num_heads,
        attn_type=attn_type, groups=groups,
        ffw_size=ffw_size,
        tie_weight=tie_weight,
        rope=rope,
    )
    N = param_dict["total"] - param_dict["embedding"]
    if tie_weight:
        N += param_dict["dense"]

    L, H, Q, T = num_layers, num_heads, embed_dim//num_heads, sequence_length
    mf_per_token = 6*N + 12*L*H*Q*T
    mf = mf_per_token * sequence_length
    return mf, N


def simple_display(
    sequence_length,
    vocab_size,
    num_layers,
    num_heads,
    embed_dim,
    attn_type="mha",
    groups=None,
    tie_weight=True,
    ffw_size=None,
    rope=False,
):
    print("Params with ratio:")
    p = params(
        num_layers, vocab_size,
        sequence_length, embed_dim, num_heads,
        attn_type=attn_type, groups=groups,
        ffw_size=ffw_size,
        tie_weight=tie_weight,
        rope=rope,
    )
    display_param_with_ratio(p)
    print("--" * 30)

    # compare our param count to that reported by PyTorch
    print("Flops with ratio:")
    flops_details = flops(
        sequence_length, vocab_size,
        num_layers, num_heads,
        embed_dim, ffw_size=ffw_size,
        attn_type=attn_type, groups=groups,
    )
    flops_ffd = flops_details['forward_total']
    flops_total = flops_details['total']
    print(f"{'name':20s} {'flops':14s} {'ratio (%)':10s}")
    for k, v in flops_details.items():
        print(f"{k:20s} {v:14d} {v/flops_ffd*100:10.4f}")

    print("--" * 30)

    palm_flops_val, _ = palm_flops(
        sequence_length, vocab_size, num_layers, num_heads, embed_dim,
        ffw_size=ffw_size, attn_type=attn_type, groups=groups,
        tie_weight=tie_weight, rope=rope,
    )
    print(f"palm_flops: {palm_flops_val:,d}\nflops: {flops_total:,d}\nratio: {palm_flops_val/flops_total:.4f}")


def flops_params_diff(
    sequence_length: int,
    vocab_size: int,
    num_layers: int,
    num_heads: int,
    embed_dim: int,
    ffw_size: int,
    attn_type: str,
    groups: int,
    tie_weight: bool,
    rope: bool,
    up_keyname: str = "sequence_length",
):
    flops_kwargs = dict(
        sequence_length=sequence_length, vocab_size=vocab_size,
        num_layers=num_layers, num_heads=num_heads,
        embed_dim=embed_dim, ffw_size=ffw_size,
        attn_type=attn_type, groups=groups,
    )
    full_kwargs = dict(**flops_kwargs, tie_weight=tie_weight, rope=rope)
    assert up_keyname in full_kwargs, f"up_keyname {up_keyname} not in kwargs"

    base_value = full_kwargs[up_keyname]
    flops_baseline = flops(**flops_kwargs)["total"]
    params_baseline = params(**full_kwargs)["total"]

    print(f"Up-scaling {up_keyname} from {base_value}")
    for scale_factor in [0.5, 1, 2, 3, 4, 6, 8, 10, 16, 32, 64, 128]:
        up_scale_value = int(base_value * scale_factor)

        flops_kwargs[up_keyname] = up_scale_value
        flops_total = flops(**flops_kwargs)['total']
        flops_up_ratio = flops_total / flops_baseline

        full_kwargs[up_keyname] = up_scale_value
        p = params(**full_kwargs)["total"]
        param_up_ratio = p / params_baseline

        mf, N = palm_flops(**full_kwargs)
        seq_len = full_kwargs["sequence_length"]
        param_flops_ratio = 6 * N / (mf / seq_len)
        print(
            f"{scale_factor:7.1f}x: flops_up: {flops_up_ratio:8.2f}, "
            f"6N/flops: {param_flops_ratio:6.2%}, "
            f"param_up: {param_up_ratio:7.2f}"
        )


def qwen_configs() -> Dict:
    QWEN_BASE = dict(
        sequence_length=131072,
        vocab_size=152064,
        attn_type="gqa",
        rope=True,
        tie_weight=False,
    )
    CONFIGS = {
        # qwen2-7b config: https://huggingface.co/Qwen/Qwen2-7B/blob/main/config.json
        "QWEN2-7B": dict(
            num_layers=28, num_heads=28, embed_dim=3584, groups=4, ffw_size=18944, **QWEN_BASE
        ),
        # qwen2-72B config: https://huggingface.co/Qwen/Qwen2-72B/blob/main/config.json
        "QWEN2-72B": dict(
            num_layers=80, num_heads=80, embed_dim=8192, groups=8, ffw_size=29568, **QWEN_BASE
        ),
        # https://huggingface.co/Qwen/Qwen2.5-0.5B/blob/main/config.json
        "QWEN2.5-0.5B": dict(
            num_layers=24, num_heads=14, embed_dim=896, groups=2, ffw_size=4864,
            attn_type="gqa", rope=True,
            sequence_length=32768, vocab_size=151936, tie_weight=True,  # diff from base
        ),
        # qwen2.5 coder 1.5B config: https://huggingface.co/Qwen/Qwen2.5-Coder-1.5B/blob/main/config.json
        "QWEN2.5-1.5B": dict(
            num_layers=28, num_heads=12, embed_dim=1536, groups=2, ffw_size=8960,
            attn_type="gqa", rope=True,
            sequence_length=32768, vocab_size=151936, tie_weight=True,  # diff from base
        ),
        # https://huggingface.co/Qwen/Qwen2.5-3B/blob/main/config.json
        "QWEN2.5-3B": dict(
            num_layers=36, num_heads=16, embed_dim=2048, groups=4, ffw_size=11008,
            attn_type="gqa", rope=True,
            sequence_length=32768, vocab_size=151936, tie_weight=True,  # diff from base
        ),
        # https://huggingface.co/Qwen/Qwen2.5-14B/blob/main/config.json
        "QWEN2.5-14B": dict(
            num_layers=48, num_heads=40, embed_dim=5120, groups=8, ffw_size=13824, **QWEN_BASE
        ),
        # https://huggingface.co/Qwen/Qwen2.5-32B/blob/main/config.json
        "QWEN2.5-32B": dict(
            num_layers=64, num_heads=40, embed_dim=5120, groups=8, ffw_size=27648, **QWEN_BASE
        ),
    }
    return CONFIGS


def param_and_flops(model_name: str = "QWEN2.5-14B"):
    CONFIGS = qwen_configs()
    kwargs = CONFIGS[model_name]
    kwargs["sequence_length"] = 4 * 1024  # 4k sequence length

    base_value = {
        "sequence_length": 2 * 1024,
        "num_layers": kwargs["num_layers"],
        "num_heads": kwargs["num_heads"],
        "embed_dim": kwargs["embed_dim"],
        "groups": 1,
        "ffw_size": kwargs["ffw_size"],
    }
    print(f"Model: {model_name}")
    for arg_k, arg_v in kwargs.items():
        print(f"{arg_k}: {arg_v}")
    print("--" * 30)

    # simple_display(**kwargs)
    for k, v in base_value.items():
        update_kwargs = kwargs.copy()
        update_kwargs[k] = v
        flops_params_diff(up_keyname=k, **update_kwargs)
        print("--" * 30)


def prefill_decode(
    model_name: str = "QWEN2.5-14B",
    start_seq_len: int = 2048,
    max_gen_len: int = 32,
):
    CONFIGS = qwen_configs()
    kwargs = CONFIGS[model_name]
    for k in ["rope", "tie_weight"]:
        kwargs.pop(k)
    kwargs["sequence_length"] = start_seq_len

    print(f"Model: {model_name}")
    for arg_k, arg_v in kwargs.items():
        print(f"{arg_k}: {arg_v}")
    print("--" * 30)

    info_table = {}
    print("Prefill stage")
    prefill_flops = flops(**kwargs)["total"]
    info_table["prefill"] = prefill_flops

    print("Decode stage")
    for idx in range(1, max_gen_len + 1):
        kwargs["sequence_length"] += 1
        decode_flops = kv_cache_flops(**kwargs)["total"]
        info_table[f"decode_{idx}"] = decode_flops
        normal_flops = flops(**kwargs)["total"]
        # kv-cache flops * seq_len equasl normal flops
        assert decode_flops * kwargs["sequence_length"] == normal_flops

    for k, v in info_table.items():
        idx = -1 if "decode" not in k else int(k.split("_")[-1])
        if idx >= 2:
            prev_flops = info_table[f"decode_{idx-1}"]
            inc_flops = v - prev_flops
        else:
            inc_flops = "nan"
        print(f"{k:10s} {v:,d}  inc_flops: {inc_flops}")

    flops_sum = sum(v for v in info_table.values())
    direct_flops = flops(**kwargs)["total"]
    triangle_cnt = max_gen_len * (max_gen_len - 1) // 2  # 0 + 1 + 2 + ... + (max_gen_len - 1)
    saved_flops_cnt = triangle_cnt + start_seq_len * max_gen_len  # saved flops for each decode
    estimate_diff = inc_flops * saved_flops_cnt
    flops_diff = direct_flops - flops_sum
    flops_info = {
        "flops_sum": flops_sum,
        "direct_flops": direct_flops,
        "flops_diff": flops_diff,
        "estimate_diff": estimate_diff,
    }
    for k, v in flops_info.items():
        print(f"{k:15s} {v:,d}")


if __name__ == "__main__":
    import fire
    fire.Fire({
        "param_flops": param_and_flops,
        "pd": prefill_decode,
    })
