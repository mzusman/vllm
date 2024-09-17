from typing import Optional

import pytest
import torch
import torch.nn.functional as F
from einops import rearrange

from vllm.model_executor.layers.mamba.ops.causal_conv1d import (
    causal_conv1d_fn, causal_conv1d_update)


def causal_conv1d_ref(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    initial_states: Optional[torch.Tensor] = None,
    return_final_states: bool = False,
    final_states_out: Optional[torch.Tensor] = None,
    activation: Optional[str] = "silu",
):
    """
    x: (batch, dim, seqlen)
    weight: (dim, width)
    bias: (dim,)
    initial_states: (batch, dim, width - 1)
    final_states_out: (batch, dim, width - 1)

    out: (batch, dim, seqlen)
    """
    if activation not in [None, "silu", "swish"]:
        raise NotImplementedError("activation must be None, silu, or swish")
    dtype_in = x.dtype
    x = x.to(weight.dtype)
    seqlen = x.shape[-1]
    dim, width = weight.shape
    if initial_states is None:
        out = F.conv1d(x,
                       weight.unsqueeze(1),
                       bias,
                       padding=width - 1,
                       groups=dim)
    else:
        x = torch.cat([initial_states, x], dim=-1)
        out = F.conv1d(x, weight.unsqueeze(1), bias, padding=0, groups=dim)
    out = out[..., :seqlen]
    if return_final_states:
        final_states = F.pad(x, (width - 1 - x.shape[-1], 0)).to(
            dtype_in)  # (batch, dim, width - 1)
        if final_states_out is not None:
            final_states_out.copy_(final_states)
        else:
            final_states_out = final_states
    out = (out if activation is None else F.silu(out)).to(dtype=dtype_in)
    return (out, None) if not return_final_states else (out, final_states_out)


def causal_conv1d_update_ref(x,
                             conv_state,
                             weight,
                             bias=None,
                             activation=None,
                             cache_seqlens=None):
    """
    x: (batch, dim) or (batch, dim, seqlen)
    conv_state: (batch, dim, state_len), where state_len >= width - 1
    weight: (dim, width)
    bias: (dim,)
    cache_seqlens: (batch,), dtype int32.
        If not None, the conv_state is treated as a circular buffer.
        The conv_state will be updated by copying x to the 
        conv_state starting at the index
        @cache_seqlens % state_len before performing the convolution.

    out: (batch, dim) or (batch, dim, seqlen)
    """
    if activation not in [None, "silu", "swish"]:
        raise NotImplementedError("activation must be None, silu, or swish")
    dtype_in = x.dtype
    unsqueeze = x.dim() == 2
    if unsqueeze:
        x = x.unsqueeze(-1)
    batch, dim, seqlen = x.shape
    width = weight.shape[1]
    state_len = conv_state.shape[-1]
    assert conv_state.shape == (batch, dim, state_len)
    assert weight.shape == (dim, width)
    if cache_seqlens is None:
        x_new = torch.cat([conv_state, x], dim=-1).to(
            weight.dtype)  # (batch, dim, state_len + seqlen)
        conv_state.copy_(x_new[:, :, -state_len:])
    else:
        width_idx = torch.arange(
            -(width - 1), 0, dtype=torch.long,
            device=x.device).unsqueeze(0) + cache_seqlens.unsqueeze(1)
        width_idx = torch.remainder(width_idx, state_len).unsqueeze(1).expand(
            -1, dim, -1)
        x_new = torch.cat([conv_state.gather(2, width_idx), x],
                          dim=-1).to(weight.dtype)
        copy_idx = torch.arange(
            seqlen, dtype=torch.long,
            device=x.device).unsqueeze(0) + cache_seqlens.unsqueeze(1)
        copy_idx = torch.remainder(copy_idx,
                                   state_len).unsqueeze(1).expand(-1, dim, -1)
        conv_state.scatter_(2, copy_idx, x)
    out = F.conv1d(x_new, weight.unsqueeze(1), bias, padding=0,
                   groups=dim)[:, :, -seqlen:]
    if unsqueeze:
        out = out.squeeze(-1)
    return (out if activation is None else F.silu(out)).to(dtype=dtype_in)


@pytest.mark.parametrize("itype", [torch.bfloat16, torch.float])
@pytest.mark.parametrize("silu_activation", [True])
@pytest.mark.parametrize("has_bias", [True])
@pytest.mark.parametrize("width", [4])
@pytest.mark.parametrize('seqlen',
    [1,8, 16, 32, 64, 128, 256, 512, 784, 1024, 2048, 4096])
@pytest.mark.parametrize('dim', [64])
@pytest.mark.parametrize('batch', [1])
def test_causal_conv1d(batch, dim, seqlen, width, has_bias, silu_activation,
                       itype):
    device = "cuda"
    rtol, atol = (3e-4, 1e-3) if itype == torch.float32 else (3e-3, 5e-3)
    if itype == torch.bfloat16:
        rtol, atol = 1e-2, 5e-2
    # set seed
    torch.random.manual_seed(0)
    x = torch.randn(batch,
                    dim,
                    seqlen,
                    device=device,
                    dtype=itype).contiguous()

    weight = torch.randn(dim, width, device=device, dtype=itype)
    bias = torch.randn(dim, device=device, dtype=itype) if has_bias else None
    initial_states = torch.randn(
        batch,
        dim,
        width - 1,
        device=device,
        dtype=itype
    )
    x_ref = x.clone()
    weight_ref = weight.clone()
    bias_ref = bias.clone() if bias is not None else None
    initial_states_ref = initial_states.clone(
    ) if initial_states is not None else None
    activation = None if not silu_activation else "silu"
    out, final_states = causal_conv1d_fn(
        x,
        weight,
        bias,
        activation=activation,
        conv_states=initial_states,
        has_initial_state=torch.ones(batch,dtype=torch.int32,device=x.device)
    )
    out_ref, final_states_ref = causal_conv1d_ref(
        x_ref,
        weight_ref,
        bias_ref,
        initial_states=initial_states_ref,
        return_final_states=True,
        activation=activation)
    assert final_states is not None and final_states_ref is not None
    assert torch.allclose(final_states,
                          final_states_ref,
                          rtol=rtol,
                          atol=atol)

    assert torch.allclose(out, out_ref, rtol=rtol, atol=atol)



@pytest.mark.parametrize("itype", [torch.bfloat16])
@pytest.mark.parametrize("silu_activation", [False, True])
@pytest.mark.parametrize("has_bias", [False, True])
@pytest.mark.parametrize("has_cache_seqlens", [False, True])
@pytest.mark.parametrize("seqlen", [1, 4, 5])
@pytest.mark.parametrize("width", [2, 3, 4])
@pytest.mark.parametrize("dim", [2048, 2048 + 16, 4096])
def test_causal_conv1d_update(dim, width, seqlen, has_cache_seqlens, has_bias,
                              silu_activation, itype):
    device = "cuda"
    rtol, atol = (3e-4, 1e-3) if itype == torch.float32 else (3e-3, 5e-3)
    if itype == torch.bfloat16:
        rtol, atol = 1e-2, 5e-2
    # set seed
    torch.random.manual_seed(0)
    batch = 64
    # batch = 1
    # dim = 64
    x = torch.randn(batch, seqlen, dim, device=device,
                    dtype=itype).transpose(-1, -2)
    state_len = torch.randint(width - 1, width + 10, (1, )).item()
    conv_state = torch.randn(batch, state_len, dim, device=device,
                             dtype=itype).transpose(-1, -2)
    weight = torch.randn(dim,
                         width,
                         device=device,
                         dtype=itype,
                         requires_grad=True)
    if has_bias:
        bias = torch.randn(dim, device=device, dtype=itype, requires_grad=True)
    else:
        bias = None
    conv_state_ref = conv_state.detach().clone()
    activation = None if not silu_activation else "silu"
    cache_seqlens = (torch.randint(
        0, 1024, (batch, ), dtype=torch.int32, device=device)
                     if has_cache_seqlens else None)
    out = causal_conv1d_update(x,
                               conv_state,
                               weight,
                               bias,
                               activation=activation,
                               cache_seqlens=cache_seqlens)
    out_ref = causal_conv1d_update_ref(x,
                                       conv_state_ref,
                                       weight,
                                       bias,
                                       activation=activation,
                                       cache_seqlens=cache_seqlens)

    assert torch.equal(conv_state, conv_state_ref)
    assert torch.allclose(out, out_ref, rtol=rtol, atol=atol)


@pytest.mark.parametrize("itype", [torch.bfloat16])
@pytest.mark.parametrize("silu_activation", [True])
@pytest.mark.parametrize("has_bias", [True])
@pytest.mark.parametrize("width", [4])
@pytest.mark.parametrize(
    'seqlen',
    [8, 16, 32, 64, 128, 256, 512, 784, 1024, 2048, 4096]
)
@pytest.mark.parametrize('dim', [64 ,4096])
def test_causal_conv1d_varlen(dim, seqlen, width, has_bias, silu_activation, itype):
    device = "cuda"
    rtol, atol = (3e-4, 1e-3) if itype == torch.float32 else (3e-3, 5e-3)
    if itype == torch.bfloat16:
        rtol, atol = 1e-2, 5e-2
    # set seed
    torch.random.manual_seed(seqlen + dim + width)
    batch = 1
    seqlens = []
    nsplits = 3
    eos_pos = torch.randperm(seqlen - 1)[:nsplits].sort().values
    seqlens.append(torch.diff(torch.cat([torch.tensor(
        [-1]
    ), eos_pos, torch.tensor([seqlen - 1])])).tolist())
    assert sum(seqlens[-1]) == seqlen
    assert all(s > 0 for s in seqlens[-1])

    cumsum = torch.cumsum(torch.tensor(seqlens[0]),dim=0).to(torch.int32)
    x = torch.randn(
        batch,
        4096 + dim + 64,
        seqlen,
        device=device,
        dtype=itype
    )[:, 4096:4096 + dim, :]
    weight = torch.randn(dim, width, device=device, dtype=itype)
    if has_bias:
        bias = torch.randn(dim, device=device, dtype=itype)
    else:
        bias = None
    x_ref = x.clone()
    weight_ref = weight.clone()
    bias_ref = bias.clone() if bias is not None else None
    activation = None if not silu_activation else "silu"
    final_states = torch.randn(nsplits + 1, dim, width - 1,
                                           device=x.device,
                                           dtype=x.dtype)
    final_states_ref = final_states.clone()
    has_initial_states = torch.ones_like(cumsum,dtype=torch.int32,device=x.device)
    cache_indices = torch.arange(cumsum.shape[0],dtype=torch.int32,device=x.device)

    out,final_states = causal_conv1d_fn(x.squeeze(0), weight, bias, cumsum.cuda(),
                                cache_indices,
                                has_initial_states,
                                final_states,
                                activation)
    out_ref = []
    out_ref_b = []
    for i, x_s in enumerate(torch.split(x_ref[[0]], seqlens[0], dim=2)):
        out_ref_b.append(causal_conv1d_ref(
            x_s,
            weight_ref,
            bias_ref,
            activation=activation,
            return_final_states=True,
            initial_states=final_states_ref[i].unsqueeze(0)
        ))
    out_ref.append(torch.cat([t[0] for t in out_ref_b], dim=2))
    out_ref = torch.cat(out_ref, dim=0)

    ref_final_states = torch.concat([t[1] for t in out_ref_b],dim=0)
    print(f"Output max diff: {(out - out_ref).abs().max().item()}")
    print(f"Output mean diff: {(out - out_ref).abs().mean().item()}")
    print(f"Output state max diff:{(final_states - ref_final_states).abs().max().item()}")
    print(f"Output state mean diff:{(final_states - ref_final_states).abs().mean().item()}")
    assert torch.allclose(out, out_ref, rtol=rtol, atol=atol)
    assert torch.allclose(final_states, ref_final_states, rtol=rtol, atol=atol)

