import os
import sys

import paddle
import paddle.distributed as dist
from paddle.distributed import fleet
import paddle.nn.functional as F

paddle.empty([32, 1024, 1024, 1024], "uint8")
paddle.set_printoptions(linewidth=200)

from utils import initialize_fleet, configure_buffer, get_buffer

E = 16
H = 4096
SEQLEN = 16384
TOPK = 8

NUM_SMS = 48


def prepare_case_inputs(group):
    # E 是本地专家数, num_experts 是全局专家数
    num_experts = group.world_size * E

    x = paddle.randn([SEQLEN, H], "bfloat16")

    scores = paddle.randn([SEQLEN, num_experts])
    # 模拟给专家选择增加一定的系统不均衡
    scores += paddle.randn([num_experts]) * 0.1

    topk_weights, topk_idx = scores.topk(TOPK)
    topk_weights = F.sigmoid(topk_weights)
    topk_weights /= topk_weights.sum(axis=-1, keepdim=True)

    x.stop_gradient = False
    topk_weights.stop_gradient = False

    return x, topk_weights, topk_idx


def run_moe_forward(group, buffer, x, token_probs, token_indices):
    num_experts = group.world_size * E

    ################################# DISPATCH #################################
    (
        num_tokens_per_rank,
        num_tokens_per_rdma_rank,
        num_tokens_per_expert,
        is_token_in_rank,
        previous_event,
    ) = buffer.get_dispatch_layout(
        token_indices,
        num_experts,
        async_finish=False,
        allocate_on_comm_stream=False,
    )

    (
        recv_x,
        recv_token_indices,
        recv_token_probs,
        num_recv_tokens_per_expert_list,
        handle,
        event,
    ) = buffer.dispatch(
        x,
        topk_idx=token_indices,
        topk_weights=token_probs,
        num_tokens_per_rank=num_tokens_per_rank,
        num_tokens_per_rdma_rank=num_tokens_per_rdma_rank,
        is_token_in_rank=is_token_in_rank,
        num_tokens_per_expert=num_tokens_per_expert,
        async_finish=False,
        allocate_on_comm_stream=False,
    )

    tokens_per_expert = num_recv_tokens_per_expert_list
    recv_token_indices = recv_token_indices.cast("int32")

    ################################## UNZIP ###################################

    (
        unzipped_tokens,
        zipped_expertwise_rowmap,
        unzipped_probs,
        unzipped_scale,
    ) = paddle.nn.functional.moe_permute(
        recv_x,
        None,  # scale
        recv_token_indices,
        recv_token_probs,
        padding_alignment=128,
        num_experts=E,
        tokens_per_expert=tokens_per_expert,
    )

    ################################### FFN ####################################

    # 模拟进行一些非线性计算, 仅用于后面检查精度
    # Note: paddle 的 probs 是在 FFN 的激活函数阶段乘进去, 不是在下面的 zip, 下面 zip 传的
    # unzipped_probs 在前向无用, 是给反向预留的接口.

    expert_out = (unzipped_tokens.float() * unzipped_probs.unsqueeze(1)).erf()
    expert_out = expert_out.cast("bfloat16")

    ################################### ZIP ####################################

    zipped_tokens, zipped_probs = paddle.nn.functional.moe_unpermute(
        expert_out,
        zipped_expertwise_rowmap,
        recv_token_indices,
        unzipped_probs,
        total_zipped_tokens=len(recv_x),
        num_experts=E,
    )

    ################################# COMBINE ##################################

    combined_x, _, event = buffer.combine(
        zipped_tokens,
        handle,
        async_finish=False,
        allocate_on_comm_stream=False,
    )

    return combined_x


def main():
    group = initialize_fleet()
    configure_buffer(48)
    buffer = get_buffer(group, H * 2)
    x, token_probs, token_indices = prepare_case_inputs(group)

    ################################# RUN TEST #################################

    # warmup
    run_moe_forward(group, buffer, x, token_probs, token_indices)
    paddle.base.core.nvprof_start()

    for i in range(5):
        paddle.base.core.nvprof_nvtx_push(f"trial_{i}")
        combined_x = run_moe_forward(group, buffer, x, token_probs, token_indices)
        paddle.base.core.nvprof_nvtx_pop()

    paddle.device.synchronize()
    paddle.base.core.nvprof_stop()

    ################################# VALIDATE #################################
    expected_x = paddle.sum(
        (x.float().unsqueeze(1) * token_probs.unsqueeze(2)).erf(), axis=1
    )
    x_diff = paddle.abs(expected_x - combined_x.float())
    print("diff avg:", x_diff.mean().item(), "max:", x_diff.max().item())


if __name__ == "__main__":
    main()
