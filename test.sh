unset PADDLE_TRAINER_ENDPOINTS
unset DISTRIBUTED_TRAINER_ENDPOINTS
rm -rf log
export CUDA_VISIBLE_DEVICES="0,1,2,3"

export FLAGS_use_system_allocator=1
python -m paddle.distributed.launch \
            tests/test_hybrid_ep.py