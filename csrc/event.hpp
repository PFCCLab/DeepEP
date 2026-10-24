#include <ATen/cuda/CUDAContext.h>
#include <c10/core/Event.h>

#include <memory>

#include "kernels/exception.cuh"

namespace deep_ep {

struct EventHandle {
    std::shared_ptr<torch::Event> event;

    EventHandle() {
        event = std::make_shared<torch::Event>(torch::kCUDA);
        event->record(at::cuda::getCurrentCUDAStream().stream());
    }

    explicit EventHandle(const at::cuda::CUDAStream& stream) {
        event = std::make_shared<torch::Event>(torch::kCUDA);
        event->record(stream.stream());
    }

    EventHandle(const EventHandle& other) = default;

    void current_stream_wait() const {
        C10_CUDA_CHECK(cudaStreamWaitEvent(at::cuda::getCurrentCUDAStream().stream(), event->cuda_event()));
    }
};

torch::Event create_event(const at::cuda::CUDAStream& s) {
    auto event = torch::Event(torch::kCUDA);
    event.record(s.stream());
    return event;
}

void stream_wait(const at::cuda::CUDAStream& s_0, const at::cuda::CUDAStream& s_1) {
    EP_HOST_ASSERT(s_0.id() != s_1.id());
    auto ev = create_event(s_1);
    C10_CUDA_CHECK(cudaStreamWaitEvent(s_0.stream(), ev.cuda_event()));
}

void stream_wait(const at::cuda::CUDAStream& s, const EventHandle& event) {
    C10_CUDA_CHECK(cudaStreamWaitEvent(s.stream(), event.event->cuda_event()));
}

}  // namespace deep_ep
