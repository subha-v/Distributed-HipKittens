// ================================================================================================
// dhk_rt — runtime glue for validating the MI300X/gfx942 GEMM-RS megakernel.
//
// The kernel translation unit deliberately contains no allocation, no IPC, and no
// descriptor plumbing (MI300X_DESIGN.md section 3: "those are the future runtime binding's
// responsibilities"). This module is that binding, in the smallest form that exercises the
// PRODUCTION host ABI rather than a re-implementation of it:
//
//   * one fine-grained symmetric heap per device, at identical offsets on every rank, so the
//     eight heap bases form a genuine symmetric allocation;
//   * hk_gemm_rs::host_abi::snapshot_allocation_descriptors called unmodified to rebase the
//     c_heap and signal anchors into the two 72-byte device PODs the kernel expects;
//   * hk_gemm_rs_mi300x::host_abi::resolve_shape called unmodified, so every derived count
//     the kernel receives is the ABI's own and never hand-computed in Python.
//
// World-8 is realized as one process driving eight devices with peer access enabled. That is
// address-space equivalent to IRIS's IPC heap for this kernel's purposes: the descriptor holds
// each rank's own base explicitly, and translate_peer only ever computes
// peer_base + (local_ptr - local_base).
// ================================================================================================

#include <hip/hip_runtime.h>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <array>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include "gemm_rs_mi300x_host_abi.hpp"

namespace py = pybind11;
namespace m3  = hk_gemm_rs_mi300x;
namespace m3h = hk_gemm_rs_mi300x::host_abi;

namespace {

void hip_ok(hipError_t status, const char* what) {
    if (status != hipSuccess) {
        throw std::runtime_error(std::string(what) + ": " +
                                 hipGetErrorString(status));
    }
}

void use_device(int device) {
    hip_ok(hipSetDevice(device), "hipSetDevice");
}

// ---------------------------------------------------------------------------
// Allocation. c_heap and the signal region must be fine-grained: the protocol
// publishes epochs with system-scope relaxed stores into peer memory and reads
// peer payload after a pure acquire, neither of which is coherent through a
// coarse-grained (device-cached) allocation.
// ---------------------------------------------------------------------------
std::uintptr_t fine_alloc(int device, std::size_t nbytes) {
    use_device(device);
    void* pointer = nullptr;
    hip_ok(hipExtMallocWithFlags(&pointer, nbytes, hipDeviceMallocFinegrained),
           "hipExtMallocWithFlags(fine-grained)");
    hip_ok(hipMemset(pointer, 0, nbytes), "hipMemset");
    hip_ok(hipDeviceSynchronize(), "hipDeviceSynchronize");
    return reinterpret_cast<std::uintptr_t>(pointer);
}

std::uintptr_t plain_alloc(int device, std::size_t nbytes) {
    use_device(device);
    void* pointer = nullptr;
    hip_ok(hipMalloc(&pointer, nbytes), "hipMalloc");
    hip_ok(hipMemset(pointer, 0, nbytes), "hipMemset");
    hip_ok(hipDeviceSynchronize(), "hipDeviceSynchronize");
    return reinterpret_cast<std::uintptr_t>(pointer);
}

void free_device(int device, std::uintptr_t pointer) {
    use_device(device);
    hip_ok(hipFree(reinterpret_cast<void*>(pointer)), "hipFree");
}

void fill_bytes(int device, std::uintptr_t pointer, int value,
                std::size_t nbytes) {
    use_device(device);
    hip_ok(hipMemset(reinterpret_cast<void*>(pointer), value, nbytes),
           "hipMemset");
    hip_ok(hipDeviceSynchronize(), "hipDeviceSynchronize");
}

py::bytes device_to_host(int device, std::uintptr_t pointer,
                         std::size_t nbytes) {
    use_device(device);
    std::string buffer;
    buffer.resize(nbytes);
    hip_ok(hipMemcpy(buffer.data(), reinterpret_cast<const void*>(pointer),
                     nbytes, hipMemcpyDeviceToHost),
           "hipMemcpy D2H");
    return py::bytes(buffer);
}

void host_to_device(int device, std::uintptr_t pointer, const py::bytes& data) {
    use_device(device);
    const std::string buffer = data;
    hip_ok(hipMemcpy(reinterpret_cast<void*>(pointer), buffer.data(),
                     buffer.size(), hipMemcpyHostToDevice),
           "hipMemcpy H2D");
}

void device_synchronize(int device) {
    use_device(device);
    hip_ok(hipDeviceSynchronize(), "hipDeviceSynchronize");
}

int device_count() {
    int count = 0;
    hip_ok(hipGetDeviceCount(&count), "hipGetDeviceCount");
    return count;
}

// ---------------------------------------------------------------------------
// Peer access. Without this, a kernel on device d cannot dereference a pointer
// owned by device p even inside one process.
// ---------------------------------------------------------------------------
py::dict enable_peer_access(int world) {
    py::dict report;
    py::list failures;
    int enabled = 0, already = 0;
    for (int device = 0; device < world; ++device) {
        use_device(device);
        for (int peer = 0; peer < world; ++peer) {
            if (peer == device) continue;
            int can = 0;
            hip_ok(hipDeviceCanAccessPeer(&can, device, peer),
                   "hipDeviceCanAccessPeer");
            if (can == 0) {
                failures.append(py::make_tuple(device, peer, "cannot access"));
                continue;
            }
            const hipError_t status = hipDeviceEnablePeerAccess(peer, 0);
            if (status == hipSuccess) {
                ++enabled;
            } else if (status == hipErrorPeerAccessAlreadyEnabled) {
                ++already;
                (void)hipGetLastError();
            } else {
                failures.append(py::make_tuple(device, peer,
                                               hipGetErrorString(status)));
                (void)hipGetLastError();
            }
        }
    }
    report["enabled"] = enabled;
    report["already_enabled"] = already;
    report["failures"] = failures;
    return report;
}

// ---------------------------------------------------------------------------
// The production shape resolver, surfaced verbatim. Every scalar the kernel
// receives comes from here; nothing is recomputed on the Python side.
// ---------------------------------------------------------------------------
py::dict resolve_shape(int m, int n, int k_global, bool has_bias) {
    const m3h::launch_config plan = m3h::resolve_shape(m, n, k_global, has_bias);
    py::dict out;
    out["m"] = plan.key.m;
    out["n"] = plan.key.n;
    out["k_local"] = plan.key.k_local;
    out["has_bias"] = plan.key.has_bias;
    out["bm"] = plan.cfg.bm;
    out["bn"] = plan.cfg.bn;
    out["bk"] = plan.cfg.bk;
    out["num_reducer_ctas"] = plan.cfg.num_reducer_ctas;
    out["config_row"] = plan.cfg.config_row;
    out["slice_rows"] = plan.slice_rows;
    out["eb"] = plan.eb;
    out["lrow_count"] = plan.lrow_count;
    out["col_count"] = plan.col_count;
    out["gemm_tiles"] = plan.gemm_tiles;
    out["red_tiles"] = plan.red_tiles;
    out["num_gemm_ctas"] = plan.num_gemm_ctas;
    out["c_heap_bytes"] = plan.c_heap_bytes;
    out["ready_words"] = plan.ready_words;
    out["credit_words"] = plan.credit_words;
    out["signal_words_total"] = plan.signal_words_total;
    out["even_k"] = plan.even_k;
    out["even_n"] = plan.even_n;
    out["packet_fast_path"] = plan.packet_fast_path;
    out["lds_bytes"] =
        2 * (plan.cfg.bm + plan.cfg.bn) * plan.cfg.bk * 2;
    return out;
}

// Override the table's reducer split without touching the shape table: used by
// the NUM_REDUCER_CTAS sweep. Only num_gemm_ctas changes; the dependency
// layout (eb/lrow/col/ready_words) is independent of the split.
py::dict resolve_shape_with_split(int m, int n, int k_global, bool has_bias,
                                  int num_reducer_ctas) {
    py::dict out = resolve_shape(m, n, k_global, has_bias);
    if (num_reducer_ctas <= 0 || num_reducer_ctas >= m3::CU_COUNT) {
        throw std::invalid_argument("num_reducer_ctas outside (0, 304)");
    }
    out["num_reducer_ctas"] = num_reducer_ctas;
    out["num_gemm_ctas"] = m3::CU_COUNT - num_reducer_ctas;
    return out;
}

// ---------------------------------------------------------------------------
// Descriptor construction through the production snapshot helper.
// Returns the two device addresses to pass as c_heap_peers and sig_peers.
// ---------------------------------------------------------------------------
py::tuple make_descriptors(int device, int rank,
                           const std::vector<std::uintptr_t>& heap_bases,
                           std::uintptr_t c_heap, std::uintptr_t signals) {
    if (heap_bases.size() != 8u) {
        throw std::invalid_argument("expected exactly 8 heap bases");
    }
    iris::iris_device_view view;
    view.rank = rank;
    view.world = 8;
    for (std::size_t i = 0; i < 8u; ++i) {
        view.heaps[i] = heap_bases[i];
    }

    const m3h::allocation_descriptors descriptors =
        m3h::snapshot_allocation_descriptors(
            view, reinterpret_cast<const void*>(c_heap),
            reinterpret_cast<const void*>(signals));

    using descriptor_type = hk_gemm_rs::symmetric_descriptor;
    static_assert(sizeof(descriptor_type) == 72);

    use_device(device);
    void* device_pods = nullptr;
    hip_ok(hipMalloc(&device_pods, 2 * sizeof(descriptor_type)),
           "hipMalloc(descriptors)");
    descriptor_type staging[2] = {descriptors.c_heap, descriptors.signals};
    hip_ok(hipMemcpy(device_pods, staging, sizeof(staging),
                     hipMemcpyHostToDevice),
           "hipMemcpy(descriptors)");
    hip_ok(hipDeviceSynchronize(), "hipDeviceSynchronize");

    const auto base = reinterpret_cast<std::uintptr_t>(device_pods);
    return py::make_tuple(base, base + sizeof(descriptor_type));
}

// Two independent symmetric allocations, each anchored at its own heap base.
//
// The payload heap and the signal heap have different coherence requirements.
// Signals are relaxed system-scope atomics with no surrounding fence, so they
// must be fine-grained. The payload is published with an explicit directional
// release (buffer_wbl2 sc0 sc1) and consumed after a pure acquire
// (buffer_inv), which is precisely the handshake that makes a cached,
// coarse-grained allocation peer-visible. Allowing the two to be allocated
// with different granularity is therefore a first-class experiment, not a
// convenience.
py::tuple make_descriptors_split(int device, int rank,
                                 const std::vector<std::uintptr_t>& c_bases,
                                 std::uintptr_t c_heap,
                                 const std::vector<std::uintptr_t>& sig_bases,
                                 std::uintptr_t signals) {
    if (c_bases.size() != 8u || sig_bases.size() != 8u) {
        throw std::invalid_argument("expected exactly 8 bases per allocation");
    }
    auto view_for = [rank](const std::vector<std::uintptr_t>& bases) {
        iris::iris_device_view view;
        view.rank = rank;
        view.world = 8;
        for (std::size_t i = 0; i < 8u; ++i) view.heaps[i] = bases[i];
        return view;
    };

    using descriptor_type = hk_gemm_rs::symmetric_descriptor;
    const descriptor_type c_descriptor =
        hk_gemm_rs::host_abi::snapshot_allocation_descriptor(
            view_for(c_bases), reinterpret_cast<const void*>(c_heap));
    const descriptor_type sig_descriptor =
        hk_gemm_rs::host_abi::snapshot_allocation_descriptor(
            view_for(sig_bases), reinterpret_cast<const void*>(signals));

    use_device(device);
    void* device_pods = nullptr;
    hip_ok(hipMalloc(&device_pods, 2 * sizeof(descriptor_type)),
           "hipMalloc(descriptors)");
    descriptor_type staging[2] = {c_descriptor, sig_descriptor};
    hip_ok(hipMemcpy(device_pods, staging, sizeof(staging),
                     hipMemcpyHostToDevice),
           "hipMemcpy(descriptors)");
    hip_ok(hipDeviceSynchronize(), "hipDeviceSynchronize");

    const auto base = reinterpret_cast<std::uintptr_t>(device_pods);
    return py::make_tuple(base, base + sizeof(descriptor_type));
}

// Host-side echo of a descriptor's peer bases, so the driver can assert the
// rebasing arithmetic before trusting a single launch.
py::dict describe_descriptors(int rank,
                              const std::vector<std::uintptr_t>& heap_bases,
                              std::uintptr_t c_heap, std::uintptr_t signals) {
    if (heap_bases.size() != 8u) {
        throw std::invalid_argument("expected exactly 8 heap bases");
    }
    iris::iris_device_view view;
    view.rank = rank;
    view.world = 8;
    for (std::size_t i = 0; i < 8u; ++i) view.heaps[i] = heap_bases[i];

    const auto descriptors = m3h::snapshot_allocation_descriptors(
        view, reinterpret_cast<const void*>(c_heap),
        reinterpret_cast<const void*>(signals));

    auto to_list = [](const hk_gemm_rs::symmetric_descriptor& descriptor) {
        py::list bases;
        bases.append(reinterpret_cast<std::uintptr_t>(descriptor.bases.rank0));
        bases.append(reinterpret_cast<std::uintptr_t>(descriptor.bases.rank1));
        bases.append(reinterpret_cast<std::uintptr_t>(descriptor.bases.rank2));
        bases.append(reinterpret_cast<std::uintptr_t>(descriptor.bases.rank3));
        bases.append(reinterpret_cast<std::uintptr_t>(descriptor.bases.rank4));
        bases.append(reinterpret_cast<std::uintptr_t>(descriptor.bases.rank5));
        bases.append(reinterpret_cast<std::uintptr_t>(descriptor.bases.rank6));
        bases.append(reinterpret_cast<std::uintptr_t>(descriptor.bases.rank7));
        py::dict entry;
        entry["local_allocation_base"] =
            reinterpret_cast<std::uintptr_t>(descriptor.local_allocation_base);
        entry["peer_bases"] = bases;
        return entry;
    };

    py::dict out;
    out["c_heap"] = to_list(descriptors.c_heap);
    out["signals"] = to_list(descriptors.signals);
    return out;
}

} // namespace

PYBIND11_MODULE(dhk_rt, module) {
    module.doc() = "Symmetric-heap runtime glue for the gfx942 GEMM-RS validation harness";

    module.def("device_count", &device_count);
    module.def("fine_alloc", &fine_alloc, py::arg("device"), py::arg("nbytes"));
    module.def("plain_alloc", &plain_alloc, py::arg("device"), py::arg("nbytes"));
    module.def("free_device", &free_device, py::arg("device"), py::arg("pointer"));
    module.def("fill_bytes", &fill_bytes, py::arg("device"), py::arg("pointer"),
               py::arg("value"), py::arg("nbytes"));
    module.def("device_to_host", &device_to_host, py::arg("device"),
               py::arg("pointer"), py::arg("nbytes"));
    module.def("host_to_device", &host_to_device, py::arg("device"),
               py::arg("pointer"), py::arg("data"));
    module.def("device_synchronize", &device_synchronize, py::arg("device"));
    module.def("enable_peer_access", &enable_peer_access, py::arg("world") = 8);
    module.def("resolve_shape", &resolve_shape, py::arg("m"), py::arg("n"),
               py::arg("k_global"), py::arg("has_bias"));
    module.def("resolve_shape_with_split", &resolve_shape_with_split,
               py::arg("m"), py::arg("n"), py::arg("k_global"),
               py::arg("has_bias"), py::arg("num_reducer_ctas"));
    module.def("make_descriptors", &make_descriptors, py::arg("device"),
               py::arg("rank"), py::arg("heap_bases"), py::arg("c_heap"),
               py::arg("signals"));
    module.def("make_descriptors_split", &make_descriptors_split,
               py::arg("device"), py::arg("rank"), py::arg("c_bases"),
               py::arg("c_heap"), py::arg("sig_bases"), py::arg("signals"));
    module.def("describe_descriptors", &describe_descriptors, py::arg("rank"),
               py::arg("heap_bases"), py::arg("c_heap"), py::arg("signals"));

    module.attr("WORLD_SIZE") = m3::WORLD_SIZE;
    module.attr("CU_COUNT") = m3::CU_COUNT;
    module.attr("CTA_THREADS") = m3::CTA_THREADS;
    module.attr("EP_U32") = m3::EP_U32;
    module.attr("EP_GEMM_U32") = m3::EP_GEMM_U32;
    module.attr("EP_RED_U32") = m3::EP_RED_U32;
    module.attr("SIGNAL_GUARD_U32") = m3::SIGNAL_GUARD_U32;
    module.attr("ERR_PRODUCER_CREDIT") = m3::ERR_PRODUCER_CREDIT;
    module.attr("ERR_REDUCER_READY") = m3::ERR_REDUCER_READY;
    module.attr("CTRL_DROP_PUBLICATION") = m3::CTRL_DROP_PUBLICATION;
    module.attr("CTRL_REROUTE_SLOT") = m3::CTRL_REROUTE_SLOT;
    module.attr("CTRL_DROP_CREDIT") = m3::CTRL_DROP_CREDIT;
}
