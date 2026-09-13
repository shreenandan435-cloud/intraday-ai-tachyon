#include "tachyon/execution/SharedMemory.hpp"
#include "tachyon/execution/SpscRing.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

namespace {

using tachyon::execution::LOBStateSlot;
using tachyon::execution::SharedMemorySegment;
using tachyon::execution::SpscRingBuffer;

std::uint64_t NowNs() {
    return static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now().time_since_epoch())
            .count());
}

float Payload(std::uint64_t sequence, int index) {
    const std::uint64_t mix = sequence * 131ULL + static_cast<std::uint64_t>(index) * 17ULL;
    return static_cast<float>(mix % 9973ULL) / 16.0f;
}

void FillSlot(LOBStateSlot& slot, std::uint64_t sequence) {
    slot.timestamp_ns = NowNs();
    slot.sequence = sequence;
    for (int j = 0; j < 8; ++j) {
        slot.lob_state[j] = Payload(sequence, j);
    }
    slot.flags = static_cast<std::uint32_t>(sequence & 0xFFFFFFFFULL);
}

bool SlotIntact(const LOBStateSlot& slot) {
    if (slot.flags != static_cast<std::uint32_t>(slot.sequence & 0xFFFFFFFFULL)) {
        return false;
    }
    for (int j = 0; j < 8; ++j) {
        float expected = Payload(slot.sequence, j);
        if (slot.lob_state[j] != expected) {
            return false;
        }
    }
    return true;
}

int TestBasicFifo() {
    SpscRingBuffer<LOBStateSlot, 8> ring;
    for (std::uint64_t seq = 1; seq <= 5; ++seq) {
        LOBStateSlot slot{};
        FillSlot(slot, seq);
        ring.Push(slot);
    }
    for (std::uint64_t expected = 1; expected <= 5; ++expected) {
        LOBStateSlot out{};
        if (!ring.TryPop(out)) {
            std::printf("FAIL basic: queue empty at seq %llu\n",
                        static_cast<unsigned long long>(expected));
            return 1;
        }
        if (out.sequence != expected || !SlotIntact(out)) {
            std::printf("FAIL basic: order/integrity mismatch at seq %llu\n",
                        static_cast<unsigned long long>(expected));
            return 1;
        }
    }
    LOBStateSlot extra{};
    if (ring.TryPop(extra)) {
        std::printf("FAIL basic: drained queue returned data\n");
        return 1;
    }

    for (std::uint64_t seq = 101; seq <= 108; ++seq) {
        LOBStateSlot slot{};
        FillSlot(slot, seq);
        ring.Push(slot);
    }
    std::vector<LOBStateSlot> drained(8);
    const std::size_t got = ring.TryPopBulk(drained.data(), drained.size());
    if (got != 8) {
        std::printf("FAIL basic: bulk drain got %zu of 8\n", got);
        return 1;
    }
    for (std::size_t i = 0; i < got; ++i) {
        if (drained[i].sequence != 101 + i || !SlotIntact(drained[i])) {
            std::printf("FAIL basic: bulk drain integrity at %zu\n", i);
            return 1;
        }
    }
    for (std::uint64_t seq = 201; seq <= 205; ++seq) {
        LOBStateSlot slot{};
        FillSlot(slot, seq);
        ring.Push(slot);
    }
    const std::size_t partial = ring.TryPopBulk(drained.data(), 8);
    if (partial != 5 || drained[0].sequence != 201) {
        std::printf("FAIL basic: partial bulk drain returned %zu\n", partial);
        return 1;
    }

    std::printf("PASS basic_fifo\n");
    return 0;
}

int TestOverwriteGapAccounting() {
    constexpr std::uint64_t kPushed = 28;
    SpscRingBuffer<LOBStateSlot, 8> ring;
    for (std::uint64_t seq = 1; seq <= kPushed; ++seq) {
        LOBStateSlot slot{};
        FillSlot(slot, seq);
        ring.Push(slot);
    }

    std::uint64_t consumed = 0;
    std::uint64_t gapped = 0;
    std::uint64_t last_seq = 0;
    for (;;) {
        LOBStateSlot out{};
        std::uint64_t gap = 0;
        if (!ring.TryPop(out, &gap)) {
            break;
        }
        if (!SlotIntact(out)) {
            std::printf("FAIL overwrite: torn or corrupt slot seq=%llu\n",
                        static_cast<unsigned long long>(out.sequence));
            return 1;
        }
        if (last_seq != 0 && out.sequence <= last_seq) {
            std::printf("FAIL overwrite: non-monotonic sequence %llu after %llu\n",
                        static_cast<unsigned long long>(out.sequence),
                        static_cast<unsigned long long>(last_seq));
            return 1;
        }
        consumed += 1;
        gapped += gap;
        last_seq = out.sequence;
    }

    if (consumed + gapped != kPushed) {
        std::printf(
            "FAIL overwrite: accounting consumed=%llu gapped=%llu vs pushed=%llu\n",
            static_cast<unsigned long long>(consumed),
            static_cast<unsigned long long>(gapped),
            static_cast<unsigned long long>(kPushed));
        return 1;
    }
    if (ring.total_dropped() != gapped) {
        std::printf("FAIL overwrite: cumulative drop counter mismatch\n");
        return 1;
    }
    std::printf("PASS overwrite_gaps (consumed=%llu, detected_gaps=%llu)\n",
                static_cast<unsigned long long>(consumed),
                static_cast<unsigned long long>(gapped));
    return 0;
}

int TestSharedMemoryTransport() {
    using Ring = SpscRingBuffer<LOBStateSlot, 4096>;
    constexpr std::size_t kSegmentBytes = 1U << 20;

    try {
        std::printf("[shm] creating owner segment\n");
        SharedMemorySegment owner(tachyon::execution::kLobSharedMemoryName, kSegmentBytes,
                                  SharedMemorySegment::Mode::kCreateOrOpen);
        if (!owner.IsValid()) {
            std::printf("FAIL shm: owner segment invalid\n");
            return 1;
        }
        auto* writer_ring = new (owner.data()) Ring();
        for (std::uint64_t seq = 1; seq <= 1000; ++seq) {
            LOBStateSlot slot{};
            FillSlot(slot, seq);
            writer_ring->Push(slot);
        }
        std::printf("[shm] 1000 pushes done\n");

        SharedMemorySegment reader(tachyon::execution::kLobSharedMemoryName, kSegmentBytes,
                                   SharedMemorySegment::Mode::kOpenExisting);
        if (!reader.IsValid() || reader.data() == owner.data()) {
            std::printf("FAIL shm: second mapping missing or aliased\n");
            return 1;
        }
        std::printf("[shm] reader mapping ready, draining\n");
        auto* reader_ring = reinterpret_cast<Ring*>(reader.data());
        std::uint64_t popped = 0;
        std::uint64_t gaps = 0;
        for (;;) {
            LOBStateSlot out{};
            std::uint64_t gap = 0;
            if (!reader_ring->TryPop(out, &gap)) {
                break;
            }
            if (!SlotIntact(out)) {
                std::printf("FAIL shm: integrity across mapping boundary\n");
                return 1;
            }
            popped += 1;
            gaps += gap;
        }
        if (popped + gaps != 1000) {
            std::printf("FAIL shm: accounting across mappings\n");
            return 1;
        }
        reader_ring->~Ring();
        writer_ring->~Ring();
        SharedMemorySegment::Remove(tachyon::execution::kLobSharedMemoryName);
        std::printf("PASS shared_memory (%llu slots via dual mapping)\n",
                    static_cast<unsigned long long>(popped));
        return 0;
    } catch (const std::exception& e) {
        std::printf("FAIL shm: exception: %s\n", e.what());
        return 1;
    }
}

int TestConcurrentStress() {
    constexpr std::uint64_t kTotalSlots = 10'000'000ULL;
    constexpr std::size_t kCapacity = 65536;
    auto ring = std::make_unique<SpscRingBuffer<LOBStateSlot, kCapacity>>();

    std::vector<std::uint64_t> latencies;
    latencies.reserve(kTotalSlots);

    std::atomic<bool> producer_done{false};
    std::atomic<std::uint64_t> integrity_failures{0};

    const auto consumer_body = [&] {
        LOBStateSlot out{};
        for (;;) {
            if (ring->TryPop(out)) {
                if (!SlotIntact(out)) {
                    integrity_failures.fetch_add(1, std::memory_order_relaxed);
                    continue;
                }
                latencies.push_back(NowNs() - out.timestamp_ns);
                continue;
            }
            if (producer_done.load(std::memory_order_acquire)) {
                break;
            }
            tachyon::execution::PauseCpu();
        }
    };

    const auto start = std::chrono::steady_clock::now();
    std::thread consumer(consumer_body);
    for (std::uint64_t seq = 1; seq <= kTotalSlots; ++seq) {
        LOBStateSlot slot{};
        FillSlot(slot, seq);
        ring->Push(slot);
    }
    producer_done.store(true, std::memory_order_release);
    consumer.join();
    const auto stop = std::chrono::steady_clock::now();

    const double seconds =
        std::chrono::duration<double>(stop - start).count();
    const std::uint64_t dropped = ring->total_dropped();

    if (integrity_failures.load() != 0) {
        std::printf("FAIL stress: %llu torn/corrupt slots observed\n",
                    static_cast<unsigned long long>(integrity_failures.load()));
        return 1;
    }
    if (latencies.size() + dropped != kTotalSlots) {
        std::printf("FAIL stress: accounting consumed=%llu dropped=%llu vs pushed=%llu\n",
                    static_cast<unsigned long long>(latencies.size()),
                    static_cast<unsigned long long>(dropped),
                    static_cast<unsigned long long>(kTotalSlots));
        return 1;
    }
    if (latencies.empty()) {
        std::printf("FAIL stress: nothing consumed\n");
        return 1;
    }

    auto percentile = [&](double q) {
        const std::size_t idx = static_cast<std::size_t>(q * static_cast<double>(latencies.size() - 1));
        std::nth_element(latencies.begin(), latencies.begin() + static_cast<std::ptrdiff_t>(idx),
                         latencies.end());
        return latencies[idx];
    };
    const std::uint64_t p50 = percentile(0.50);
    const std::uint64_t p99 = percentile(0.99);
    const std::uint64_t p999 = percentile(0.999);
    const std::uint64_t max_ns = *std::max_element(latencies.begin(), latencies.end());

    std::printf(
        "stress: pushed=%llu consumed=%llu dropped=%llu throughput=%.1fM ops/s | "
        "queue-wait p50=%lluns p99=%lluns max=%lluns\n",
        static_cast<unsigned long long>(kTotalSlots),
        static_cast<unsigned long long>(latencies.size()),
        static_cast<unsigned long long>(dropped),
        static_cast<double>(kTotalSlots) / seconds / 1.0e6,
        static_cast<unsigned long long>(p50),
        static_cast<unsigned long long>(p99),
        static_cast<unsigned long long>(max_ns));

    std::printf("PASS concurrent_stress (zero tearing, exact sequence accounting)\n");
    return 0;
}

int TestHandoffLatency() {
    constexpr std::uint64_t kTotalSlots = 1'000'000ULL;
    constexpr std::size_t kCapacity = 65536;
    auto ring = std::make_unique<SpscRingBuffer<LOBStateSlot, kCapacity>>();

    std::vector<std::uint64_t> latencies;
    latencies.reserve(kTotalSlots);
    std::atomic<std::uint64_t> last_popped{0};
    std::atomic<std::uint64_t> integrity_failures{0};

    const auto consumer_body = [&] {
        LOBStateSlot out{};
        while (last_popped.load(std::memory_order_acquire) < kTotalSlots) {
            if (!ring->TryPop(out)) {
                tachyon::execution::PauseCpu();
                continue;
            }
            if (!SlotIntact(out)) {
                integrity_failures.fetch_add(1, std::memory_order_relaxed);
                last_popped.store(kTotalSlots, std::memory_order_release);
                return;
            }
            latencies.push_back(NowNs() - out.timestamp_ns);
            last_popped.store(out.sequence, std::memory_order_release);
        }
    };

    std::thread consumer(consumer_body);
    for (std::uint64_t seq = 1; seq <= kTotalSlots; ++seq) {
        LOBStateSlot slot{};
        FillSlot(slot, seq);
        ring->Push(slot);
        while (last_popped.load(std::memory_order_acquire) < seq) {
            tachyon::execution::PauseCpu();
        }
    }
    consumer.join();

    if (integrity_failures.load() != 0) {
        std::printf("FAIL latency: torn slots observed\n");
        return 1;
    }
    auto percentile = [&](double q) {
        const std::size_t idx = static_cast<std::size_t>(q * static_cast<double>(latencies.size() - 1));
        std::nth_element(latencies.begin(), latencies.begin() + static_cast<std::ptrdiff_t>(idx),
                         latencies.end());
        return latencies[idx];
    };
    const std::uint64_t p50 = percentile(0.50);
    const std::uint64_t p99 = percentile(0.99);
    const std::uint64_t p999 = percentile(0.999);
    std::printf("handoff: n=%llu | p50=%lluns p99=%lluns p99.9=%lluns\n",
                static_cast<unsigned long long>(latencies.size()),
                static_cast<unsigned long long>(p50),
                static_cast<unsigned long long>(p99),
                static_cast<unsigned long long>(p999));

    if (p99 > 25'000ULL) {
        std::printf("FAIL latency: p99 %lluns exceeds 25us gate\n",
                    static_cast<unsigned long long>(p99));
        return 1;
    }
    std::printf("PASS handoff_latency\n");
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    const std::string filter = argc > 1 ? argv[1] : "all";
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    int rc = 0;
    if (filter == "all" || filter == "basic") rc |= TestBasicFifo();
    if (filter == "all" || filter == "gaps") rc |= TestOverwriteGapAccounting();
    if (filter == "all" || filter == "shm") rc |= TestSharedMemoryTransport();
    if (filter == "all" || filter == "stress") rc |= TestConcurrentStress();
    if (filter == "all" || filter == "latency") rc |= TestHandoffLatency();
    if (rc == 0) {
        std::printf("ALL SPSC TESTS PASSED\n");
    }
    return rc;
}
