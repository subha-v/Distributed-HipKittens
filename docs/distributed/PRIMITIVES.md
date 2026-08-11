# Distributed primitive contracts

The distributed layer is included by `kittens.cuh` for CDNA3 and CDNA4. Its
types live in `kittens`; its operations live in `kittens::distributed`.
Everything is non-owning. Allocation, rank setup, dependency keys, CTA roles,
progress, and timeout policy remain explicit in operator code.

## Address projection

`pgl<GL, World>` wraps an ordinary global layout, its local symmetric-allocation
base, and eight named peer bases. `on<Rank>()` or `on(rank)` returns an ordinary
`GL` with this address rule:

```text
peer raw_ptr = peer allocation base
             + (local raw_ptr - local allocation base)
```

The offset term is mandatory for suballocations. PGL contains no completion,
epoch, synchronization, or scheduling state. Runtime invalid ranks fail closed
with a null `raw_ptr`; compile-time ranks are checked statically.

`translate_peer` and `peer_offset` expose the same rule for typed pointers that
are not represented by a global layout.

## Payload movement

`packet16`, `store_packet16`, `store_peer_packets`, and `load_peer_packets`
move aligned 16-byte packets. They do not publish readiness or insert a fence.
The checked store validates 16-byte alignment and byte count; the unchecked
forms rely on the caller's layout contract.

Use `store_packet_row<Chunks>` for register-resident packet arrays. Its compile-
time chunk count avoids the runtime indexing shape that previously demoted an
embedded packet array to private scratch.

## Ordering and visibility

Every metadata operation takes an explicit `memory_scope::agent` or
`memory_scope::system`. The common sequence for peer-visible payload is:

```cpp
// All producer threads have issued payload stores.
kittens::distributed::producer_drain_release<
    kittens::distributed::memory_scope::system>();

if (threadIdx.x == 0) {
  kittens::distributed::publish_epoch_relaxed<
      kittens::distributed::memory_scope::system>(ready, epoch);
}
```

`producer_drain_release` drains VMEM in every producer wave, converges the CTA,
and has the converged leader issue a directional release. Publication remains
separate so one release can cover multiple readiness cells.

On the consumer, initialize caller-owned result storage before every wait:

```cpp
kittens::distributed::wait_result observed;
kittens::distributed::init_wait_result(observed);
kittens::distributed::bounded_poll_relaxed_into(ready, epoch, limit, observed);
if (!observed.ready) {
  // Report or abandon this operator path. Do not read the payload.
  return;
}
kittens::distributed::cta_acquire<
    kittens::distributed::memory_scope::system>();
```

Use `bounded_observe_acquire_into` only when one calling thread is also the
payload consumer. A wider consumer role must poll, converge, and acquire as
shown above. The acquire is success-only and directional; it must not be
replaced with bidirectional `__threadfence_system()`.

Caller-owned `wait_result&` is an intentional code-generation constraint.
Returning the aggregate by value changed the bounded-poll CFG in the gfx942
experiments, while the caller-owned spelling matched the manual twin.

MoE-style self-describing completions use `epoch_word(epoch, payload)` and
`publish_epoch_word_relaxed`. Initialize a caller-owned `wait_result64`, then
use `bounded_poll_epoch_word_relaxed_into` to require an exact high-half epoch
while retaining the low-half row count or extent. Exact equality is deliberate
for an address-reused word: seeing a newer epoch means the expected payload is
already unavailable, not that it is safe to consume.

## Epochs, fan-in, and reservations

`epoch32` advances a caller-owned monotonic cell once per convergent CTA call
and broadcasts the captured value. The cell must not be shared by concurrently
running CTAs. Two CTA barriers prevent a fast leader from advancing the next
epoch while a slower wave still captures the current one.

`counted_arrive_into` implements cumulative local fan-in for a fixed, exact
producer count. It never resets the counter; the quotient and remainder return
the epoch and producer ordinal. Its agent-scope acq_rel chain makes prior
producer releases visible to the last arriver, but the raw form does not export
those newly acquired effects to a peer. Use
`counted_arrive_release_into<system>` before a relaxed remote completion store;
only the last arriver pays that final release, and that same thread must perform
the publication.

The 32-bit protocols use the donors' plain monotonic comparisons, reserve zero
for initialization, and do not infer modular rollover. Quiesce all producers
and consumers and reinitialize the complete protocol instance before its epoch
or cumulative counter would wrap. Callers inspect `wait_result::ready`
directly, so timeout is never encoded in a bit pattern that could collide with
a valid epoch.

`fetch_add_relaxed` and `reserve_rows_relaxed` return the pre-increment value of
an agent- or system-visible monotonic counter. They reserve storage only; they
do not imply payload publication.

## Role partitions

`finish_order_partition(ticket_cell, service_ctas, total_ctas)` splits a
persistent grid once per phase boundary by *arrival order*: one elected thread
per CTA reserves a monotonic agent ticket, the first `service_ctas` finishers
become minimum-progress service CTAs, and every other CTA receives a dense
compute id in `[0, total_ctas - service_ctas)`. The partition adds no grid
barrier — it rides the completion order of the phase that just ended. The
contract:

- The ticket cell is caller-owned, agent-local, and re-zeroed by the caller
  once per epoch/phase.
- The split is fixed for the phase. There is no in-phase resizing; this is
  deliberately the COMET/MoK profile-then-fix model, with the AMD-specific
  refinement that service CTAs exist from the first finisher and compute CTAs
  join service/reduction work the moment their own queue drains.
- Downstream task loops stride by the *dense* pool count so reserving CTAs
  never silently drops tasks.

`publish_tile_release` (release + relaxed store of a tile key) and
`wait_tile_acquire_into` (bounded poll + success-only acquire) are the tile
granularity aliases of the ordering spine above; `retire_epoch` names the
monotonic epoch-cell store. They exist so operator code reads as its dataflow.

## Reused storage

Readiness does not protect storage lifetime. If epoch N and epoch N+1 reuse the
same address, the producer must wait for a directed credit from the unique
consumer of epoch N:

```cpp
kittens::distributed::init_wait_result(observed);
kittens::distributed::bounded_wait_slot_reusable_into(
    credit, next_epoch, limit, observed);
if (!observed.ready) return;

// Produce, release, and publish the next payload.
```

After its final payload load, the consumer calls `drain_and_retire_slot` with
the consumed epoch. Epoch one is a fast path because the slot has no prior
payload, but its result object must still be initialized first.

The retirement credit carries only permission to overwrite. It is sound after
all relevant consumer VMEM reads drain and the CTA converges; it does not
publish consumer result writes. If a producer must observe consumer-written
data, establish a separate release/publication/acquire edge for that data.

## What is deliberately absent

These headers do not provide a graph, planner, scheduler, collective, hidden
grid barrier, dependency-key allocator, or distributed compute-tile type. A
bounded wait prevents an infinite spin; it does not prove that a producer CTA
will be scheduled. That progress proof belongs beside the operator's CTA-role
mapping.
