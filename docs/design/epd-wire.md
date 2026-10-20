# Embedding transfer wire contract

The encoder-to-prefill embedding path uses the existing Mooncake data plane and
ZMQ control frames. Endpoint registration has six fields. Per-request transfer
frames have thirteen fields, including the full image span and the receiver's row
range. Device pointers are unsigned 64-bit little-endian values; other fields use
ASCII. The deepstack presence flag is exactly `0` or `1`.

Parsing rejects truncated messages, invalid presence flags, nonpositive receiver
fanout, and null destination pointers for nonempty payloads. A nonempty transfer
with deepstack enabled must provide both destination pointers: omitting the
second write must never look like successful delivery. Empty row shards remain
valid registration heartbeats and need no writable destination.

The existing append-only frame behavior is retained. The parser also retains the
legacy/default zero-span representation; the sender's full-image span check
still determines whether a transfer matches its actual encoded chunk. Parsing
does not replace the sender's shape, dtype, row-range, fanout coverage, and
image-token-count validation.

This format does not negotiate model or processor revisions. Deployment must
keep peers on matching model, processor, and wire implementations. These CPU
serialization checks do not establish remote buffer ownership, cancellation
quiescence, or GPU visibility after an RDMA write; those require transport and
hardware validation.

## Sender failure completion

A room's terminal status permits its encoder ring slot to be reused. The sender
therefore drains every submitted synchronous write before publishing any
completion notification or terminal transfer failure, even if one write raises
or a later submission fails. Transport-owned write deadlines remain responsible
for making progress; the barrier cannot abandon a still-running write safely.
Notification failures are isolated per receiver and cannot terminate the queue
worker. Already-terminal queued rooms are discarded before using their pointers.

This barrier does not establish cancellation quiescence for external failures
concurrent with active DMA, or recovery from a failed CUDA copy event. Those
remain separate ownership and hardware validation requirements.
