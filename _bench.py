import traceback
from pathlib import Path

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import time

    B, INPUT, HIDDEN = 20000, 773, 256
    MAX_ACTIVE, ACTUAL = 37, 33

    indices = torch.randint(0, INPUT, (B, MAX_ACTIVE), dtype=torch.int16)
    counts = torch.full((B,), ACTUAL, dtype=torch.int64)

    # Approach: EmbeddingBag module with (INPUT, HIDDEN) weight
    # + separate bias parameter
    emb = nn.EmbeddingBag(INPUT, HIDDEN, mode="sum", sparse=False)
    bias1 = nn.Parameter(torch.zeros(HIDDEN))
    fc2 = nn.Linear(HIDDEN, 32)
    fc3 = nn.Linear(32, 1)

    def make_sparse_inputs():
        max_c = int(counts.max())
        active = indices[:, :max_c].long()
        mask = torch.arange(max_c).unsqueeze(0) < counts.unsqueeze(1)
        flat = active[mask]
        offsets = torch.zeros(B, dtype=torch.long)
        offsets[1:] = counts[:-1].cumsum(0)
        return flat, offsets

    def forward_emb(flat_idx, offsets):
        h1 = torch.clamp(emb(flat_idx, offsets) + bias1, 0.0, 1.0)
        h2 = torch.clamp(fc2(h1), 0.0, 1.0)
        return torch.tanh(fc3(h2)).squeeze(-1)

    # Also test dense approach for comparison
    fc1 = nn.Linear(INPUT, HIDDEN)
    # Copy weights so outputs match
    fc1.weight.data.copy_(emb.weight.data.t())
    fc1.bias.data.copy_(bias1.data)

    def forward_dense():
        max_c = int(counts.max())
        active = indices[:, :max_c].long()
        mask = torch.arange(max_c).unsqueeze(0) < counts.unsqueeze(1)
        features = torch.zeros(B, INPUT, dtype=torch.float32)
        features.scatter_(1, active * mask.long(), mask.float())
        h1 = torch.clamp(fc1(features), 0.0, 1.0)
        h2 = torch.clamp(fc2(h1), 0.0, 1.0)
        return torch.tanh(fc3(h2)).squeeze(-1)

    # Test outputs match
    flat, offsets = make_sparse_inputs()
    out_emb = forward_emb(flat, offsets)
    out_dense = forward_dense()
    match = torch.allclose(out_emb, out_dense, atol=1e-5)

    # Test gradient flow
    loss = out_emb.sum()
    loss.backward()
    grad_ok = emb.weight.grad is not None and emb.weight.grad.abs().sum() > 0

    # Benchmark full forward+backward+step
    params_emb = list(emb.parameters()) + [bias1] + list(fc2.parameters()) + list(fc3.parameters())
    opt_emb = torch.optim.Adam(params_emb, lr=0.001)
    opt_dense = torch.optim.Adam([p for p in [fc1.weight, fc1.bias] + list(fc2.parameters()) + list(fc3.parameters())], lr=0.001)
    target = torch.zeros(B)

    # warmup
    for _ in range(3):
        flat, offsets = make_sparse_inputs()
        F.mse_loss(forward_emb(flat, offsets), target).backward()
        opt_emb.zero_grad()

    N = 50
    results = [f"outputs_match: {match}", f"gradient_ok: {grad_ok}"]

    t0 = time.perf_counter()
    for _ in range(N):
        flat, offsets = make_sparse_inputs()
        out = forward_emb(flat, offsets)
        loss = F.mse_loss(out, target)
        opt_emb.zero_grad()
        loss.backward()
        opt_emb.step()
    emb_ms = (time.perf_counter() - t0) / N * 1000
    results.append(f"emb_bag fwd+bwd+step: {emb_ms:.1f} ms/batch")

    t0 = time.perf_counter()
    for _ in range(N):
        max_c = int(counts.max())
        active = indices[:, :max_c].long()
        mask = torch.arange(max_c).unsqueeze(0) < counts.unsqueeze(1)
        features = torch.zeros(B, INPUT, dtype=torch.float32)
        features.scatter_(1, active * mask.long(), mask.float())
        h1 = torch.clamp(fc1(features), 0.0, 1.0)
        h2 = torch.clamp(fc2(h1), 0.0, 1.0)
        out = torch.tanh(fc3(h2)).squeeze(-1)
        loss = F.mse_loss(out, target)
        opt_dense.zero_grad()
        loss.backward()
        opt_dense.step()
    dense_ms = (time.perf_counter() - t0) / N * 1000
    results.append(f"dense fwd+bwd+step: {dense_ms:.1f} ms/batch")

    results.append(f"speedup: {dense_ms / emb_ms:.1f}x")

    Path("_bench_results.txt").write_text("\n".join(results))
except Exception:
    Path("_bench_results.txt").write_text(traceback.format_exc())
