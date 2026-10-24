# Supplementary Material: WARI-L Algorithms

## Algorithm 1: WARI-L Sublayer Dispatch
```python
def dispatch(layer_name, x, core, dense_blob, controller, gate, memory_sim):
    is_pinned = controller.is_pinned(layer_name)
    decision, score = gate.decide(layer_name, x, core, dense_fn=lambda a: F.linear(a, dense_blob.W, dense_blob.b))
    
    # Telemetry and I/O tracking
    memory_sim.record_sublayer(
        layer_name=layer_name,
        decision=decision,
        is_pinned=is_pinned,
        core_bytes=core.core_bytes,
        dense_bytes=core.dense_bytes,
    )
    
    if decision == "skip":
        return torch.zeros(x.shape[:-1] + (core.out_features,), dtype=x.dtype, device=x.device)
    elif decision == "core":
        # Evaluated purely in RAM via thin GEMMs: y = U(V(x)) + b
        return core(x)
    elif decision == "full":
        # Load W if unpinned; NEVER add UV
        return F.linear(x, dense_blob.W, dense_blob.b)
```

## Algorithm 2: Session-Adaptive Working-Set Controller
```python
def update_working_set(controller, layer_decisions, beta, pin_budget_bytes):
    # Step 1: Exponential Moving Average update
    for layer in controller.all_layers:
        is_full = 1.0 if layer_decisions.get(layer) == "full" else 0.0
        controller.ema[layer] = beta * controller.ema[layer] + (1.0 - beta) * is_full

    # Step 2: Greedy priority pinning under hard byte budget
    sorted_layers = sorted(controller.all_layers, key=lambda l: controller.ema[l], reverse=True)
    
    new_pinned = set()
    used_bytes = 0
    for layer in sorted_layers:
        cost = controller.layer_sizes[layer]
        if used_bytes + cost <= pin_budget_bytes:
            new_pinned.add(layer)
            used_bytes += cost
            
    controller.pinned_layers = new_pinned
```
