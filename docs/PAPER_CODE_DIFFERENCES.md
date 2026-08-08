# Paper, uploaded code, and package behavior

This document records differences rather than silently choosing one interpretation.
The source baseline is upstream commit `9481a822e73e5a7520c6c0a425a8a402f2878c03`.

| Topic | Audited uploaded code | Package response |
| --- | --- | --- |
| Reaction primitive library | Active runs use `sin, cos, exp`; a larger variable also contains `x, x2, id` | Separate `legacy`, `paper`, and `corrected` YAML profiles |
| Readout | Trainable `nn.Linear` weight; fixed-weight line commented | `trainable_linear` compatibility mode and explicit `fixed_sum` mode |
| Validation | `F.gumbel_softmax` remains stochastic under `eval()` | Evaluation uses deterministic softmax; hardened evaluation uses argmax |
| AdamW decay | Function records `weight_decay`, but normal group omits it | Every parameter group's effective decay is explicit and testable |
| “NMS” | Implemented as winner off-mass on each edge | Off-mass retained separately; NMS is pairwise edge-distribution overlap |
| Best checkpoint | Saved periodically, but final post-processing uses final model | Best soft state is restored before hardening; best hardened state is restored for export |
| Volterra quadrature | Global Gauss weights truncated by a causal mask | Nodes and weights are remapped to every interval `[0, x]` |
| Volterra complexity | Recreates a 1000×1000 matrix in each loss call | Cached base rule and O(NQ) mapped evaluation |
| Inverse primitive | `z + eps*sign(z)` is still zero at `z=0` | Sign is defined as +1 at zero, yielding a finite denominator |
| Configuration | Several behavior-changing fields are absent from snapshot | Full model/training/problem mapping is serializable |
| Dtype | Global float32 | Explicit float32/float64 configuration |
| Unit pruning | Implemented but disabled in public active runs | Optional and disabled in paper/legacy profiles unless requested |

## Interpretation rules

- `legacy` is a compatibility target, not an endorsement of every upstream numerical choice.
- `paper` encodes manuscript-stated settings where identifiable, but does not guarantee
  reproduction because not all paper experiments were uploaded.
- `corrected` is the maintainers' derivative behavior and must be reported as such.
- `smoke` validates code paths only; it is not a scientific result.
