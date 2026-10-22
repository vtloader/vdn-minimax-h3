"""The delta-rule state-update backends: how one frame's statistics
(A = k^T beta k, B = v^T beta k) become the scan's transition and injection.
"""
import torch


def _blocked_triangular_inverse(lower, block=16):
    """Invert a batched lower-triangular matrix with NPU GEMMs.

    ``torch.linalg.solve_triangular`` has no NPU kernel and falls back to AiCPU,
    which serialises every DiT block.  Invert each 16x16 diagonal block by forward
    substitution over its rows, then merge blocks with the Schur complement
    ``-D^-1 C A^-1``.  The arithmetic is the same exact triangular inverse; only the
    association changes, and all matrix products stay on the NPU.
    """
    shape = lower.shape
    n = shape[-1]
    if n % block or n < block:
        raise ValueError(f"blocked triangular inverse needs a multiple of {block}, got {n}")
    flat = lower.reshape(-1, n, n)
    batch, blocks = flat.shape[0], n // block
    diagonal = torch.stack(
        [flat[:, offset:offset + block, offset:offset + block]
         for offset in range(0, n, block)], dim=1)
    identity = torch.eye(block, device=flat.device, dtype=flat.dtype)
    inverse = torch.zeros_like(diagonal)
    for row in range(block):
        rhs = identity[row].expand(batch, blocks, block)
        if row:
            rhs = rhs - (diagonal[:, :, row, :row].unsqueeze(-2)
                         @ inverse[:, :, :row, :]).squeeze(-2)
        inverse[:, :, row, :] = rhs / diagonal[:, :, row, row].unsqueeze(-1)

    size = block
    while size < n:
        left = inverse[:, 0::2]
        right = inverse[:, 1::2]
        cross = torch.stack(
            [flat[:, offset + size:offset + 2 * size, offset:offset + size]
             for offset in range(0, n, 2 * size)], dim=1)
        lower_left = -(right @ cross @ left)
        inverse = torch.cat([
            torch.cat([left, torch.zeros_like(left)], dim=-1),
            torch.cat([lower_left, right], dim=-1),
        ], dim=-2)
        size *= 2
    return inverse.reshape(shape)


def _triangular_inverse(lower):
    """Exact inverse of a lower-triangular matrix, keeping inference GEMMs on the NPU.

    The blocked path uses in-place row assembly, so it is inference-only. Training
    keeps the autograd-safe trsm spelling (and its NPU AiCPU fallback).
    """
    if (lower.device.type == "npu" and lower.shape[-1] % 16 == 0
            and not torch.is_grad_enabled()):
        return _blocked_triangular_inverse(lower)
    identity = torch.eye(lower.shape[-1], device=lower.device, dtype=lower.dtype)
    return torch.linalg.solve_triangular(
        lower, identity.expand_as(lower), upper=False, left=True)


class SanaDelta:
    """Scaled subtractive delta: S_out = (S_in Diag(D))(I - c^2 A) + c B, c=1/sqrt(S).
    L2-normalized keys give trace(c^2 A) <= 1, hence 0 <= c^2 A <= I and (I - c^2 A) is
    non-expansive; the diagonal gate D in (0,1] preserves that."""

    def __init__(self, tokens_per_frame: int):
        self.inv_tokens = 1.0 / tokens_per_frame               # c^2
        self.inv_sqrt_tokens = self.inv_tokens ** 0.5          # c

    def factor_apply(self, alpha, A_raw, B_raw):
        """[F,H,d,d] batched. transition = Diag(alpha)(I - c^2 A) (row-scaled),
        injection = c B."""
        eye = torch.eye(A_raw.shape[-1], device=A_raw.device, dtype=A_raw.dtype)
        transition = alpha.unsqueeze(-1) * (eye - self.inv_tokens * A_raw)
        injection = self.inv_sqrt_tokens * B_raw
        return transition, injection

    def step_ref(self, state, alpha, A_raw, B_raw):
        """One frame of the recurrence, written out. The shipped path factors ALL frames
        at once (factor_apply) and then runs the bare recurrence in _run_scans; this is
        here so the tests can replay a scan step by step."""
        transition, injection = self.factor_apply(alpha, A_raw, B_raw)
        return state @ transition + injection


class VdnDelta:
    """Unscaled joint solve: S_out = (S_in Diag(D) + B)(I + A)^{-1}.

    I + A is SPD. Phase A does ONE batched inverse over all frames and forms the
    transition explicitly, because the serial scan applies it repeatedly anyway. This
    is the rule the released checkpoints use; its small equilibrium beta is cancelled
    by the branch norm before to_out_linear ever sees it.

    The inverse is an exact Cholesky.  On CUDA the batched trsm runs on cuSOLVER; on
    NPU, where trsm has no kernel and falls back to AiCPU, `_triangular_inverse`
    replaces it with blocked NPU GEMMs."""

    def __init__(self, tokens_per_frame: int = None):
        pass                                     # unscaled: S plays no part

    def factor_apply(self, alpha, A_raw, B_raw):
        A32 = A_raw.float()
        eye = torch.eye(A32.shape[-1], device=A32.device, dtype=torch.float32).expand_as(A32)
        chol = torch.linalg.cholesky(A32 + eye)

        # (I+A)^{-1} = L^{-T} L^{-1}, as ONE triangular solve and a matmul rather
        # than `cholesky_solve(I, L)`, which is two triangular solves. A batched trsm
        # at 128x128 runs an order of magnitude below a batched GEMM at the same
        # shape, so trading the second solve for a product is a clear win over the
        # F*H systems a frame bank holds. Same matrix to fp32 rounding on a different
        # association, not an approximation.
        linv = _triangular_inverse(chol)
        inv = linv.transpose(-1, -2) @ linv                   # symmetric by construction
        transition = alpha.unsqueeze(-1) * inv
        injection = B_raw.float() @ inv
        return transition.to(A_raw.dtype), injection.to(B_raw.dtype)

    def step_ref(self, state, alpha, A_raw, B_raw):
        """See SanaDelta.step_ref — reference only, the scan does not call this."""
        transition, injection = self.factor_apply(alpha, A_raw, B_raw)
        return state @ transition + injection


class VdnScaledDelta(VdnDelta):
    """Exact joint solve WITH SANA's key scaling:
    S_out = (S_in Diag(D) + cB)(I + c^2 A)^-1.

    A control arm, kept for interpretability, not to train with: once c^2 = 1/S forces
    trace(c^2 A) <= 1, the exact inverse and the first-order truncation (I - c^2 A) are
    very nearly the same operator."""

    def __init__(self, tokens_per_frame: int):
        super().__init__(tokens_per_frame)
        self.inv_tokens = 1.0 / tokens_per_frame              # c^2
        self.inv_sqrt_tokens = self.inv_tokens ** 0.5         # c

    def factor_apply(self, alpha, A_raw, B_raw):
        A32 = A_raw.float() * self.inv_tokens
        eye = torch.eye(A32.shape[-1], device=A32.device,
                        dtype=torch.float32).expand_as(A32)
        chol = torch.linalg.cholesky(A32 + eye)
        inv = torch.cholesky_solve(eye.contiguous(), chol)
        transition = alpha.unsqueeze(-1) * inv
        injection = (B_raw.float() * self.inv_sqrt_tokens) @ inv
        return transition.to(A_raw.dtype), injection.to(B_raw.dtype)


DELTA_BACKENDS = {"sana_scaled": SanaDelta, "vdn_solve": VdnDelta,
                  "vdn_scaled": VdnScaledDelta}

