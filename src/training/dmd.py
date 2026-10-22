"""DMD2 (no GAN) for Stage-DMD: three roles on ONE FSDP model, the few-step sampler the
generator is trained through, the distribution-matching gradient and the fake-score
regression. Pure functions over packed fp32 rows, so the math is testable on CPU.

Roles are (teacher_mode x active adapters), switched between forwards -- no second copy
of the 33B base, the same trick as Stage-B's teacher forward:

  generator   VDN + `default` (frozen Stage-B LoRA) + `turbo` (trainable, initialised
              from an external few-step LoRA)
  real score  real_score=dense: the released dense model (teacher_mode, no adapters)
              real_score=vdn:   VDN + `default` -- the generator minus `turbo`
  fake score  the real score's architecture + `fake` (trainable; B=0 at init, so the
              fake IS the real score at step 0)

Conventions are the repo's (t2va_batch): t = 1 - sigma, x_t = t*x0 + (1-t)*eps, the
model predicts the data-ward velocity v = x0 - eps, x0 = x_t + (1-t)*v, and Euler is the
scheduler's r*x_t + (1-r)*x0 blend, arithmetic verbatim (euler_blend).
"""
import torch
import torch.nn.functional as F

from src.models.hybrid_transform import set_teacher_mode
from src.training import fsdp_stage as fs
from src.training.t2va_batch import (euler_blend, few_step_sigmas, few_step_timesteps,
                                     x0_from_velocity)
from src.training.turbo_adapter import set_active_adapters

REAL_SCORES = ("dense", "vdn")
GENERATOR, REAL, FAKE = "generator", "real", "fake"
WARMUP_TURN, GENERATOR_TURN, FAKE_TURN = "warmup", "generator", "fake"


def turn(fake_step, warmup_steps, updates_per_step):
    """Which kind of sub-iteration comes next -- DMD2's loop shape. EVERY sub-iteration
    draws a fresh prompt, rolls the generator out and updates the fake once on that
    sample; the generator additionally updates on the first sub-iteration of every
    group of `updates_per_step` (sharing that sample with the fake), once the
    `warmup_steps` fake-only sub-iterations are done. A pure function of the saved
    `fake_step`, so a resume lands on the right turn."""
    if fake_step < warmup_steps:
        return WARMUP_TURN
    return (GENERATOR_TURN if (fake_step - warmup_steps) % updates_per_step == 0
            else FAKE_TURN)


class Roles:
    """Which forward is which. `set(role)` flips teacher_mode and the active adapter
    list; it never touches requires_grad (FSDP2 owns that after sharding)."""

    def __init__(self, model, real_score, vdn_adapter="default", turbo_adapter="turbo",
                 fake_adapter="fake"):
        if real_score not in REAL_SCORES:
            raise ValueError(f"real_score must be one of {REAL_SCORES}, got {real_score!r}")
        dense = real_score == "dense"
        self.model = model
        self.real_score = real_score
        self.table = {
            GENERATOR: (False, [vdn_adapter, turbo_adapter]),
            REAL: (True, []) if dense else (False, [vdn_adapter]),
            FAKE: (True, [fake_adapter]) if dense else (False, [vdn_adapter, fake_adapter]),
        }
        self.current = None

    def teacher_mode(self, role):
        return self.table[role][0]

    def adapters(self, role):
        return list(self.table[role][1])

    def set(self, role):
        teacher, adapters = self.table[role]
        set_teacher_mode(self.model, teacher)
        set_active_adapters(self.model, adapters)
        self.current = role


class FewStepSchedule:
    """The NFE grid the generator is sampled on: the paired forward times
    (`MiniMaxH3Scheduler.timesteps`) and the sigma grids (`.sigmas`) for both shifts."""

    def __init__(self, num_steps, video_shift, audio_shift):
        self.num_steps = int(num_steps)
        self.video_shift, self.audio_shift = float(video_shift), float(audio_shift)
        self.t_v, self.t_a = few_step_timesteps(num_steps, video_shift, audio_shift)
        self.sigma_v, self.sigma_a = few_step_sigmas(num_steps, video_shift, audio_shift)

    def times(self, index):
        return float(self.t_v[index]), float(self.t_a[index])

    def step(self, index, video_rows, audio_rows, velocity_v, velocity_a):
        """One Euler step from grid point `index` (== scheduler.step at that index)."""
        video_rows = euler_blend(video_rows, velocity_v, self.t_v[index],
                                 self.sigma_v[index], self.sigma_v[index + 1])
        audio_rows = euler_blend(audio_rows, velocity_a, self.t_a[index],
                                 self.sigma_a[index], self.sigma_a[index + 1])
        return video_rows, audio_rows

    def x0(self, index, video_rows, audio_rows, velocity_v, velocity_a):
        """The x0 prediction at grid point `index` (the scheduler's `denoised`)."""
        return (x0_from_velocity(video_rows, velocity_v, self.t_v[index]),
                x0_from_velocity(audio_rows, velocity_a, self.t_a[index]))


def sample_rollout_index(generator, num_steps):
    """DMD2 backward simulation: the ONE grid step that gets a gradient, uniform."""
    return int(torch.randint(num_steps, (), generator=generator))


def shared_rollout_index(seed, iteration, num_steps):
    """The rollout index for one iteration, identical on EVERY rank and a pure function
    of (seed, iteration) so a resume reproduces it. It must not come from a per-rank
    stream: k decides how many forwards -- hence how many FSDP2 all-gathers -- a rank
    issues, and ranks that disagree on that count deadlock in NCCL."""
    generator = torch.Generator().manual_seed(int(seed) * 100031 + int(iteration))
    return sample_rollout_index(generator, num_steps)


@torch.no_grad()
def rollout(model, roles, schedule, packed, video_rows, audio_rows, steps):
    """The generator's own first `steps` Euler steps from the given (pure-noise) rows,
    gradient-free: the sampler the student actually runs at inference, so step k's
    input is on-policy rather than a forward-noised real clip."""
    roles.set(GENERATOR)
    for index in range(steps):
        t_v, t_a = schedule.times(index)
        velocity_v, velocity_a = model(**packed.inputs(video_rows, audio_rows, t_v, t_a))
        video_rows, audio_rows = schedule.step(
            index, video_rows, audio_rows, velocity_v[0].float(), velocity_a[0].float())
    return video_rows, audio_rows


def generator_x0(model, roles, schedule, packed, video_rows, audio_rows, index,
                 offload_activations, build_graph=True):
    """Grid step `index` of the generator -> its x0 prediction. With `build_graph` the
    forward is the graph-building student forward (saved tensors offloaded); without,
    a plain no_grad forward (the fake warm-up only needs the sample)."""
    roles.set(GENERATOR)
    t_v, t_a = schedule.times(index)
    inputs = packed.inputs(video_rows, audio_rows, t_v, t_a)
    if build_graph:
        velocity_v, velocity_a = fs.student_forward(model, inputs, offload_activations)
    else:
        with torch.no_grad():
            velocity_v, velocity_a = model(**inputs)
    return schedule.x0(index, video_rows, audio_rows,
                       velocity_v[0].float(), velocity_a[0].float())


def noised(x0, t, noise):
    """Forward process at t: x_t = t*x0 + (1-t)*eps."""
    return t * x0 + (1.0 - t) * noise


@torch.no_grad()
def score_x0(model, roles, role, packed, x_v, x_a, t_v, t_a):
    """The real or fake score's x0 prediction on (x_t, t), no graph, memory handed
    back afterwards so the next forward's peak does not stack on this one's.

    Restores the role that was live on entry. Activation checkpointing RECOMPUTES the
    generator's blocks during its backward with whatever teacher_mode / adapters are
    set at that moment, and the score forwards necessarily run between the generator's
    forward and its backward. Left in the fake role, that backward dies with a
    CheckpointError -- and had the shapes happened to match, it would have been a
    silently wrong gradient instead."""
    previous = roles.current
    roles.set(role)
    try:
        velocity_v, velocity_a = model(**packed.inputs(x_v, x_a, t_v, t_a))
        x0_v = x0_from_velocity(x_v, velocity_v[0].float(), t_v)
        x0_a = x0_from_velocity(x_a, velocity_a[0].float(), t_a)
    finally:
        if previous is not None:
            roles.set(previous)
        from src.utils.device import empty_cache
        empty_cache(x_v.device)
    return x0_v, x0_a


def require_role(roles, role, what):
    """Refuse a backward whose checkpoint recompute would run under the wrong role."""
    if roles.current != role:
        raise RuntimeError(f"{what} needs the {role!r} role live for the checkpoint "
                           f"recompute, but {roles.current!r} is active")


def distribution_matching_loss(x0_g, x0_r, x0_f):
    """DMD's generator gradient for one modality,

        g = (x0_fake - x0_real) / mean|x0_g - x0_real|,

    packaged as a loss whose gradient w.r.t. x0_g is exactly g / numel (the 0.5 * mean
    MSE against the detached x0_g - g). Nothing else in the graph carries gradient.
    Returns (loss, normaliser, RMS of x0_fake - x0_real)."""
    with torch.no_grad():
        x0_g_detached = x0_g.detach()
        normaliser = (x0_g_detached - x0_r).abs().mean()
        g = (x0_f - x0_r) / normaliser.clamp_min(1e-6)
        target = x0_g_detached - g
        gap = (x0_f - x0_r).pow(2).mean().sqrt()
    loss = 0.5 * F.mse_loss(x0_g, target)
    return loss, normaliser, gap


def fake_regression_loss(velocity_pred, x0, noise):
    """Flow-matching regression of the fake score onto the generator's samples: the
    target is the data-ward velocity x0 - eps of the (x0, eps) pair that made x_t."""
    return F.mse_loss(velocity_pred, x0 - noise)
