"""CPU-only setup helpers for the rank-serialized experimental TP entrypoint."""
from contextlib import contextmanager
from functools import wraps
import os
from pathlib import Path
import time

import torch


def assembly_thread_count():
    """Use at most eight affinity/quota CPUs, independently of torchrun's OMP=1.

    Check visible cgroup v1/v2 quota ancestors too; affinity alone can overestimate
    the CPU budget in containers. No environment variables are changed.
    """
    try:
        available = len(os.sched_getaffinity(0))
    except AttributeError:
        available = os.cpu_count() or 1
    roots = [(Path('/sys/fs/cgroup'), 'cpu.max'),
             (Path('/sys/fs/cgroup/cpu'), 'cpu.cfs_quota_us'),
             (Path('/sys/fs/cgroup/cpu,cpuacct'), 'cpu.cfs_quota_us')]
    try:
        groups = Path('/proc/self/cgroup').read_text().splitlines()
    except OSError:
        groups = []
    limits = []
    for root, filename in roots:
        paths = {root}
        for group in groups:
            _, controllers, relative = group.split(':', 2)
            if not controllers or 'cpu' in controllers.split(','):
                current = root / relative.lstrip('/')
                if '..' not in current.parts:
                    paths.update(p for p in (current, *current.parents)
                                 if p == root or root in p.parents)
        for path in paths:
            try:
                if filename == 'cpu.max':
                    quota, period = (path / filename).read_text().split()
                    if quota == 'max':
                        continue
                else:
                    quota = (path / filename).read_text().strip()
                    period = (path / 'cpu.cfs_period_us').read_text().strip()
                if int(quota) > 0 and int(period) > 0:
                    limits.append(max(1, int(quota) // int(period)))
            except (OSError, ValueError):
                continue
    return max(1, min(8, available, *limits))


@contextmanager
def cpu_assembly_threads(record):
    """Temporarily increase intra-op threads; restore even if assembly/sharding fails."""
    old = torch.get_num_threads()
    new = assembly_thread_count()
    record.update(old_threads=old, assembly_threads=new)
    try:
        if new != old:
            torch.set_num_threads(new)
        yield
    finally:
        if torch.get_num_threads() != old:
            torch.set_num_threads(old)
        record['restored_threads'] = torch.get_num_threads()


@contextmanager
def time_cpu_assembly(assembly, record):
    """Time shared assembly calls without changing their arguments or arithmetic.

    Scoped wrappers are installed only during TP's serialized CPU build, on its
    main Python thread, and restored in finally. No wrappers remain for sampling,
    onload or decoding. Stage times include lazy page faults, not exclusive I/O.
    """
    originals = {}
    stages = record.setdefault('stages_seconds', {})

    def timed(name, fn):
        @wraps(fn)
        def call(*args, **kwargs):
            started = time.perf_counter()
            try:
                return fn(*args, **kwargs)
            finally:
                stages[name] = stages.get(name, 0.0) + time.perf_counter() - started
        return call

    try:
        for name in ('load_checkpoint', 'load_models', 'apply_hybrid_attention_transform',
                     'load_model_weights', 'merge_lora_state'):
            originals[name] = getattr(assembly, name)
            setattr(assembly, name, timed(name, originals[name]))
        yield
    finally:
        for name, fn in originals.items():
            setattr(assembly, name, fn)
