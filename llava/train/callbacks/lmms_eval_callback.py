# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
"""TrainerCallback that queues a SLURM eval job after each checkpoint save.

The callback runs *inside* the training container on a compute node, but the
ADLR `submit_job` wrapper (used by scripts/slurm/submit_eval.sh) lives on a
different lustre mount (/lustre/fsw) that the container can't see/exec. So
instead of calling Popen directly, the callback drops a marker file under
`runs/eval_queue/` and a host-side watcher
(scripts/slurm/eval_queue_watcher.sh) actually submits the SLURM job.

Marker filename: `<run>__step<N>__<tasks_with_underscores>.marker`
Marker contents (one key=value per line):
    ckpt_dir=<absolute path to checkpoint-N>
    tasks=<comma-separated lmms-eval task list>
"""
import os

import transformers

# Project-relative dir where markers are written. Watcher polls this same path.
EVAL_QUEUE_DIR = "runs/eval_queue"


class LmmsEvalCallback(transformers.TrainerCallback):
    """Queue an lmms-eval SLURM job after every saved checkpoint.

    Args:
        tasks: comma-separated lmms-eval task names (e.g. "r4d_bench" or
            "r4d_bench,stibench"). Written verbatim into the marker; the
            watcher passes it through to submit_eval.sh.
    """

    def __init__(self, tasks: str):
        self.tasks = tasks

    def on_save(self, args, state, control, **kwargs):
        # Only global rank 0 queues, otherwise every rank drops an identical
        # marker. is_world_process_zero (global) — not is_local_process_zero
        # (per-node), which would still drop N copies across N nodes.
        if not state.is_world_process_zero:
            return control

        ckpt_dir = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
        if not os.path.isdir(ckpt_dir):
            # Save still in progress / something else wrote a different layout.
            print(f"[LmmsEvalCallback] skip: checkpoint dir not found at {ckpt_dir}", flush=True)
            return control

        # Derive a stable run name from output_dir (e.g. runs/train/<run>/model).
        run_name = os.path.basename(os.path.normpath(args.output_dir)) or "run"
        tasks_slug = self.tasks.replace(",", "_").replace("/", "_")
        marker_name = f"{run_name}__step{state.global_step}__{tasks_slug}.marker"

        try:
            os.makedirs(EVAL_QUEUE_DIR, exist_ok=True)
            marker_path = os.path.join(EVAL_QUEUE_DIR, marker_name)
            # Write atomically via tmp+rename so the watcher never sees a
            # partially-written marker.
            tmp_path = marker_path + ".tmp"
            with open(tmp_path, "w") as f:
                f.write(f"ckpt_dir={os.path.abspath(ckpt_dir)}\n")
                f.write(f"tasks={self.tasks}\n")
            os.replace(tmp_path, marker_path)
            print(
                f"[LmmsEvalCallback] queued eval marker {marker_path} tasks={self.tasks}",
                flush=True,
            )
        except Exception as e:
            # Never crash training because eval queuing failed.
            print(f"[LmmsEvalCallback] marker write failed: {e}", flush=True)

        return control
