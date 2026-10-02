"""
writer.py — the WRITER of the PROBE-GATE system (gate_mode="probe").

On a trigger from the ingester's yes/no probe gate it takes an MVCC SNAPSHOT of
the shared cache (an independent clone) and free-generates ONE short answer line
on the clone: it holds no lock, never blocks the ingester, and its tokens are
never written back (single-writer invariant preserved).

This is the `main`-branch writer restored for the probe-vs-controller head-to-
head, adapted to the current backend: forward() returns GPU-resident logits, so
sampling happens on GPU (shared `_sample` from controller.py) and only the token
id crosses the bus. In deterministic lockstep the ingester blocks on
writer_q.join(), so the write happens synchronously at the trigger frame.
"""
import queue
import time

import torch

from util import log
from controller import _sample, _cache_to


def _looping(ids, window):
    if len(ids) >= 3 and ids[-1] == ids[-2] == ids[-3]:
        return True
    if len(ids) >= 4 and ids[-2:] == ids[-4:-2]:
        return True
    return len(ids) >= window and len(set(ids[-window:])) <= 2


def writer_thread(cfg, mgr, writer_q, stop, prof=None, evaluator=None, wb=None,
                  feed_done=None):
    b = wb if wb is not None else mgr.b
    cross_gpu = b.device != mgr.b.device
    try:
        gen = torch.Generator(device=b.device).manual_seed(int(cfg.writer_seed))
    except (RuntimeError, TypeError):
        gen = torch.Generator().manual_seed(int(cfg.writer_seed))
    # per-sample task fill: {event} = the monitored condition
    writer_prompt = cfg.writer_prompt.replace("{event}", cfg.event or cfg.instruction)

    # Exit gate: the writer must OUTLIVE the ingester's feed. On a late fire the
    # ingester (deterministic) does writer_q.put(vt); writer_q.join(), so if the
    # writer exited on the shared `stop` (which the ENCODER sets at end-of-stream,
    # while the ingester is still draining buffered frames) that join() would
    # block forever -> the whole sample hangs. Key the exit off feed_done, which
    # the ingester sets ONLY after it has finished feeding: every put() is then
    # guaranteed a live servicer. (Fallback to `stop` if feed_done wasn't wired.)
    done = feed_done if feed_done is not None else stop
    while not done.is_set() or not writer_q.empty():
        try:
            vt = writer_q.get(timeout=0.5)
        except queue.Empty:
            continue
        # task_done() MUST run for every get() (even on failure) or the ingester's
        # writer_q.join() deadlocks; wrap the whole item in try/finally.
        try:
            t_trig = time.time()

            # MVCC snapshot: independent clone + logical position. No lock held below.
            cache, pos, phys = mgr.snapshot_clone()
            if cross_gpu:
                cache = _cache_to(cache, b.device)

            def step(embeds):
                nonlocal pos, phys, cache
                logits, cache = b.forward(embeds, cache, pos_start=pos, phys_start=phys)
                pos += embeds.shape[1]
                phys += embeds.shape[1]
                return logits

            logits = step(b.embed_text(writer_prompt + cfg.writer_cue))
            ids = []
            for _ in range(cfg.writer_max_tokens):
                if not ids:                      # mask EOS only for the FIRST token (the
                    logits[b.eos_id] = float("-inf")   # raw-splice immediate-EOS bug)
                tok_id = _sample(logits, ids, cfg, gen)
                if tok_id == b.eos_id or _looping(ids, cfg.writer_repeat_window):
                    break
                # one line only: stop on any surface newline
                if tok_id in b.newline_ids or "\n" in b.tok.decode([tok_id]):
                    break
                ids.append(tok_id)
                logits = step(b.embed_token(tok_id))

            total = time.time() - t_trig
            text = b.decode(ids)
            if prof is not None:
                prof.observe("writer_total_s", total)
                prof.observe("writer_tokens", len(ids))
            if evaluator is not None:
                evaluator.record_write(vt, text, total)
            log("WRITER", vt, f"[{cfg.video_id or '?'}] \U0001F4E2  {text!r}")
        except Exception as e:                   # never let a generation error strand join()
            log("writer", vt, f"generation FAILED: {e!r}")
        finally:
            writer_q.task_done()                 # releases the ingester's join() (deterministic)
    log("writer", 0.0, "writer stopped")
