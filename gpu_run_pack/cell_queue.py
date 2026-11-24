"""Restart-safe parallel queue over (method, dataset) attribution cells.

DMKD revision. Each worker claims the next unclaimed cell with an O_EXCL lock
file, runs it as a separate `run_instance_attribution.py` process, and moves
on. A cell is done when its JSON exists; the runner itself is idempotent. On a
restart, stale locks (lock present, JSON absent, owning pid dead) are cleared,
so re-running the same command after a crash, sleep or power cut simply
continues.

    python3 gpu_run_pack/cell_queue.py --workers 6 --threads 2 \
        --methods attnmil,... --datasets letter_frost,... \
        --outdir results/instance_level_r1 --device cpu

Logs: <outdir>/logs/<method>__<dataset>.log and <outdir>/logs/queue.log.
"""

import argparse
import os
import subprocess
import sys
import time
from multiprocessing import Process

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _alive(pid):
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def _log(outdir, msg):
    line = f'{time.strftime("%Y-%m-%d %H:%M:%S")} {msg}'
    print(line, flush=True)
    with open(os.path.join(outdir, 'logs', 'queue.log'), 'a') as f:
        f.write(line + '\n')


def clear_stale(outdir, cells):
    for m, d in cells:
        lock = os.path.join(outdir, 'locks', f'{m}__{d}.lock')
        js = os.path.join(outdir, f'{m}__{d}.json')
        if os.path.exists(lock) and not os.path.exists(js):
            try:
                pid = int(open(lock).read().strip() or 0)
            except ValueError:
                pid = 0
            if not pid or not _alive(pid):
                os.remove(lock)
                _log(outdir, f'cleared stale lock {m}__{d}')


def worker(wid, cells, args):
    for m, d in cells:
        js = os.path.join(args.outdir, f'{m}__{d}.json')
        if os.path.exists(js):
            continue
        lock = os.path.join(args.outdir, 'locks', f'{m}__{d}.lock')
        try:
            fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            continue
        cmd = [sys.executable, 'run_instance_attribution.py',
               '--methods', m, '--datasets', d,
               '--outdir', args.outdir, '--device', args.device,
               '--threads', str(args.threads)] + args.extra
        logf = open(os.path.join(args.outdir, 'logs', f'{m}__{d}.log'), 'a')
        t0 = time.time()
        p = subprocess.Popen(cmd, cwd=ROOT, stdout=logf,
                             stderr=subprocess.STDOUT)
        os.write(fd, str(p.pid).encode())
        os.close(fd)
        _log(args.outdir, f'w{wid} START {m}__{d} pid={p.pid}')
        rc = p.wait()
        logf.close()
        ok = os.path.exists(js)
        _log(args.outdir, f'w{wid} {"DONE" if ok else "FAIL"} {m}__{d} '
                          f'rc={rc} {(time.time() - t0) / 60:.1f} min')
        if not ok and os.path.exists(lock):
            os.remove(lock)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--methods', required=True)
    ap.add_argument('--datasets', required=True)
    ap.add_argument('--outdir', required=True)
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--threads', type=int, default=2)
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--order', default='dataset',
                    choices=['dataset', 'method'],
                    help='claim cells dataset-major (all methods of the first '
                         'dataset first) or method-major')
    args, extra = ap.parse_known_args()
    args.extra = extra
    args.outdir = os.path.abspath(args.outdir) if not os.path.isabs(
        args.outdir) else args.outdir
    os.makedirs(os.path.join(args.outdir, 'logs'), exist_ok=True)
    os.makedirs(os.path.join(args.outdir, 'locks'), exist_ok=True)

    methods = [x for x in args.methods.split(',') if x]
    datasets = [x for x in args.datasets.split(',') if x]
    if args.order == 'dataset':
        cells = [(m, d) for d in datasets for m in methods]
    else:
        cells = [(m, d) for m in methods for d in datasets]
    clear_stale(args.outdir, cells)
    todo = [c for c in cells
            if not os.path.exists(os.path.join(args.outdir, f'{c[0]}__{c[1]}.json'))]
    _log(args.outdir, f'queue start: {len(todo)}/{len(cells)} cells to run, '
                      f'{args.workers} workers x {args.threads} threads')
    procs = [Process(target=worker, args=(i, cells, args))
             for i in range(args.workers)]
    for p in procs:
        p.start()
    for p in procs:
        p.join()
    done = sum(os.path.exists(os.path.join(args.outdir, f'{m}__{d}.json'))
               for m, d in cells)
    _log(args.outdir, f'queue end: {done}/{len(cells)} cells complete')


if __name__ == '__main__':
    main()
