"""Control only the supplement collection, using shared pause/OS-lock helpers."""
import argparse
import os
from pathlib import Path
import time
from .embedding_control import request_pause,worker_active
from .gapfill import directory
from .paths import JOB,contained


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,default=directory(JOB)/'embeddings/georsclip-vith14-s256-st192')
    parser.add_argument('--check-running',action='store_true')
    parser.add_argument('--keep-awake',action='store_true')
    args=parser.parse_args(argv);output=contained(directory(JOB)/'embeddings',args.output_dir)
    if args.keep_awake:
        deadline=time.monotonic()+60
        while not worker_active(output) and time.monotonic()<deadline:time.sleep(.25)
        if os.name!='nt' or not worker_active(output):return 0
        import ctypes
        kernel=ctypes.WinDLL('kernel32')
        kernel.SetThreadExecutionState(0x80000001)
        try:
            while worker_active(output):time.sleep(5)
        finally:kernel.SetThreadExecutionState(0x80000000)
        return 0
    if args.check_running:return 0 if worker_active(output) else 1
    request_pause(output);return 0


if __name__=='__main__':raise SystemExit(main())
