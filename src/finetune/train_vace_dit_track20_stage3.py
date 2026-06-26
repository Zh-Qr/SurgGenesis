#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Compatibility entrypoint for stage-3 Track20 training.

The VACE/Wan2.1 trainer has been replaced by the DiffSynth-Studio
Wan2.2-TI2V-5B trainer.  This filename is kept so existing stage-3 commands
still land on the new backend.
"""

from train_diffsynth_wan_ti2v import main


if __name__ == "__main__":
    main(default_stage="stage3")
