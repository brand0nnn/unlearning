#!/bin/bash
# Stage 3, the "en" row: relearn the en-unlearned model in all five languages.
# Five independent jobs, ~3.2h each. See submit_stage3.sh for the preflight checks and
# for why this is one row at a time (storage) and five jobs rather than one (failure
# isolation + queue backfill).
#
#     bash studies/learn_french/slurm/submit_stage3_en.sh          # 3 epochs (default)
#     bash studies/learn_french/slurm/submit_stage3_en.sh 4        # override
exec bash "$(dirname "$0")/submit_stage3.sh" en "$@"
