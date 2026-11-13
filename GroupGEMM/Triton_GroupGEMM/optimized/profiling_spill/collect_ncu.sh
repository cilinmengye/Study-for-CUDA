#!/usr/bin/env bash
# Requires GPU performance-counter access. Does not change driver permissions.
set -euo pipefail
profile_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
for config in A B; do
    PYTHONDONTWRITEBYTECODE=1 ncu \
        --profile-from-start off --clock-control none --cache-control all \
        --launch-count 1 --section LaunchStats --section Occupancy \
        --section SpeedOfLight --section MemoryWorkloadAnalysis_Tables \
        --section SchedulerStats --section WarpStateStats \
        --section InstructionStats --section ComputeWorkloadAnalysis \
        --export "$profile_dir/${config}_profile" --force-overwrite \
        python -u "$profile_dir/analyze.py" --profile "$config"
done
