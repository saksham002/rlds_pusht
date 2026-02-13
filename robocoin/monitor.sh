#!/bin/bash
JOB_ID="2026-02-08_14_57_53-4200502868882200614"
PROJECT="cmu-aidm-v2"
REGION="europe-west4"

while true; do
    echo "=== $(date '+%Y-%m-%d %H:%M:%S') ==="

    # Job state
    STATE=$(gcloud dataflow jobs describe "$JOB_ID" --project="$PROJECT" --region="$REGION" --format="value(currentState)" 2>/dev/null)
    echo "Job state: $STATE"

    if [ "$STATE" = "JOB_STATE_FAILED" ] || [ "$STATE" = "JOB_STATE_CANCELLED" ] || [ "$STATE" = "JOB_STATE_DONE" ]; then
        echo "Job finished with state: $STATE"
        break
    fi

    # Beam counters
    echo "--- Beam Counters ---"
    gcloud alpha dataflow metrics list "$JOB_ID" --project="$PROJECT" --region="$REGION" --format="table(name.context.original_name,scalar)" 2>/dev/null | grep -E "robocoin|ProcessRepos"

    # Memory
    echo "--- Memory ---"
    gcloud logging read "resource.type=\"dataflow_step\" AND resource.labels.job_id=\"$JOB_ID\" AND jsonPayload.message=~\"cgroup\"" --project="$PROJECT" --limit=1 --format="value(timestamp,jsonPayload.message)" 2>/dev/null

    # OOM check
    OOM=$(gcloud logging read "resource.type=\"dataflow_step\" AND resource.labels.job_id=\"$JOB_ID\" AND (jsonPayload.message=~\"Detected.*OOM\" OR jsonPayload.message=~\"Out of memory: Killed\" OR jsonPayload.message=~\"oom_reaper\")" --project="$PROJECT" --limit=1 --format="value(jsonPayload.message)" 2>/dev/null)
    if [ -n "$OOM" ]; then
        echo "!!! OOM DETECTED: $OOM"
    fi

    # Norm stats files
    echo "--- Norm Stats Files ---"
    gsutil ls gs://saksham-euw4/robocoin/norm_stats/*/norm_stats.json 2>/dev/null | wc -l | xargs -I{} echo "Completed repos (norm_stats): {}"

    echo ""
    sleep 60
done
