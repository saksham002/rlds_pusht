#!/bin/bash

#SBATCH --job-name=monitor_dataflow
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --gres=gpu:L40S:1
#SBATCH --nodes=1
#SBATCH --partition=general
#SBATCH --output=logs/monitor.out
#SBATCH --error=logs/monitor.err

source ~/miniconda3/etc/profile.d/conda.sh
conda activate rlds

PROJECT="cmu-aidm-v2"
REGION="europe-west4"
POLL_INTERVAL=60

echo "Monitor started at $(date)"
echo "Looking for active Dataflow job..."

JOB_ID=""
for i in $(seq 1 60); do
    JOB_ID=$(gcloud dataflow jobs list --project="$PROJECT" --region="$REGION" \
        --status=active --limit=1 --format="value(JOB_ID)" 2>/dev/null)
    if [ -n "$JOB_ID" ]; then
        break
    fi
    echo "No active job found, retrying... (attempt $i/60)"
    sleep 30
done

if [ -z "$JOB_ID" ]; then
    echo "ERROR: No active Dataflow job found after 30 minutes."
    exit 1
fi

echo "Monitoring job: $JOB_ID"
echo ""

while true; do
    echo "=== $(date '+%Y-%m-%d %H:%M:%S') ==="

    STATE=$(gcloud dataflow jobs describe "$JOB_ID" --project="$PROJECT" --region="$REGION" \
        --format="value(currentState)" 2>/dev/null)
    echo "State: $STATE"

    if [ "$STATE" = "JOB_STATE_FAILED" ] || [ "$STATE" = "JOB_STATE_CANCELLED" ] || [ "$STATE" = "JOB_STATE_DONE" ]; then
        echo "Job finished with state: $STATE"
        break
    fi

    # Counter values
    echo "--- Counters ---"
    gcloud alpha dataflow metrics list "$JOB_ID" \
        --project="$PROJECT" --region="$REGION" \
        --format="table(name.name,scalar)" 2>/dev/null \
        | grep -iE "episodes_|repos_completed|rate_limit"

    echo "--- Memory & Disk (per worker) ---"
    WORKER_INFO=$(gcloud compute instances list --project="$PROJECT" \
        --filter="labels.dataflow_job_id = $JOB_ID" \
        --format="csv[no-heading](name,zone.basename())" 2>/dev/null)
    if [ -n "$WORKER_INFO" ]; then
        while IFS=',' read -r INSTANCE ZONE; do
            STATS=$(gcloud compute ssh "$INSTANCE" --zone="$ZONE" \
                --project="$PROJECT" \
                --command="free -m | awk '/^Mem:/{printf \"%d %d\", \$3, \$2}'; echo; df -BM /mnt/stateful_partition 2>/dev/null | awk 'NR==2{gsub(/M/,\"\"); printf \"%d %d\", \$3, \$2}'" \
                -- -o StrictHostKeyChecking=no -o ConnectTimeout=5 < /dev/null 2>/dev/null)
            if [ -n "$STATS" ]; then
                MEM_LINE=$(echo "$STATS" | sed -n '1p')
                DISK_LINE=$(echo "$STATS" | sed -n '2p')
                MEM_USED=$(echo "$MEM_LINE" | awk '{print $1}')
                MEM_TOTAL=$(echo "$MEM_LINE" | awk '{print $2}')
                MEM_PCT=$(echo "$MEM_USED * 100 / $MEM_TOTAL" | bc)
                if [ -n "$DISK_LINE" ]; then
                    DISK_USED=$(echo "$DISK_LINE" | awk '{print $1}')
                    DISK_TOTAL=$(echo "$DISK_LINE" | awk '{print $2}')
                    DISK_PCT=$(echo "$DISK_USED * 100 / $DISK_TOTAL" | bc)
                    printf "  %s (%s): mem %dMB/%dMB (%d%%)  disk %dMB/%dMB (%d%%)\n" \
                        "${INSTANCE##*-}" "$ZONE" "$MEM_USED" "$MEM_TOTAL" "$MEM_PCT" \
                        "$DISK_USED" "$DISK_TOTAL" "$DISK_PCT"
                else
                    printf "  %s (%s): mem %dMB/%dMB (%d%%)  disk N/A\n" \
                        "${INSTANCE##*-}" "$ZONE" "$MEM_USED" "$MEM_TOTAL" "$MEM_PCT"
                fi
            fi
        done <<< "$WORKER_INFO"
    else
        echo "  (no workers found)"
    fi

    # OOM check
    OOM=$(gcloud logging read \
        "resource.type=\"dataflow_step\" AND resource.labels.job_id=\"$JOB_ID\" AND (jsonPayload.message=~\"Detected.*OOM\" OR jsonPayload.message=~\"Out of memory: Killed\" OR jsonPayload.message=~\"oom_reaper\")" \
        --project="$PROJECT" --limit=1 --format="value(jsonPayload.message)" 2>/dev/null)
    if [ -n "$OOM" ]; then
        echo "!!! OOM DETECTED: $OOM"
    fi

    # Norm stats progress
    NORM_COUNT=$(gsutil ls gs://saksham-euw4/robocoin/norm_stats/*/norm_stats.json 2>/dev/null | wc -l)
    echo "Norm stats repos: $NORM_COUNT"

    echo ""
    sleep "$POLL_INTERVAL"
done

echo "Monitor finished at $(date)"
