set -euo pipefail

SHARED_DIR="${NOOD_SHARED_DIR:-/mnt/shared}"
SERVICE_URL="${NOOD_SERVICE_URL:-http://localhost:5050}"
INPUT_FILE="input.mp4"
RESULT_FILE="result.json"
STATUS_FILE=".nood_status"
POLL_INTERVAL=2  # seconds

echo "╔══════════════════════════════════════════════════╗"
echo "║  NOOD Watcher — Service Client Mode              ║"
echo "║  Shared:  $SHARED_DIR                            ║"
echo "║  Service: $SERVICE_URL                           ║"
echo "╚══════════════════════════════════════════════════╝"
echo ""

if [ ! -d "$SHARED_DIR" ]; then
    echo "[ERROR] Shared directory not found: $SHARED_DIR"
    exit 1
fi

# Check that the service is running
echo "  Checking service health..."
if ! curl -sf "$SERVICE_URL/health" > /dev/null 2>&1; then
    echo "[ERROR] NOOD service is not running at $SERVICE_URL"
    echo "        Start it with: bash nood_service.sh"
    exit 1
fi
echo "  ✓ Service is running"
echo "  Watching $SHARED_DIR for new videos..."
echo ""

while true; do
    INPUT_PATH="$SHARED_DIR/$INPUT_FILE"
    RESULT_PATH="$SHARED_DIR/$RESULT_FILE"
    STATUS_PATH="$SHARED_DIR/$STATUS_FILE"

    if [ -f "$INPUT_PATH" ]; then
        FILE_SIZE=$(stat -c%s "$INPUT_PATH" 2>/dev/null || echo 0)

        if [ "$FILE_SIZE" -gt 0 ]; then
            echo "══ [$(date '+%H:%M:%S')] New video detected: $INPUT_PATH ($FILE_SIZE bytes)"

            echo "processing" > "$STATUS_PATH"

            # Submit to the service
            echo "  Submitting to analysis service..."
            RESPONSE=$(curl -sf -X POST "$SERVICE_URL/analyze" \
                -H "Content-Type: application/json" \
                -d "{\"video_path\": \"$INPUT_PATH\"}" 2>&1) || {
                echo "  ✗ Failed to submit job"
                echo "error" > "$STATUS_PATH"
                rm -f "$INPUT_PATH"
                continue
            }

            JOB_ID=$(echo "$RESPONSE" | python3 -c "import sys,json; print(json.load(sys.stdin).get('job_id',''))" 2>/dev/null)

            if [ -z "$JOB_ID" ]; then
                echo "  ✗ No job ID returned"
                echo "error" > "$STATUS_PATH"
                rm -f "$INPUT_PATH"
                continue
            fi

            echo "  Job submitted: $JOB_ID"

            # Poll for completion
            while true; do
                STATUS=$(curl -sf "$SERVICE_URL/jobs/$JOB_ID" 2>/dev/null | \
                    python3 -c "import sys,json; print(json.load(sys.stdin).get('status','unknown'))" 2>/dev/null || echo "error")

                if [ "$STATUS" = "done" ]; then
                    # Fetch result and write to shared dir
                    curl -sf "$SERVICE_URL/jobs/$JOB_ID/result" > "$RESULT_PATH" 2>/dev/null
                    echo "  ✓ Analysis complete: $RESULT_PATH"
                    echo "done" > "$STATUS_PATH"
                    break
                elif [ "$STATUS" = "failed" ]; then
                    echo "  ✗ Analysis failed!"
                    echo "error" > "$STATUS_PATH"
                    break
                fi

                sleep "$POLL_INTERVAL"
            done

            rm -f "$INPUT_PATH"
            echo "  Cleaned up input file. Waiting for next video..."
            echo ""
        fi
    fi

    sleep "$POLL_INTERVAL"
done
