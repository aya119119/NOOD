set -euo pipefail

PORT="${NOOD_RELAY_PORT:-5050}"
SHARED_DIR="${NOOD_SHARED_DIR:-/mnt/shared}"
INPUT_FILE="input.mp4"
RESULT_FILE="result.json"
STATUS_FILE=".nood_status"

echo "╔══════════════════════════════════════════════════╗"
echo "║  NOOD Relay Server                               ║"
echo "║  Port:   $PORT                                   ║"
echo "║  Shared: $SHARED_DIR                             ║"
echo "╚══════════════════════════════════════════════════╝"
echo ""

if [ ! -d "$SHARED_DIR" ]; then
    echo "[ERROR] Shared directory not found: $SHARED_DIR"
    echo "        Mount the Samba share first:"
    echo "        sudo mount -t cifs //10.12.124.79/Shared $SHARED_DIR -o username=guest"
    exit 1
fi

rm -f "$SHARED_DIR/$RESULT_FILE" "$SHARED_DIR/$STATUS_FILE"

python3 - "$PORT" "$SHARED_DIR" "$INPUT_FILE" "$RESULT_FILE" "$STATUS_FILE" << 'PYSERVER'
import sys
import os
import json
import cgi
import shutil
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path

PORT       = int(sys.argv[1])
SHARED_DIR = sys.argv[2]
INPUT_FILE = sys.argv[3]
RESULT_FILE= sys.argv[4]
STATUS_FILE= sys.argv[5]

class RelayHandler(BaseHTTPRequestHandler):

    def _cors_headers(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')

    def do_OPTIONS(self):
        self.send_response(204)
        self._cors_headers()
        self.end_headers()

    def do_GET(self):
        if self.path == '/status':
            self._handle_status()
        elif self.path == '/result':
            self._handle_result()
        elif self.path == '/health':
            self._respond_json(200, {"status": "ok"})
        else:
            self._respond_json(404, {"error": "Not found"})

    def do_POST(self):
        if self.path == '/upload':
            self._handle_upload()
        else:
            self._respond_json(404, {"error": "Not found"})


    def _handle_upload(self):
        try:
            content_type = self.headers.get('Content-Type', '')

            if 'multipart/form-data' in content_type:
                form = cgi.FieldStorage(
                    fp=self.rfile,
                    headers=self.headers,
                    environ={
                        'REQUEST_METHOD': 'POST',
                        'CONTENT_TYPE': content_type,
                    }
                )
                video_field = form['video']
                data = video_field.file.read()
            else:
                length = int(self.headers.get('Content-Length', 0))
                data = self.rfile.read(length)

            if not data:
                self._respond_json(400, {"error": "No video data received"})
                return

            dest = os.path.join(SHARED_DIR, INPUT_FILE)
            with open(dest, 'wb') as f:
                f.write(data)

            result_path = os.path.join(SHARED_DIR, RESULT_FILE)
            if os.path.exists(result_path):
                os.remove(result_path)

            status_path = os.path.join(SHARED_DIR, STATUS_FILE)
            with open(status_path, 'w') as f:
                f.write('pending')

            size_mb = len(data) / (1024 * 1024)
            print(f"  ✓ Video saved: {dest} ({size_mb:.1f} MB)")

            self._respond_json(200, {"ok": True, "size_mb": round(size_mb, 1)})

        except Exception as e:
            print(f"  [ERROR] Upload failed: {e}")
            self._respond_json(500, {"error": str(e)})

    def _handle_status(self):
        status_path = os.path.join(SHARED_DIR, STATUS_FILE)
        result_path = os.path.join(SHARED_DIR, RESULT_FILE)

        if os.path.exists(result_path):
            status = 'done'
        elif os.path.exists(status_path):
            try:
                with open(status_path) as f:
                    status = f.read().strip() or 'pending'
            except:
                status = 'pending'
        else:
            status = 'idle'

        self._respond_json(200, {"status": status})

    def _handle_result(self):
        result_path = os.path.join(SHARED_DIR, RESULT_FILE)

        if not os.path.exists(result_path):
            self._respond_json(404, {"error": "Result not ready"})
            return

        try:
            with open(result_path, 'r') as f:
                data = json.load(f)
            self._respond_json(200, data)
        except Exception as e:
            self._respond_json(500, {"error": f"Failed to read result: {e}"})


    def _respond_json(self, code, data):
        body = json.dumps(data, ensure_ascii=False).encode('utf-8')
        self.send_response(code)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Content-Length', str(len(body)))
        self._cors_headers()
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):
        print(f"  [{self.log_date_time_string()}] {format % args}")

if __name__ == '__main__':
    server = HTTPServer(('0.0.0.0', PORT), RelayHandler)
    print(f"  Relay server listening on http://0.0.0.0:{PORT}")
    print(f"  Endpoints:")
    print(f"    POST /upload  — upload a video")
    print(f"    GET  /status  — check processing status")
    print(f"    GET  /result  — fetch result JSON")
    print(f"    GET  /health  — server health check")
    print()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n  Relay server stopped.")
        server.server_close()
PYSERVER
