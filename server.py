"""
Heightmap Studio — Minimal static file server.

All processing now runs client-side in Web Workers.
This server only serves static files.

Run with:  python server.py
Open:      http://localhost:8000

Alternative (no dependencies):
  cd static && python -m http.server 8000
"""

import http.server
import os

PORT = 8000
STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")

os.chdir(STATIC_DIR)

handler = http.server.SimpleHTTPRequestHandler

print(f"\n  Heightmap Studio")
print(f"  http://localhost:{PORT}\n")

with http.server.HTTPServer(("0.0.0.0", PORT), handler) as httpd:
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down...")
