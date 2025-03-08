# serve_tests.py
from http.server import HTTPServer, SimpleHTTPRequestHandler
import os

# Change to the directory containing wheel and test files
os.chdir("dist")

# Start server
port = 8000
server = HTTPServer(("localhost", port), SimpleHTTPRequestHandler)
print(f"Serving at http://localhost:{port}")
server.serve_forever()
