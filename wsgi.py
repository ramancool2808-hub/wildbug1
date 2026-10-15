"""
wsgi.py - Production WSGI entry point for MachinaSense web app
Run with: gunicorn wsgi:app --bind 0.0.0.0:$PORT --workers 1 --threads 4 --timeout 120
Or dev:   python web/app.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from web.app import app, start_training_once   # noqa: E402

# web.app already starts training on import; this call is a harmless no-op safeguard.
start_training_once()
