#!/usr/bin/env python3
"""
MarketMind AI — Indian Markets Intelligence Platform
Browser-based entry point: starts FastAPI server and opens the browser.
"""

import sys
import os
import webbrowser
import threading
import time
import signal
import logging

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    datefmt='%H:%M:%S',
)
logger = logging.getLogger(__name__)

PORT = 8000
HOST = "127.0.0.1"


def open_browser():
    """Wait for the server to start then open the browser."""
    time.sleep(2.0)
    url = f"http://{HOST}:{PORT}"
    logger.info(f"Opening browser → {url}")
    webbrowser.open(url)


def main():
    signal.signal(signal.SIGINT, signal.SIG_DFL)

    # Open browser in background after short delay
    t = threading.Thread(target=open_browser, daemon=True)
    t.start()

    import uvicorn
    logger.info(f"Starting MarketMind AI on http://{HOST}:{PORT}")
    uvicorn.run(
        "server:app",
        host=HOST,
        port=PORT,
        reload=False,
        log_level="warning",
    )


if __name__ == "__main__":
    main()
