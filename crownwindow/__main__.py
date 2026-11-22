"""Run the local CrownWindow application with a bounded HTTP worker."""

import os

from waitress import serve

from .http import create_app


def main() -> None:
    host = os.environ.get("HOST", "127.0.0.1")
    port = int(os.environ.get("PORT", "19963"))
    serve(create_app(), host=host, port=port, threads=5, channel_timeout=70,
          max_request_body_size=70 * 1024 * 1024)


if __name__ == "__main__":
    main()
