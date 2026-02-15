#!/usr/bin/env python3
"""Main entry point for the xagent Web module

Usage:
    python -m xagent.web
    python -m xagent.web --host 0.0.0.0 --port 8000
    python -m xagent.web --reload --debug
"""

import argparse
import logging
import sys

import uvicorn
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


def setup_logging(debug: bool = False) -> None:
    """Configure logging"""
    level = logging.DEBUG if debug else logging.INFO

    # Basic logging configuration
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    if debug:
        # Debug mode: configure detailed logging for specific modules
        # XAgent modules - set to DEBUG for all submodules
        logging.getLogger("xagent.core").setLevel(logging.DEBUG)
        logging.getLogger("xagent.web").setLevel(logging.DEBUG)
        logging.getLogger("xagent.entrypoint").setLevel(logging.DEBUG)
        logging.getLogger("xagent.skills").setLevel(logging.DEBUG)

        # LangChain/LangGraph detailed logging
        logging.getLogger("langchain").setLevel(logging.DEBUG)
        logging.getLogger("langgraph").setLevel(logging.DEBUG)
        logging.getLogger("langchain_core").setLevel(logging.DEBUG)

        # Suppress verbose logs from third-party libraries
        logging.getLogger("aiohttp").setLevel(logging.WARNING)
        logging.getLogger("sqlalchemy").setLevel(logging.WARNING)
        logging.getLogger("urllib3").setLevel(logging.WARNING)
        logging.getLogger("uvicorn.access").setLevel(logging.WARNING)

        print("🐛 Debug mode enabled")
        print("🔍 LLM responses and tool call details will be logged")
        print("-" * 50)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Start xagent Web service",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python -m xagent.web                     # Start with default configuration
    python -m xagent.web --port 8001         # Specify port
    python -m xagent.web --reload --debug    # Development mode + debug mode
    python -m xagent.web --host 0.0.0.0      # Listen on all interfaces
    python -m xagent.web --debug             # Enable verbose logging (LLM responses, etc.)
        """,
    )

    parser.add_argument(
        "--host", default="127.0.0.1", help="Server host address (default: 127.0.0.1)"
    )
    parser.add_argument(
        "--port", type=int, default=8000, help="Server port (default: 8000)"
    )
    parser.add_argument(
        "--reload", action="store_true", help="Enable auto-reload (development mode)"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode (verbose logging, including LLM responses)",
    )
    parser.add_argument(
        "--log-level",
        choices=["debug", "info", "warning", "error"],
        default="info",
        help="Log level (default: info)",
    )

    return parser.parse_args()


def main() -> None:
    """Main function"""
    args = parse_args()

    # Configure logging
    setup_logging(args.debug)
    logger = logging.getLogger(__name__)

    logger.info("🚀 Starting xagent Web service...")
    logger.info(f"📍 Service URL: http://{args.host}:{args.port}")

    if args.reload:
        logger.info("🔄 Development mode: auto-reload enabled")

    if args.debug:
        logger.info("🐛 Debug mode: verbose logging enabled")

    try:
        uvicorn.run(
            "xagent.web.app:app",
            host=args.host,
            port=args.port,
            reload=args.reload,
            log_level=args.log_level,
        )
    except KeyboardInterrupt:
        logger.info("⏹️  Service stopped")
    except Exception as e:
        logger.error(f"❌ Startup failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
