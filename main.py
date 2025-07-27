"""
Smart Campus Surveillance - Main Entry Point
Real-time video processing with AI-based threat detection
"""

import os
import sys

# Add project modules to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


def main():
    # Entry point for launching the dashboard
    print("=" * 60)
    print("SMART CAMPUS SURVEILLANCE SYSTEM")
    print("=" * 60)
    print("Launching surveillance dashboard...")
    print("=" * 60)
    os.system("streamlit run dashboard/app.py")


if __name__ == "__main__":
    main()