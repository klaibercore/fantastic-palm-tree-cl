#!/usr/bin/env python3
"""
Utility script to start/stop TensorBoard manually.
"""

import argparse
import subprocess
import webbrowser
import time
import socket
import logging
import sys
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def is_port_available(port: int) -> bool:
    """Check if a port is available."""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        result = sock.connect_ex(('localhost', port))
        sock.close()
        return result != 0
    except Exception:
        return False


def start_tensorboard(logdir: str, port: int, open_browser: bool = True):
    """Start TensorBoard server with multiple fallback methods."""
    
    # If logdir is a specific run directory, ensure it exists
    # If it's the base tensorboard directory, check for runs inside
    if not os.path.exists(logdir):
        # Check if this is the base logs directory and we should look for tensorboard subdir
        base_logs_dir = os.path.dirname(logdir) if os.path.basename(logdir) == "tensorboard" else None
        if base_logs_dir and os.path.exists(os.path.join(base_logs_dir, "tensorboard")):
            logdir = os.path.join(base_logs_dir, "tensorboard")
        elif not os.path.exists(logdir):
            logger.error(f"Log directory does not exist: {logdir}")
            return False
    
    # If it's the base tensorboard directory, look for the latest run
    if os.path.basename(logdir) == "tensorboard" and os.path.isdir(logdir):
        runs = [d for d in os.listdir(logdir) if d.startswith("run_") and os.path.isdir(os.path.join(logdir, d))]
        if runs:
            latest_run = sorted(runs)[-1]  # Get the latest run
            logdir = os.path.join(logdir, latest_run)
            logger.info(f"Using latest TensorBoard run: {logdir}")
        else:
            logger.warning(f"No runs found in {logdir}, using directory as-is")
    
    if not is_port_available(port):
        logger.info(f"TensorBoard already running on http://localhost:{port}")
        if open_browser:
            webbrowser.open(f"http://localhost:{port}")
        return True
    
    logger.info(f"Starting TensorBoard on http://localhost:{port}")
    logger.info(f"Log directory: {logdir}")
    
    # Try different methods to start TensorBoard
    methods = [
        # Method 1: Python module with tensorboard.main
        {
            'name': 'Python module (tensorboard.main)',
            'cmd': [sys.executable, '-m', 'tensorboard.main', '--logdir', logdir, '--port', str(port), '--host', 'localhost']
        },
        # Method 2: Direct tensorboard command
        {
            'name': 'Direct tensorboard command',
            'cmd': ['tensorboard', '--logdir', logdir, '--port', str(port), '--host', 'localhost']
        },
        # Method 3: Python module with just tensorboard
        {
            'name': 'Python module (tensorboard)',
            'cmd': [sys.executable, '-m', 'tensorboard', '--logdir', logdir, '--port', str(port), '--host', 'localhost']
        }
    ]
    
    for method in methods:
        try:
            logger.info(f"Trying TensorBoard method: {method['name']}")
            
            # Test if command is available
            test_result = subprocess.run(method['cmd'] + ['--help'], 
                                     capture_output=True, text=True, timeout=5)
            
            if test_result.returncode != 0:
                logger.warning(f"Method {method['name']} failed test, trying next method")
                continue
            
            # Start tensorboard process
            process = subprocess.Popen(
                method['cmd'],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                start_new_session=True
            )
            
            # Wait for process to start
            time.sleep(3)
            
            # Check if process is still running
            if process.poll() is None:
                logger.info(f"TensorBoard started successfully using {method['name']}")
                
                if open_browser:
                    webbrowser.open(f"http://localhost:{port}")
                    logger.info("TensorBoard opened in browser")
                
                logger.info("TensorBoard started successfully!")
                logger.info("Press Ctrl+C to stop TensorBoard")
                
                # Keep script running
                try:
                    process.wait()
                except KeyboardInterrupt:
                    logger.info("Stopping TensorBoard...")
                    process.terminate()
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
                    logger.info("TensorBoard stopped")
                
                return True
            else:
                stdout, stderr = process.communicate()
                logger.warning(f"Method {method['name']} failed to start TensorBoard")
                logger.warning(f"STDERR: {stderr}")
                continue
                
        except FileNotFoundError:
            logger.warning(f"Method {method['name']} command not found")
            continue
        except subprocess.TimeoutExpired:
            logger.warning(f"Method {method['name']} timed out")
            continue
        except Exception as e:
            logger.warning(f"Method {method['name']} failed: {e}")
            continue
    
    # All methods failed
    logger.error("All TensorBoard startup methods failed")
    logger.error("Please ensure TensorBoard is installed: pip install tensorboard")
    return False


def stop_tensorboard(port: int):
    """Stop TensorBoard server."""
    try:
        import psutil
        
        # Find processes using the port
        for proc in psutil.process_iter(['pid', 'name', 'connections']):
            try:
                for conn in proc.info['connections'] or []:
                    if conn.laddr.port == port and 'tensorboard' in proc.info['name'].lower():
                        logger.info(f"Found TensorBoard process with PID {proc.info['pid']}")
                        proc.terminate()
                        proc.wait(timeout=5)
                        logger.info("TensorBoard stopped successfully")
                        return True
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        
        logger.info(f"No TensorBoard process found on port {port}")
        return False
        
    except ImportError:
        logger.warning("psutil not available. Install with: pip install psutil")
        return False
    except Exception as e:
        logger.error(f"Failed to stop TensorBoard: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="TensorBoard utility for satellite coordinate prediction")
    parser.add_argument("action", choices=["start", "stop", "status"], 
                       help="Action to perform")
    parser.add_argument("--logdir", type=str, default="logs/tensorboard",
                       help="TensorBoard log directory")
    parser.add_argument("--port", type=int, default=6006,
                       help="TensorBoard port")
    parser.add_argument("--no-browser", action="store_true",
                       help="Don't open browser automatically")
    
    args = parser.parse_args()
    
    if args.action == "start":
        success = start_tensorboard(args.logdir, args.port, not args.no_browser)
        sys.exit(0 if success else 1)
    
    elif args.action == "stop":
        success = stop_tensorboard(args.port)
        sys.exit(0 if success else 1)
    
    elif args.action == "status":
        if is_port_available(args.port):
            logger.info(f"Port {args.port} is available - TensorBoard not running")
        else:
            logger.info(f"TensorBoard appears to be running on http://localhost:{args.port}")
            logger.info("Use 'python tensorboard_utils.py stop' to stop it")


if __name__ == "__main__":
    main()