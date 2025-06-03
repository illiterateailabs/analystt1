#!/usr/bin/env python3
"""
WebSocket Demo Client for Analystt1 Platform

This script demonstrates the WebSocket functionality for real-time task progress tracking.
It authenticates with the API, runs a sample crew, and displays real-time progress updates.

Usage:
    python scripts/websocket_demo.py [--host HOST] [--port PORT] [--username USERNAME] [--password PASSWORD]

Requirements:
    pip install requests websocket-client colorama tqdm
"""

import argparse
import json
import signal
import sys
import threading
import time
from typing import Dict, List, Optional, Any, Union
from urllib.parse import urljoin

import requests
import websocket
from colorama import Fore, Style, init
from tqdm import tqdm

# Initialize colorama
init(autoreset=True)

# Default configuration
DEFAULT_HOST = "localhost"
DEFAULT_PORT = 8000
DEFAULT_USERNAME = "admin"
DEFAULT_PASSWORD = "admin123"
DEFAULT_CREW = "fraud_pattern_hunter"  # Default crew to run

# Event types from backend
class EventType:
    # Task lifecycle events
    TASK_STARTED = "task_started"
    TASK_PROGRESS = "task_progress"
    TASK_COMPLETED = "task_completed"
    TASK_FAILED = "task_failed"
    
    # Agent lifecycle events
    AGENT_STARTED = "agent_started"
    AGENT_PROGRESS = "agent_progress"
    AGENT_COMPLETED = "agent_completed"
    AGENT_FAILED = "agent_failed"
    
    # Tool events
    TOOL_STARTED = "tool_started"
    TOOL_COMPLETED = "tool_completed"
    TOOL_FAILED = "tool_failed"
    
    # Crew lifecycle events
    CREW_STARTED = "crew_started"
    CREW_PROGRESS = "crew_progress"
    CREW_COMPLETED = "crew_completed"
    CREW_FAILED = "crew_failed"
    
    # HITL events
    HITL_REVIEW_REQUESTED = "hitl_review_requested"
    HITL_REVIEW_APPROVED = "hitl_review_approved"
    HITL_REVIEW_REJECTED = "hitl_review_rejected"
    
    # System events
    SYSTEM_INFO = "system_info"
    SYSTEM_WARNING = "system_warning"
    SYSTEM_ERROR = "system_error"
    
    # WebSocket events
    CONNECTED = "connected"
    HEARTBEAT = "heartbeat"
    PONG = "pong"


class WebSocketDemo:
    """WebSocket demo client for Analystt1 platform."""
    
    def __init__(self, host: str, port: int, username: str, password: str):
        """Initialize the WebSocket demo client."""
        self.host = host
        self.port = port
        self.username = username
        self.password = password
        self.base_url = f"http://{host}:{port}"
        self.ws_base_url = f"ws://{host}:{port}"
        self.token = None
        self.task_id = None
        self.ws = None
        self.progress_bar = None
        self.running = True
        self.connected = False
        self.last_heartbeat = 0
        self.events = []
        
        # Set up signal handler for graceful shutdown
        signal.signal(signal.SIGINT, self.signal_handler)
    
    def signal_handler(self, sig, frame):
        """Handle keyboard interrupt."""
        print(f"\n{Fore.YELLOW}Keyboard interrupt detected. Closing connections...{Style.RESET_ALL}")
        self.running = False
        if self.ws:
            self.ws.close()
        sys.exit(0)
    
    def authenticate(self) -> bool:
        """Authenticate with the API and get a JWT token."""
        print(f"{Fore.CYAN}Authenticating with API...{Style.RESET_ALL}")
        try:
            response = requests.post(
                urljoin(self.base_url, "/api/v1/auth/login"),
                json={"username": self.username, "password": self.password}
            )
            
            if response.status_code == 200:
                data = response.json()
                self.token = data.get("access_token")
                print(f"{Fore.GREEN}Authentication successful!{Style.RESET_ALL}")
                return True
            else:
                print(f"{Fore.RED}Authentication failed: {response.status_code} - {response.text}{Style.RESET_ALL}")
                return False
        except Exception as e:
            print(f"{Fore.RED}Authentication error: {str(e)}{Style.RESET_ALL}")
            return False
    
    def run_crew(self, crew_name: str) -> bool:
        """Run a crew and get the task ID."""
        print(f"{Fore.CYAN}Running crew: {crew_name}...{Style.RESET_ALL}")
        try:
            headers = {"Authorization": f"Bearer {self.token}"}
            response = requests.post(
                urljoin(self.base_url, "/api/v1/crew/run"),
                headers=headers,
                json={"crew_name": crew_name, "inputs": {}}
            )
            
            if response.status_code == 200:
                data = response.json()
                self.task_id = data.get("task_id")
                print(f"{Fore.GREEN}Crew started with task ID: {self.task_id}{Style.RESET_ALL}")
                return True
            else:
                print(f"{Fore.RED}Failed to run crew: {response.status_code} - {response.text}{Style.RESET_ALL}")
                return False
        except Exception as e:
            print(f"{Fore.RED}Error running crew: {str(e)}{Style.RESET_ALL}")
            return False
    
    def list_available_crews(self) -> List[str]:
        """List available crews."""
        print(f"{Fore.CYAN}Listing available crews...{Style.RESET_ALL}")
        try:
            headers = {"Authorization": f"Bearer {self.token}"}
            response = requests.get(
                urljoin(self.base_url, "/api/v1/crew"),
                headers=headers
            )
            
            if response.status_code == 200:
                data = response.json()
                crews = data.get("crews", [])
                print(f"{Fore.GREEN}Available crews: {', '.join(crews)}{Style.RESET_ALL}")
                return crews
            else:
                print(f"{Fore.RED}Failed to list crews: {response.status_code} - {response.text}{Style.RESET_ALL}")
                return []
        except Exception as e:
            print(f"{Fore.RED}Error listing crews: {str(e)}{Style.RESET_ALL}")
            return []
    
    def on_message(self, ws, message):
        """Handle incoming WebSocket messages."""
        try:
            data = json.loads(message)
            event_type = data.get("type")
            
            # Update progress bar if progress is available
            if "progress" in data and self.progress_bar:
                self.progress_bar.update(data["progress"] - self.progress_bar.n)
            
            # Store event
            self.events.append(data)
            
            # Handle heartbeat
            if event_type == "heartbeat":
                self.last_heartbeat = time.time()
                ws.send(json.dumps({"type": "pong", "timestamp": time.time()}))
                return
            
            # Skip pong messages
            if event_type == "pong":
                return
            
            # Format and print event
            self.print_event(data)
            
            # Check for completion or failure
            if event_type in [EventType.CREW_COMPLETED, EventType.TASK_COMPLETED]:
                print(f"\n{Fore.GREEN}Task completed successfully!{Style.RESET_ALL}")
                if self.progress_bar:
                    self.progress_bar.close()
            elif event_type in [EventType.CREW_FAILED, EventType.TASK_FAILED]:
                print(f"\n{Fore.RED}Task failed!{Style.RESET_ALL}")
                if self.progress_bar:
                    self.progress_bar.close()
        except json.JSONDecodeError:
            print(f"{Fore.RED}Error decoding message: {message}{Style.RESET_ALL}")
        except Exception as e:
            print(f"{Fore.RED}Error handling message: {str(e)}{Style.RESET_ALL}")
    
    def on_error(self, ws, error):
        """Handle WebSocket errors."""
        print(f"{Fore.RED}WebSocket error: {str(error)}{Style.RESET_ALL}")
    
    def on_close(self, ws, close_status_code, close_msg):
        """Handle WebSocket connection close."""
        self.connected = False
        print(f"{Fore.YELLOW}WebSocket connection closed: {close_status_code} {close_msg}{Style.RESET_ALL}")
    
    def on_open(self, ws):
        """Handle WebSocket connection open."""
        self.connected = True
        print(f"{Fore.GREEN}WebSocket connection established{Style.RESET_ALL}")
        print(f"{Fore.CYAN}Waiting for events...{Style.RESET_ALL}")
        
        # Initialize progress bar
        self.progress_bar = tqdm(
            total=100,
            desc="Task Progress",
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]"
        )
    
    def print_event(self, event: Dict[str, Any]):
        """Format and print an event with appropriate colors."""
        event_type = event.get("type", "unknown")
        timestamp = event.get("timestamp_iso", "")
        message = event.get("message", "No message")
        agent_id = event.get("agent_id", "")
        tool_id = event.get("tool_id", "")
        progress = event.get("progress", 0)
        
        # Format timestamp
        if timestamp:
            try:
                timestamp = timestamp.split("T")[1].split(".")[0]
            except:
                pass
        
        # Choose color based on event type
        color = Fore.WHITE
        if event_type.startswith("crew_"):
            color = Fore.BLUE
        elif event_type.startswith("agent_"):
            color = Fore.CYAN
        elif event_type.startswith("tool_"):
            color = Fore.MAGENTA
        elif event_type.startswith("task_"):
            color = Fore.GREEN
        elif event_type.startswith("hitl_"):
            color = Fore.YELLOW
        elif event_type.startswith("system_"):
            color = Fore.RED
        
        # Format event type for display
        display_type = event_type.replace("_", " ").title()
        
        # Build output string
        output = f"\n{color}[{timestamp}] {display_type}{Style.RESET_ALL}"
        
        if agent_id:
            output += f" - Agent: {Fore.CYAN}{agent_id}{Style.RESET_ALL}"
        
        if tool_id:
            output += f" - Tool: {Fore.MAGENTA}{tool_id}{Style.RESET_ALL}"
        
        if progress:
            output += f" - Progress: {Fore.GREEN}{progress}%{Style.RESET_ALL}"
        
        output += f"\n  {message}"
        
        # Print the formatted event
        print(output)
    
    def heartbeat_monitor(self):
        """Monitor heartbeats and reconnect if needed."""
        while self.running:
            if self.connected and time.time() - self.last_heartbeat > 60:
                print(f"{Fore.YELLOW}No heartbeat received in 60 seconds. Reconnecting...{Style.RESET_ALL}")
                self.ws.close()
                self.connect_websocket()
            time.sleep(5)
    
    def connect_websocket(self):
        """Connect to the WebSocket endpoint."""
        if not self.task_id or not self.token:
            print(f"{Fore.RED}Task ID or token not available. Cannot connect to WebSocket.{Style.RESET_ALL}")
            return False
        
        ws_url = f"{self.ws_base_url}/api/v1/ws/tasks/{self.task_id}?token={self.token}"
        print(f"{Fore.CYAN}Connecting to WebSocket: {ws_url}{Style.RESET_ALL}")
        
        # Initialize WebSocket connection
        websocket.enableTrace(False)
        self.ws = websocket.WebSocketApp(
            ws_url,
            on_open=self.on_open,
            on_message=self.on_message,
            on_error=self.on_error,
            on_close=self.on_close
        )
        
        # Start heartbeat monitor
        threading.Thread(target=self.heartbeat_monitor, daemon=True).start()
        
        # Start WebSocket connection in a separate thread
        ws_thread = threading.Thread(target=self.ws.run_forever)
        ws_thread.daemon = True
        ws_thread.start()
        
        return True
    
    def run(self, crew_name: str = DEFAULT_CREW):
        """Run the WebSocket demo."""
        try:
            # Authenticate
            if not self.authenticate():
                return False
            
            # List available crews
            available_crews = self.list_available_crews()
            if not available_crews:
                print(f"{Fore.RED}No crews available.{Style.RESET_ALL}")
                return False
            
            # Use specified crew or prompt for selection
            if crew_name not in available_crews:
                print(f"{Fore.YELLOW}Specified crew '{crew_name}' not found.{Style.RESET_ALL}")
                crew_name = self.prompt_crew_selection(available_crews)
            
            # Run crew
            if not self.run_crew(crew_name):
                return False
            
            # Connect to WebSocket
            if not self.connect_websocket():
                return False
            
            # Keep the main thread running
            try:
                while self.running:
                    time.sleep(0.1)
            except KeyboardInterrupt:
                print(f"\n{Fore.YELLOW}Keyboard interrupt detected. Closing connections...{Style.RESET_ALL}")
                self.running = False
                if self.ws:
                    self.ws.close()
            
            return True
            
        except Exception as e:
            print(f"{Fore.RED}Error running WebSocket demo: {str(e)}{Style.RESET_ALL}")
            return False
    
    def prompt_crew_selection(self, crews: List[str]) -> str:
        """Prompt the user to select a crew."""
        print(f"{Fore.CYAN}Please select a crew:{Style.RESET_ALL}")
        for i, crew in enumerate(crews, 1):
            print(f"{i}. {crew}")
        
        while True:
            try:
                choice = input("Enter number: ")
                index = int(choice) - 1
                if 0 <= index < len(crews):
                    return crews[index]
                else:
                    print(f"{Fore.RED}Invalid choice. Please enter a number between 1 and {len(crews)}.{Style.RESET_ALL}")
            except ValueError:
                print(f"{Fore.RED}Please enter a valid number.{Style.RESET_ALL}")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="WebSocket Demo for Analystt1 Platform")
    parser.add_argument("--host", default=DEFAULT_HOST, help=f"API host (default: {DEFAULT_HOST})")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help=f"API port (default: {DEFAULT_PORT})")
    parser.add_argument("--username", default=DEFAULT_USERNAME, help=f"Username (default: {DEFAULT_USERNAME})")
    parser.add_argument("--password", default=DEFAULT_PASSWORD, help=f"Password (default: {DEFAULT_PASSWORD})")
    parser.add_argument("--crew", default=DEFAULT_CREW, help=f"Crew to run (default: {DEFAULT_CREW})")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    demo = WebSocketDemo(args.host, args.port, args.username, args.password)
    success = demo.run(args.crew)
    sys.exit(0 if success else 1)
