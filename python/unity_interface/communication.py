"""
Unity Communication Module
Handles low-level communication protocols between Unity and Python
"""

import socket
import json
import logging
from typing import Dict, Any, Optional
import struct

logger = logging.getLogger(__name__)


class UnityCommunication:
    """
    Handles custom communication with Unity for simulation data exchange
    Useful for sending/receiving data outside of ML-Agents protocol
    """

    def __init__(self, host: str = 'localhost', port: int = 9000):
        """
        Initialize communication handler

        Args:
            host: Host address
            port: Communication port
        """
        self.host = host
        self.port = port
        self.socket: Optional[socket.socket] = None
        self.connected = False

    def connect(self, timeout: int = 10) -> bool:
        """
        Establish connection with Unity

        Args:
            timeout: Connection timeout in seconds

        Returns:
            bool: True if connected, False otherwise
        """
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.settimeout(timeout)
            self.socket.connect((self.host, self.port))
            self.connected = True
            logger.info(f"Connected to Unity at {self.host}:{self.port}")
            return True

        except Exception as e:
            logger.error(f"Failed to connect to Unity: {e}")
            self.connected = False
            return False

    def send_message(self, message: Dict[str, Any]) -> bool:
        """
        Send JSON message to Unity

        Args:
            message: Dictionary to send as JSON

        Returns:
            bool: True if successful, False otherwise
        """
        if not self.connected or self.socket is None:
            logger.error("Not connected to Unity")
            return False

        try:
            # Convert message to JSON bytes
            json_data = json.dumps(message).encode('utf-8')

            # Send message length first (4 bytes)
            msg_length = struct.pack('>I', len(json_data))
            self.socket.sendall(msg_length)

            # Send actual message
            self.socket.sendall(json_data)

            logger.debug(f"Sent message: {message}")
            return True

        except Exception as e:
            logger.error(f"Failed to send message: {e}")
            return False

    def receive_message(self, timeout: Optional[int] = None) -> Optional[Dict[str, Any]]:
        """
        Receive JSON message from Unity

        Args:
            timeout: Receive timeout in seconds (None for blocking)

        Returns:
            dict: Received message or None if failed
        """
        if not self.connected or self.socket is None:
            logger.error("Not connected to Unity")
            return None

        try:
            if timeout is not None:
                self.socket.settimeout(timeout)

            # Receive message length (4 bytes)
            length_data = self._receive_all(4)
            if not length_data:
                return None

            msg_length = struct.unpack('>I', length_data)[0]

            # Receive actual message
            json_data = self._receive_all(msg_length)
            if not json_data:
                return None

            message = json.loads(json_data.decode('utf-8'))
            logger.debug(f"Received message: {message}")

            return message

        except socket.timeout:
            logger.warning("Receive timeout")
            return None

        except Exception as e:
            logger.error(f"Failed to receive message: {e}")
            return None

    def _receive_all(self, n: int) -> Optional[bytes]:
        """
        Helper to receive n bytes from socket

        Args:
            n: Number of bytes to receive

        Returns:
            bytes: Received data or None if connection closed
        """
        data = bytearray()
        while len(data) < n:
            packet = self.socket.recv(n - len(data))
            if not packet:
                return None
            data.extend(packet)
        return bytes(data)

    def send_simulation_parameters(self, parameters: Dict[str, Any]) -> bool:
        """
        Send simulation parameters to Unity

        Args:
            parameters: Simulation parameters dictionary

        Returns:
            bool: True if successful
        """
        message = {
            'type': 'simulation_parameters',
            'data': parameters
        }
        return self.send_message(message)

    def request_simulation_data(self) -> Optional[Dict[str, Any]]:
        """
        Request current simulation data from Unity

        Returns:
            dict: Simulation data or None if failed
        """
        request = {
            'type': 'request_data',
            'timestamp': self._get_timestamp()
        }

        if self.send_message(request):
            return self.receive_message(timeout=5)
        return None

    def _get_timestamp(self) -> float:
        """Get current timestamp"""
        import time
        return time.time()

    def disconnect(self):
        """Close the connection"""
        if self.socket is not None:
            try:
                self.socket.close()
                logger.info("Disconnected from Unity")
            except Exception as e:
                logger.error(f"Error disconnecting: {e}")
            finally:
                self.connected = False
                self.socket = None

    def __enter__(self):
        """Context manager entry"""
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.disconnect()