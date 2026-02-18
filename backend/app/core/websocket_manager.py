from fastapi import WebSocket
from typing import Dict, List
import json
import logging

logger = logging.getLogger(__name__)


class ConnectionManager:
    """Manages WebSocket connections for real-time updates"""
    
    def __init__(self):
        self.active_connections: Dict[str, List[WebSocket]] = {}
    
    async def connect(self, websocket: WebSocket, fight_id: str):
        """Connect a client to a fight stream"""
        await websocket.accept()
        if fight_id not in self.active_connections:
            self.active_connections[fight_id] = []
        self.active_connections[fight_id].append(websocket)
        logger.info(f"Client connected to fight {fight_id}")
    
    def disconnect(self, websocket: WebSocket, fight_id: str):
        """Disconnect a client"""
        if fight_id in self.active_connections:
            self.active_connections[fight_id].remove(websocket)
            logger.info(f"Client disconnected from fight {fight_id}")
    
    async def send_personal_message(self, message: dict, websocket: WebSocket):
        """Send message to specific client"""
        await websocket.send_text(json.dumps(message))
    
    async def broadcast(self, message: dict, fight_id: str):
        """Broadcast message to all clients watching a fight"""
        if fight_id in self.active_connections:
            for connection in self.active_connections[fight_id]:
                try:
                    await connection.send_text(json.dumps(message))
                except Exception as e:
                    logger.error(f"Error broadcasting to client: {e}")


manager = ConnectionManager()
