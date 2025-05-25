import json
import logging
import uuid
from typing import Dict, Set

from fastapi import WebSocket

from game import Player

logger = logging.getLogger("gameoflife")

class ConnectionManager:
    def __init__(self):
        self.active_connections: Set[WebSocket] = set()
        self.players: Dict[str, Player] = {}

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.add(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.discard(websocket)

    def add_player(self, websocket: WebSocket, name: str) -> str:
        player_id = str(uuid.uuid4())
        color = len(self.players) % 10 + 1  # Couleurs 1-10
        self.players[player_id] = Player(player_id, name, color)
        logger.info(f"Nouveau joueur connecté: {name} (ID: {player_id[:8]}...) - Total: {len(self.players)} joueurs")
        return player_id

    def remove_player(self, player_id: str):
        if player_id in self.players:
            player = self.players[player_id]
            del self.players[player_id]
            logger.info(f"Joueur déconnecté: {player.name} (ID: {player_id[:8]}...) - Total: {len(self.players)} joueurs")

    def get_player(self, player_id: str) -> Player:
        return self.players.get(player_id)

    async def broadcast(self, message: dict):
        """Broadcast un message à tous les clients connectés."""
        if self.active_connections:
            message_str = json.dumps(message)
            disconnected = set()

            for connection in self.active_connections:
                try:
                    await connection.send_text(message_str)
                except:
                    disconnected.add(connection)

            # Nettoyage des connexions fermées
            for conn in disconnected:
                self.active_connections.discard(conn)