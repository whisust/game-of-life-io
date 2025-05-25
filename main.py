import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
import asyncio
import json
import numpy as np
from typing import Dict, List, Set, Optional
import uuid
from dataclasses import dataclass
from enum import Enum
import time
from scipy import ndimage

from config import CONFIG
from game import Player, PatternType, Orientation, PlacePatternCommand, GameState

# Configuration du logging
logging.basicConfig(
    level=CONFIG['LOG_LEVEL'],
    format='[%(asctime)s][%(levelname)s][%(name)s] %(message)s'
)
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
        self.players[player_id] = Player(player_id, websocket, name, color)
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

class ServerState:
    def __init__(self):
        self.connection_manager = ConnectionManager()
        self.game_state: Optional[GameState] = None
        self.grid_size = CONFIG['GRID_SIZE']
        self.tick_rate = CONFIG['TICK_RATE_MS']  # ms
        self.max_players = CONFIG['MAX_PLAYERS']

    def reset_game(self):
        """Create a new game state or reset the existing one."""
        if self.game_state is None:
            # Create a new game state
            self.game_state = GameState(self.grid_size)
            logger.info("Game state initialized")
        else:
            # Reset the grid but keep the players
            self.game_state.grid = np.zeros((self.grid_size, self.grid_size), dtype=bool)
            self.game_state.command_queue = []
            self.game_state.generation = 0
            self.game_state.last_update = time.time()
            logger.info("Game state reset")

        return self.game_state

    async def game_loop(self):
        """Boucle principale du jeu."""
        logger.info(f"Starting game loop")
        while True:
            try:
                # Skip if game state is not initialized yet
                if self.game_state is None:
                    await asyncio.sleep(self.tick_rate / 1000.0)
                    continue

                logger.info(f"Running loop {self.game_state.generation}")

                # Traiter les commandes des joueurs
                self.game_state.process_commands()

                # Calculer la prochaine génération
                self.game_state.next_generation()

                # Broadcaster le nouvel état
                players_count = len(self.connection_manager.players)
                await self.connection_manager.broadcast(self.game_state.get_state_for_client(players_count))

                # Attendre le prochain tick
                await asyncio.sleep(self.tick_rate / 1000.0)

            except Exception as e:
                print(f"Erreur dans la boucle de jeu: {e}")
                await asyncio.sleep(1)  # Éviter une boucle d'erreur infinie

# Create the global server state
server_state = ServerState()

@asynccontextmanager
async def lifespan(app: FastAPI):
    asyncio.create_task(server_state.game_loop())
    yield

app = FastAPI(lifespan=lifespan)

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await server_state.connection_manager.connect(websocket)
    player_id = None

    try:
        # Attendre le message d'initialisation
        data = await websocket.receive_text()
        init_data = json.loads(data)

        if init_data.get("type") == "join":
            player_name = init_data.get("name", "Anonymous")
            player_id = server_state.connection_manager.add_player(websocket, player_name)

            # Confirmer la connexion
            await websocket.send_text(json.dumps({
                "type": "joined",
                "player_id": player_id,
                "message": f"Bienvenue {player_name}!"
            }))

            # Envoyer l'état initial si le jeu est déjà initialisé
            if server_state.game_state is not None:
                players_count = len(server_state.connection_manager.players)
                await websocket.send_text(json.dumps(server_state.game_state.get_state_for_client(players_count)))

        # Boucle de réception des commandes
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)

            if message.get("type") == "reset_game":
                # Create or reset the game
                server_state.reset_game()

                # Notify all clients about the reset
                players_count = len(server_state.connection_manager.players)
                await server_state.connection_manager.broadcast({
                    "type": "game_reset",
                    "message": "Game has been reset",
                    "players_count": players_count
                })

                # Send the initial state to all clients
                await server_state.connection_manager.broadcast(server_state.game_state.get_state_for_client(players_count))

            elif message.get("type") == "place_pattern" and server_state.game_state is not None:
                # Get orientation from message, default to UP if not provided
                orientation_str = message.get("orientation", "up")
                orientation = Orientation(orientation_str)

                player = server_state.connection_manager.get_player(player_id)
                if player:
                    command = PlacePatternCommand(
                        player_id=player_id,
                        pattern_type=PatternType(message["pattern"]),
                        x=message["x"],
                        y=message["y"],
                        timestamp=time.time(),
                        orientation=orientation
                    )
                    server_state.game_state.add_command(command, player)

    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.error(f"Erreur WebSocket: {e}")
    finally:
        if player_id:
            server_state.connection_manager.remove_player(player_id)
        server_state.connection_manager.disconnect(websocket)


@app.get("/")
async def get_index():
    """Servir la page principale."""
    html_path = Path("static/index.html")
    html_content = html_path.read_text()
    return HTMLResponse(content=html_content)

@app.get("/health")
async def health_check():
    response = {
        "status": "ok",
        "players": len(server_state.connection_manager.players)
    }

    if server_state.game_state is not None:
        response["generation"] = server_state.game_state.generation
    else:
        response["generation"] = 0
        response["game_state"] = "not_initialized"

    return response

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=CONFIG['PORT'])
