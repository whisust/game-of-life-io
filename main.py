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

# Configuration du logging
logging.basicConfig(
    level=CONFIG['LOG_LEVEL'],
    format='[%(asctime)s][%(levelname)s][%(name)s] %(message)s'
)
logger = logging.getLogger("gameoflife")

# Configuration du jeu
GRID_SIZE = CONFIG['GRID_SIZE']
TICK_RATE = CONFIG['TICK_RATE_MS']  # ms
MAX_PLAYERS = CONFIG['MAX_PLAYERS']

class PatternType(Enum):
    GLIDER = "glider"
    OSCILLATOR = "oscillator"
    SPACESHIP = "spaceship"
    BLOCK = "block"

@dataclass
class Player:
    id: str
    websocket: WebSocket
    name: str
    color: int  # Pour identifier visuellement les contributions
    last_action: float = 0

    @classmethod
    def factory(cls, name: Optional[str] = None, color: Optional[int] = None, websocket: Optional[WebSocket] = None):
        player_id = str(uuid.uuid4())
        color = color or 1  # Couleurs 1-10
        return Player(player_id, websocket=websocket, name=name, color=color)

class Orientation(Enum):
    UP = "up"
    RIGHT = "right"
    DOWN = "down"
    LEFT = "left"

@dataclass
class PlacePatternCommand:
    player_id: str
    pattern_type: PatternType
    x: int
    y: int
    timestamp: float
    orientation: Orientation = Orientation.UP

@dataclass
class ResetGameCommand:
    player_id: str
    timestamp: float

@dataclass
class StartGameCommand:
    player_id: str
    timestamp: float

# Patterns prédéfinis du Jeu de la Vie
PATTERNS = {
    PatternType.GLIDER: np.array([
        [0, 1, 1],
        [1, 0, 1],
        [0, 0, 1]
    ], dtype=bool),

    PatternType.OSCILLATOR: np.array([
        [1, 1, 1]
    ], dtype=bool),

    PatternType.SPACESHIP: np.array([
        [1, 1, 1, 0],
        [1, 0, 0, 1],
        [1, 0, 0, 0],
        [1, 0, 0, 0],
        [0, 1, 0, 1]
    ], dtype=bool),

    PatternType.BLOCK: np.array([
        [1, 1],
        [1, 1]
    ], dtype=bool)
}

class GameState:
    def __init__(self, grid_size: int = 256):
        self.grid_size = grid_size
        self.grid = np.zeros((self.grid_size, self.grid_size), dtype=bool)
        self.command_queue: List[PlacePatternCommand] = []
        self.generation = 0
        self.last_update = time.time()

    def add_command(self, command: PlacePatternCommand, player: Player):
        # Vérification basique pour éviter le spam
        if time.time() - player.last_action > 0.2:  # Limite à 2 actions/sec
            self.command_queue.append(command)
            player.last_action = time.time()
            logger.debug(f"Pattern {command.pattern_type.value} placé par {player.name} à ({command.x}, {command.y})")
        else:
            logger.debug(f"Action trop rapide ignorée pour {player.name}")

    def rotate_pattern(self, pattern: np.ndarray, orientation: Orientation) -> np.ndarray:
        """Rotate a pattern according to the specified orientation."""
        if orientation == Orientation.UP:
            return pattern
        elif orientation == Orientation.RIGHT:
            return np.rot90(pattern, k=3)  # Rotate 270° clockwise (90° counterclockwise)
        elif orientation == Orientation.DOWN:
            return np.rot90(pattern, k=2)  # Rotate 180°
        elif orientation == Orientation.LEFT:
            return np.rot90(pattern, k=1)  # Rotate 90° clockwise
        return pattern  # Default case

    def place_pattern(self, pattern_type: PatternType, x: int, y: int, orientation: Orientation = Orientation.UP) -> bool:
        """Place un pattern sur la grille. Retourne True si succès."""
        if pattern_type not in PATTERNS:
            return False

        pattern = PATTERNS[pattern_type]
        # Rotate the pattern according to the orientation
        pattern = self.rotate_pattern(pattern, orientation)
        h, w = pattern.shape

        # Vérification des limites
        if x < 0 or y < 0 or x + w > self.grid_size or y + h > self.grid_size:
            return False

        # Placement du pattern
        self.grid[y:y+h, x:x+w] |= pattern
        return True

    def process_commands(self):
        """Traite toutes les commandes en attente."""
        if not self.command_queue:
            return

        for command in self.command_queue:
            self.place_pattern(command.pattern_type, command.x, command.y, command.orientation)
        self.command_queue.clear()

    def next_generation(self):
        """Calcule la prochaine génération selon les règles du Jeu de la Vie."""
        start_time = time.time()
        alive_before = np.sum(self.grid)

        # Calcul des voisins pour chaque cellule
        neighbors = np.zeros((self.grid_size, self.grid_size), dtype=int)

        # Utilisation de convolution pour compter les voisins efficacement
        kernel = np.array([[1, 1, 1],
                           [1, 0, 1],
                           [1, 1, 1]])

        neighbors = ndimage.convolve(self.grid.astype(int), kernel, mode='constant')

        # Règles du Jeu de la Vie:
        # - Une cellule vivante avec 2 ou 3 voisins survit
        # - Une cellule morte avec exactement 3 voisins naît
        new_grid = np.zeros_like(self.grid)
        # A living cell with 2 or 3 neighbors survives
        new_grid[self.grid & ((neighbors == 2) | (neighbors == 3))] = True
        # A dead cell with exactly 3 neighbors becomes alive
        new_grid[(~self.grid) & (neighbors == 3)] = True

        self.grid = new_grid
        self.generation += 1

        # Statistiques de la génération
        alive_after = np.sum(self.grid)
        calc_time = (time.time() - start_time) * 1000  # en ms

        # Log détaillé toutes les 50 générations, sinon juste les stats importantes
        if self.generation % 50 == 0 or alive_before != alive_after:
            logger.info(f"Génération {self.generation}: {alive_before} -> {alive_after} cellules vivantes "
                        f"({alive_after - alive_before:+d}) | Calcul: {calc_time:.1f}ms")
        else:
            logger.debug(f"Gen {self.generation}: {alive_after} cellules | {calc_time:.1f}ms")

    def get_state_for_client(self, players_count: int) -> dict:
        """Retourne l'état du jeu pour les clients."""
        return {
            "type": "game_state",
            "generation": self.generation,
            "grid": self.grid.tolist(),
            "players_count": players_count,
            "timestamp": time.time()
        }

# Instance globale du jeu (initialisée à None)
game_state: Optional[GameState] = None

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

manager = ConnectionManager()

def reset_game():
    """Create a new game state or reset the existing one."""
    global game_state

    if game_state is None:
        # Create a new game state
        game_state = GameState(GRID_SIZE)
        logger.info("Game state initialized")
    else:
        # Reset the grid but keep the players
        game_state.grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=bool)
        game_state.command_queue = []
        game_state.generation = 0
        game_state.last_update = time.time()
        logger.info("Game state reset")

    return game_state

async def game_loop():
    """Boucle principale du jeu."""
    logger.info(f"Starting game loop")
    while True:
        try:
            # Skip if game state is not initialized yet
            if game_state is None:
                await asyncio.sleep(TICK_RATE / 1000.0)
                continue

            logger.info(f"Running loop {game_state.generation}")

            # Traiter les commandes des joueurs
            game_state.process_commands()

            # Calculer la prochaine génération
            game_state.next_generation()

            # Broadcaster le nouvel état
            players_count = len(manager.players)
            await manager.broadcast(game_state.get_state_for_client(players_count))

            # Attendre le prochain tick
            await asyncio.sleep(TICK_RATE / 1000.0)

        except Exception as e:
            print(f"Erreur dans la boucle de jeu: {e}")
            await asyncio.sleep(1)  # Éviter une boucle d'erreur infinie

@asynccontextmanager
async def lifespan(app: FastAPI):
    asyncio.create_task(game_loop())
    yield

app = FastAPI(lifespan=lifespan)
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    player_id = None

    try:
        # Attendre le message d'initialisation
        data = await websocket.receive_text()
        init_data = json.loads(data)

        if init_data.get("type") == "join":
            player_name = init_data.get("name", "Anonymous")
            player_id = manager.add_player(websocket, player_name)

            # Confirmer la connexion
            await websocket.send_text(json.dumps({
                "type": "joined",
                "player_id": player_id,
                "message": f"Bienvenue {player_name}!"
            }))

            # Envoyer l'état initial si le jeu est déjà initialisé
            if game_state is not None:
                players_count = len(manager.players)
                await websocket.send_text(json.dumps(game_state.get_state_for_client(players_count)))

        # Boucle de réception des commandes
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)

            if message.get("type") == "reset_game":
                # Create or reset the game
                reset_game()

                # Notify all clients about the reset
                players_count = len(manager.players)
                await manager.broadcast({
                    "type": "game_reset",
                    "message": "Game has been reset",
                    "players_count": players_count
                })

                # Send the initial state to all clients
                await manager.broadcast(game_state.get_state_for_client(players_count))

            elif message.get("type") == "place_pattern" and game_state is not None:
                # Get orientation from message, default to UP if not provided
                orientation_str = message.get("orientation", "up")
                orientation = Orientation(orientation_str)

                player = manager.get_player(player_id)
                if player:
                    command = PlacePatternCommand(
                        player_id=player_id,
                        pattern_type=PatternType(message["pattern"]),
                        x=message["x"],
                        y=message["y"],
                        timestamp=time.time(),
                        orientation=orientation
                    )
                    game_state.add_command(command, player)

    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.error(f"Erreur WebSocket: {e}")
    finally:
        if player_id:
            manager.remove_player(player_id)
        manager.disconnect(websocket)


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
        "players": len(manager.players)
    }

    if game_state is not None:
        response["generation"] = game_state.generation
    else:
        response["generation"] = 0
        response["game_state"] = "not_initialized"

    return response

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=CONFIG['PORT'])
