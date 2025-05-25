import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
import asyncio
import json
import numpy as np
from typing import Dict, List, Set
import uuid
from dataclasses import dataclass
from enum import Enum
import time
from scipy import ndimage

CONFIG = {
    'LOG_LEVEL': os.getenv('LOG_LEVEL', logging.DEBUG),
    'PORT': os.getenv('PORT', 8000),
    'MAX_PLAYERS': int(os.getenv('MAX_PLAYERS', 50)),
    'TICK_RATE_MS': int(os.getenv('TICK_RATE_MS', 250)),
}

# Configuration du logging
logging.basicConfig(
    level=CONFIG['LOG_LEVEL'],
    format='[%(asctime)s][%(levelname)s][%(name)s] %(message)s'
)
logger = logging.getLogger("gameoflife")

# Configuration du jeu
GRID_SIZE = 256
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

@dataclass
class PlacePatternCommand:
    player_id: str
    pattern_type: PatternType
    x: int
    y: int
    timestamp: float

# Patterns prédéfinis du Jeu de la Vie
PATTERNS = {
    PatternType.GLIDER: np.array([
        [0, 1, 0],
        [0, 0, 1],
        [1, 1, 1]
    ], dtype=bool),

    PatternType.OSCILLATOR: np.array([
        [1, 1, 1]
    ], dtype=bool),

    PatternType.SPACESHIP: np.array([
        [0, 1, 1, 1, 1],
        [1, 0, 0, 0, 1],
        [0, 0, 0, 0, 1],
        [1, 0, 0, 1, 0]
    ], dtype=bool),

    PatternType.BLOCK: np.array([
        [1, 1],
        [1, 1]
    ], dtype=bool)
}

class GameState:
    def __init__(self):
        self.grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=bool)
        self.players: Dict[str, Player] = {}
        self.command_queue: List[PlacePatternCommand] = []
        self.generation = 0
        self.last_update = time.time()

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

    def add_command(self, command: PlacePatternCommand):
        # Vérification basique pour éviter le spam
        player = self.players.get(command.player_id)
        if player and time.time() - player.last_action > 0.2:  # Limite à 2 actions/sec
            self.command_queue.append(command)
            player.last_action = time.time()
            logger.debug(f"Pattern {command.pattern_type.value} placé par {player.name} à ({command.x}, {command.y})")
        elif player:
            logger.debug(f"Action trop rapide ignorée pour {player.name}")

    def place_pattern(self, pattern_type: PatternType, x: int, y: int) -> bool:
        """Place un pattern sur la grille. Retourne True si succès."""
        if pattern_type not in PATTERNS:
            return False

        pattern = PATTERNS[pattern_type]
        h, w = pattern.shape

        # Vérification des limites
        if x < 0 or y < 0 or x + w > GRID_SIZE or y + h > GRID_SIZE:
            return False

        # Placement du pattern
        self.grid[y:y+h, x:x+w] |= pattern
        return True

    def process_commands(self):
        """Traite toutes les commandes en attente."""
        for command in self.command_queue:
            self.place_pattern(command.pattern_type, command.x, command.y)
        self.command_queue.clear()

    def next_generation(self):
        """Calcule la prochaine génération selon les règles du Jeu de la Vie."""
        start_time = time.time()
        alive_before = np.sum(self.grid)

        # Calcul des voisins pour chaque cellule
        neighbors = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)

        # # Utilisation de convolution pour compter les voisins efficacement
        # kernel = np.array([[1, 1, 1],
        #                    [1, 0, 1],
        #                    [1, 1, 1]])
        #
        # neighbors = ndimage.convolve(self.grid.astype(int), kernel, mode='constant')
        #
        # # Règles du Jeu de la Vie:
        # # - Une cellule vivante avec 2 ou 3 voisins survit
        # # - Une cellule morte avec exactement 3 voisins naît
        # new_grid = np.zeros_like(self.grid)
        # # A living cell with 2 or 3 neighbors survives
        # new_grid[self.grid & ((neighbors == 2) | (neighbors == 3))] = True
        # # A dead cell with exactly 3 neighbors becomes alive
        # new_grid[(~self.grid) & (neighbors == 3)] = True

        # self.grid = new_grid
        self.generation += 1

        # Statistiques de la génération
        alive_after = np.sum(self.grid)
        calc_time = (time.time() - start_time) * 1000  # en ms

        # Log détaillé toutes les 50 générations, sinon juste les stats importantes
        if self.generation % 50 == 0 or alive_before != alive_after:
            logger.info(f"Génération {self.generation}: {alive_before} -> {alive_after} cellules vivantes "
                        f"({alive_after - alive_before:+d}) | Calcul: {calc_time:.1f}ms | Joueurs: {len(self.players)}")
        else:
            logger.debug(f"Gen {self.generation}: {alive_after} cellules | {calc_time:.1f}ms")

    def get_state_for_client(self) -> dict:
        """Retourne l'état du jeu pour les clients."""
        return {
            "type": "game_state",
            "generation": self.generation,
            "grid": self.grid.tolist(),
            "players_count": len(self.players),
            "timestamp": time.time()
        }

# Instance globale du jeu
game_state = GameState()

class ConnectionManager:
    def __init__(self):
        self.active_connections: Set[WebSocket] = set()

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.add(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.discard(websocket)

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

async def game_loop():
    """Boucle principale du jeu."""
    logger.info(f"Starting game loop")
    while True:
        logger.info(f"Running loop {game_state.generation}")
        try:
            # Traiter les commandes des joueurs
            game_state.process_commands()

            # Calculer la prochaine génération
            game_state.next_generation()

            # Broadcaster le nouvel état
            await manager.broadcast(game_state.get_state_for_client())

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
            player_id = game_state.add_player(websocket, player_name)

            # Confirmer la connexion
            await websocket.send_text(json.dumps({
                "type": "joined",
                "player_id": player_id,
                "message": f"Bienvenue {player_name}!"
            }))

            # Envoyer l'état initial
            await websocket.send_text(json.dumps(game_state.get_state_for_client()))

        # Boucle de réception des commandes
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)

            if message.get("type") == "place_pattern":
                command = PlacePatternCommand(
                    player_id=player_id,
                    pattern_type=PatternType(message["pattern"]),
                    x=message["x"],
                    y=message["y"],
                    timestamp=time.time()
                )
                game_state.add_command(command)

    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.error(f"Erreur WebSocket: {e}")
    finally:
        if player_id:
            game_state.remove_player(player_id)
        manager.disconnect(websocket)


@app.get("/")
async def get_index():
    """Servir la page principale."""
    html_path = Path("static/index.html")
    html_content = html_path.read_text()
    return HTMLResponse(content=html_content)

@app.get("/health")
async def health_check():
    return {
        "status": "ok",
        "players": len(game_state.players),
        "generation": game_state.generation
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=CONFIG['PORT'])
