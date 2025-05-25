import time
import uuid
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional

import numpy as np
from scipy import ndimage
from config import logger


class PatternType(Enum):
    GLIDER = "glider"
    OSCILLATOR = "oscillator"
    SPACESHIP = "spaceship"
    BLOCK = "block"
    PULSAR = "pulsar"
    GLIDER_GUN = "glider_gun"

@dataclass
class Player:
    id: str
    name: str
    color: int  # Pour identifier visuellement les contributions
    last_action: float = 0
    is_admin: bool = False

    @classmethod
    def factory(cls, name: Optional[str] = None, color: Optional[int] = None):
        player_id = str(uuid.uuid4())
        color = color or 1  # Couleurs 1-10
        return Player(player_id, name=name, color=color)

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
    ], dtype=bool),

    PatternType.PULSAR: np.array([
        [0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1],
        [0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0],
        [1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0]
    ], dtype=bool),

    PatternType.GLIDER_GUN: np.rot90(np.array([
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
        [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    ], dtype=bool), k=1)
}

class GameState:
    def __init__(self, grid_size: int = 256):
        self.grid_size = grid_size
        self.grid = np.zeros((self.grid_size, self.grid_size), dtype=bool)
        self.command_queue: List[PlacePatternCommand] = []
        self.generation = 0

    def add_command(self, command: PlacePatternCommand, player: Player):
        # Vérification basique pour éviter le spam
        if time.time() - player.last_action > 0.2:  # Limite à 2 actions/sec
            self.command_queue.append(command)
            player.last_action = time.time()
            logger.info(f"Pattern {command.pattern_type.value} placé par {player.name} en ({command.x}, {command.y})")
        else:
            logger.info(f"Action trop rapide ignorée pour {player.name}")

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

        # Log détaillé toutes les 100 générations, sinon juste les stats importantes
        if self.generation % 100 == 0 or alive_before != alive_after:
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
