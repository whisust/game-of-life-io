import asyncio
import time
from typing import Optional

import numpy as np

from config import CONFIG, logger
from connection import ConnectionManager
from game import GameState

class ServerState:
    def __init__(self):
        self.connection_manager = ConnectionManager()
        self.game_state: Optional[GameState] = None
        self.grid_size = CONFIG['GRID_SIZE']
        self.tick_rate = CONFIG['TICK_RATE_MS']  # ms
        self.max_players = CONFIG['MAX_PLAYERS']
        self.paused = False
        self.running = False

    def init_game(self, grid_size: int = CONFIG['GRID_SIZE']):
        if self.game_state is None:
            # Create a new game state
            self.grid_size = grid_size
            self.game_state = GameState(grid_size)
            logger.info(f"Game initialized on grid {self.grid_size}x{self.grid_size}")
            self.start()
        return self.game_state

    def reset_game(self, grid_size: Optional[int] = None):
        if self.game_state is None:
            return self.init_game(grid_size)
        else:
            """Create a new game state or reset the existing one."""
            self.grid_size = grid_size or self.grid_size
            self.game_state = GameState(self.grid_size)
            logger.info(f"Game reset on grid {self.grid_size}x{self.grid_size}")

            return self.game_state

    def pause_game(self):
        """Pause the game."""
        if self.game_state is not None:
            self.paused = True
            logger.info("Game paused")
            return True
        return False

    def resume_game(self):
        """Resume the game."""
        if self.game_state is not None:
            self.paused = False
            logger.info("Game resumed")
            return True
        return False

    def stop(self):
        """Stop the game loop and cleanup the server state."""
        logger.info("Stopping game loop and cleaning up server state")
        self.game_state = None
        self.running = False
        self.paused = False
        return True

    def start(self):
        """Start the game loop."""
        logger.info("Starting game loop")
        if not self.running:
            self.running = True
            asyncio.create_task(self.game_loop())
            return True
        return False

    async def game_loop(self):
        """Boucle principale du jeu."""
        logger.info("Starting server game loop")
        while self.running:
            try:
                # Skip if game state is not initialized yet
                if self.game_state is None:
                    await asyncio.sleep(self.tick_rate / 1000.0)
                    continue

                logger.debug(f"Running loop {self.game_state.generation}")

                # Traiter les commandes des joueurs
                self.game_state.process_commands()

                # Calculer la prochaine génération seulement si le jeu n'est pas en pause
                if not self.paused:
                    self.game_state.next_generation()

                # Broadcaster le nouvel état
                players_count = len(self.connection_manager.players)
                await self.connection_manager.broadcast(self.game_state.get_state_for_client(players_count))

                # Attendre le prochain tick
                await asyncio.sleep(self.tick_rate / 1000.0)

            except Exception as e:
                print(f"Erreur dans la boucle de jeu: {e}")
                await asyncio.sleep(1)  # Éviter une boucle d'erreur infinie
