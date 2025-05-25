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
        logger.info("Starting game loop")
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
