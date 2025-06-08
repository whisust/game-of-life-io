import asyncio
import json
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

from config import CONFIG, logger
from game import PatternType, Orientation, PlacePatternCommand
from server import ServerState


# Create the global server state
server_state = ServerState()

# @asynccontextmanager
# async def lifespan(app: FastAPI):
#     # Start the game loop
#     game_loop_task = asyncio.create_task(server_state.game_loop())
#     yield

app = FastAPI()

@app.websocket("/game-of-life/ws")
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
            player = server_state.connection_manager.get_player(player_id)

            # Confirmer la connexion
            await websocket.send_text(json.dumps({
                "type": "joined",
                "player": {
                    "id": player.id,
                    "name": player.name,
                    "is_admin": True or player.is_admin, # everybody is admin now
                },
                "message": f"Bienvenue {player_name}!",
                "grid_size": server_state.grid_size
            }))

            # Envoyer l'état initial si le jeu est déjà initialisé
            if server_state.game_state is not None:
                players_count = len(server_state.connection_manager.players)
                await websocket.send_text(json.dumps(server_state.game_state.get_state_for_client(players_count)))

        # Boucle de réception des commandes
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)
            player = server_state.connection_manager.get_player(player_id)

            if message.get("type") == "reset_game" and player.is_admin:
                # Create or reset the game
                server_state.reset_game(message.get('grid_size'))

                # Notify all clients about the reset
                players_count = len(server_state.connection_manager.players)
                await server_state.connection_manager.broadcast({
                    "type": "game_reset",
                    "message": "Game has been reset",
                    "grid_size": server_state.grid_size,
                    "players_count": players_count
                })

                # Send the initial state to all clients
                await server_state.connection_manager.broadcast(server_state.game_state.get_state_for_client(players_count))

            elif message.get("type") == "init_game" and player.is_admin:
                # Initialize the game with specified grid size
                grid_size = message.get("grid_size", CONFIG['GRID_SIZE'])
                server_state.init_game(grid_size)

                # Notify all clients about the initialization
                players_count = len(server_state.connection_manager.players)
                await server_state.connection_manager.broadcast({
                    "type": "game_initialized",
                    "message": f"Game has been initialized with grid size {grid_size}x{grid_size}",
                    "grid_size": grid_size,
                    "players_count": players_count
                })

                # Send the initial state to all clients
                if server_state.game_state is not None:
                    await server_state.connection_manager.broadcast(server_state.game_state.get_state_for_client(players_count))

            elif message.get("type") == "pause_game" and server_state.game_state is not None and player.is_admin:
                # Pause the game
                if server_state.pause_game():
                    # Notify all clients about the pause
                    await server_state.connection_manager.broadcast({
                        "type": "game_paused",
                        "message": "Game has been paused"
                    })

            elif message.get("type") == "resume_game" and server_state.game_state is not None and player.is_admin:
                # Resume the game
                if server_state.resume_game():
                    # Notify all clients about the resume
                    await server_state.connection_manager.broadcast({
                        "type": "game_resumed",
                        "message": "Game has been resumed"
                    })

            elif message.get("type") == "stop_game" and player.is_admin:
                # Stop the game
                if server_state.stop():
                    # Notify all clients about the stop
                    await server_state.connection_manager.broadcast({
                        "type": "game_stopped",
                        "message": "Game has been stopped"
                    })

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


@app.get("/game-of-life")
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

@app.get("/")
async def get_home():
    """Serve the homepage."""
    html_path = Path("static/home.html")
    html_content = html_path.read_text()
    return HTMLResponse(content=html_content)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=CONFIG['HOST'], port=CONFIG['PORT'])
