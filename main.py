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
