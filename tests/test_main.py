# File: tests/test_main.py

import time
from unittest.mock import AsyncMock

import numpy as np
import pytest
from fastapi.websockets import WebSocket
from main import GameState, PlacePatternCommand, PatternType, Orientation, PATTERNS, GRID_SIZE


@pytest.fixture
def game_state():
    return GameState()

@pytest.fixture
def mock_websocket():
    return AsyncMock(spec=WebSocket)

def test_add_player(game_state, mock_websocket):
    player_name = "Player1"
    player_id = game_state.add_player(mock_websocket, player_name)

    assert player_id in game_state.players
    assert game_state.players[player_id].name == player_name
    assert isinstance(player_id, str)

def test_remove_player(game_state, mock_websocket):
    player_name = "Player1"
    player_id = game_state.add_player(mock_websocket, player_name)

    game_state.remove_player(player_id)

    assert player_id not in game_state.players

def test_add_command_valid(game_state, mock_websocket):
    player_name = "Player1"
    player_id = game_state.add_player(mock_websocket, player_name)

    command = PlacePatternCommand(
        player_id=player_id,
        pattern_type=PatternType.GLIDER,
        x=1,
        y=1,
        timestamp=0
    )

    game_state.add_command(command)

    assert len(game_state.command_queue) == 1
    assert game_state.command_queue[0] == command

def test_add_command_too_fast(game_state, mock_websocket):
    player_name = "Player1"
    player_id = game_state.add_player(mock_websocket, player_name)
    game_state.players[player_id].last_action = time.time()  # Simulate recent action

    command = PlacePatternCommand(
        player_id=player_id,
        pattern_type=PatternType.GLIDER,
        x=1,
        y=1,
        timestamp=0
    )

    game_state.add_command(command)

    assert len(game_state.command_queue) == 0

def test_place_pattern_success(game_state):
    pattern_type = PatternType.BLOCK
    x, y = 1, 1

    game_state.grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=bool)
    success = game_state.place_pattern(pattern_type, x, y)

    assert success
    assert game_state.grid[y:y+2, x:x+2].all()

def test_place_pattern_out_of_bounds(game_state):
    pattern_type = PatternType.BLOCK
    x, y = GRID_SIZE - 1, GRID_SIZE - 1  # Placement near the edge

    success = game_state.place_pattern(pattern_type, x, y)

    assert not success

def test_place_pattern_with_orientation(game_state):
    """Test that patterns are correctly rotated based on orientation."""

    # Test with a glider pattern which has a distinct shape for each orientation
    pattern_type = PatternType.GLIDER
    x, y = 10, 10

    # Clear the grid
    game_state.grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=bool)

    # Place pattern with UP orientation (default)
    success_up = game_state.place_pattern(pattern_type, x, y, Orientation.UP)
    assert success_up

    # Get the pattern as placed with UP orientation
    up_pattern = game_state.grid[y:y+3, x:x+3].copy()

    # Clear the grid
    game_state.grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=bool)

    # Place pattern with RIGHT orientation
    success_right = game_state.place_pattern(pattern_type, x, y, Orientation.RIGHT)
    assert success_right

    # Get the pattern as placed with RIGHT orientation
    right_pattern = game_state.grid[y:y+3, x:x+3].copy()

    # Verify that the patterns are different (rotated)
    assert not np.array_equal(up_pattern, right_pattern)

    # Verify that the RIGHT pattern is the UP pattern rotated 270° clockwise
    assert np.array_equal(right_pattern, np.rot90(up_pattern, k=3))

def test_process_commands(game_state, mock_websocket):
    player_name = "Player1"
    player_id = game_state.add_player(mock_websocket, player_name)

    command = PlacePatternCommand(
        player_id=player_id,
        pattern_type=PatternType.BLOCK,
        x=1,
        y=1,
        timestamp=0
    )

    game_state.add_command(command)
    game_state.process_commands()

    assert game_state.grid[1:3, 1:3].all()
    assert not game_state.command_queue

def test_next_generation(game_state):
    game_state.grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=bool)
    game_state.grid[10:13, 10:13] = PATTERNS[PatternType.GLIDER]

    for i in range(1, 5):
        game_state.next_generation()

        assert np.sum(game_state.grid) == PATTERNS[PatternType.GLIDER].sum()
        assert game_state.generation == i
