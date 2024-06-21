"""poke_env module init.
"""
import logging
import src.environment as environment
import src.exceptions as exceptions
import src.player as player
import src.client as ps_client
import src.stats as stats
import src.teambuilder as teambuilder
from src.data import gen_data, to_id_str
from src.exceptions import ShowdownException
from src.client import AccountConfiguration
from src.client.server_configuration import (
    LocalhostServerConfiguration,
    ServerConfiguration,
    ShowdownServerConfiguration,
)
from src.stats import compute_raw_stats

__logger = logging.getLogger("poke-env")
__stream_handler = logging.StreamHandler()
__formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
__stream_handler.setFormatter(__formatter)
__logger.addHandler(__stream_handler)
logging.addLevelName(25, "PS_ERROR")

__all__ = [
    "AccountConfiguration",
    "LocalhostServerConfiguration",
    "ServerConfiguration",
    "ShowdownException",
    "ShowdownServerConfiguration",
    "compute_raw_stats",
    "environment",
    "exceptions",
    "gen_data",
    "player",
    "ps_client",
    "stats",
    "teambuilder",
    "to_id_str",
]
