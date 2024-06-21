"""poke_env module init.
"""
import logging
import env.environment as environment
import env.exceptions as exceptions
import env.player as player
import env.client as ps_client
import env.stats as stats
import env.teambuilder as teambuilder
from env.data import gen_data, to_id_str
from env.exceptions import ShowdownException
from env.client import AccountConfiguration
from env.client.server_configuration import (
    LocalhostServerConfiguration,
    ServerConfiguration,
    ShowdownServerConfiguration,
)
from env.stats import compute_raw_stats

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
