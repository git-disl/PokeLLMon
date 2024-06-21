# PokéLLMon

[//]: # (<div align="center">)

[//]: # (  <img src="./resource/LLM_attrition_strategy.gif" alt="PokemonBattle">)

[//]: # (</div>)

[//]: # ()

### Requirements:

```sh
python >= 3.8
openai >= 1.7.2
``` 

### Setting up a local battle engine

1. Install Node.js v10+.
2. Clone the Pokémon Showdown repository and set it up:

```sh
git clone https://github.com/smogon/pokemon-showdown.git
cd pokemon-showdown
npm install
cp config/config-example.js config/config.js
node pokemon-showdown start --no-security
Enter "http://localhost:8000/" in your browsers.
``` 

### Configuring OpenAI API

Get GPT-4 API from https://platform.openai.com/account/api-keys

### Configuring Players

Register in your account on https://play.pokemonshowdown.com/ and get your password.

```sh
from poke_env import AccountConfiguration
# No authentication required for the local server
my_account_config = AccountConfiguration("your_username", "your_password")
player = Player(account_configuration=my_account_config)
``` 

### Local Battles
```sh
python main.py # fill in your username and password for PokeLLMon
``` 

