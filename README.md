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

Get OPENAI API from https://platform.openai.com/account/api-keys

```sh
export OPENAI_API_KEY=<your key>
```

### Local Battles
```sh
python src/main.py # fill in your username and password for PokeLLMon
``` 

