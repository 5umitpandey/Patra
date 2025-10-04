# Patra ML Backend

This folder contains the machine learning backend for the Patra project. It provides user recommendation logic, bio-based matching, interaction-based personalization, and dynamic ranking using Elo scores. The backend is designed to be run as a CLI tool for generating and updating user recommendations.

## Features
- Structured user matching based on profile features
- Semantic bio similarity using transformer embeddings
- Personalized recommendations using swipe/interactions
- Dynamic user ranking with Elo scoring
- Modular, configurable pipeline

## Folder Structure
- `production/` — Core backend modules (matching, ranking, logging, etc.)
- `config/` — YAML configuration files
- `data/` — User and interaction data (CSV files)
- `models/` — Pretrained or locally stored ML models
- `tests/` — (Empty) Placeholder for future tests
- `requirements.txt` — Python dependencies

## Main Pipeline Overview
The backend pipeline consists of the following steps:
1. **Data Match**: Computes structured similarity between users (age, location, hobbies, etc.).
2. **Bio Match**: Uses transformer models to compute semantic similarity between user bios.
3. **Interaction Adjustment**: Adjusts candidate scores based on past user interactions (like, superlike, reject).
4. **Elo Update**: Updates user Elo ratings based on interactions to reflect dynamic desirability.
5. **Recommendation**: Combines all signals to generate a ranked feed for each user.

## Configuration
- Main settings are in `config/settings.yaml`.
- You can adjust weights for different matching signals, model paths, logging, and other parameters.

## Usage
### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Main Orchestrator
Generate recommendations for a user:
```bash
python production/main.py --user_id <USER_ID> [--top_n N]
```

Record an interaction (like, superlike, reject) and update recommendations:
```bash
python production/main.py --user_id <USER_ID> --interact <TARGET_ID> <ACTION>
```

- `<USER_ID>`: The user for whom to generate recommendations
- `<TARGET_ID>`: The user being interacted with
- `<ACTION>`: One of `like`, `superlike`, or `reject`

### 3. Configuration
Edit `config/settings.yaml` to tune weights, model paths, and other parameters.

## Core Modules
- `main.py`: Orchestrates the pipeline, CLI entry point
- `recommender.py`: Combines all signals to generate recommendations
- `data_match.py`: Structured feature-based matching
- `bio_match.py`: Semantic similarity using transformer models
- `reject_superlike_like.py`: Adjusts scores based on user interactions
- `elo_update.py`: Maintains and updates Elo ratings
- `config_loader.py`: Loads YAML configuration
- `logger.py`: Logging setup
- `utils.py`: Helper functions

## Notes
- All data is stored locally in the `data/` folder.
- The backend is modular and can be extended with new matching or ranking strategies.
- For best results, ensure the `sentence-transformers` model specified in the config is available locally or can be downloaded.

