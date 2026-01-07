# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A Vietnamese real estate price prediction chatbot using LangGraph for conversational flow orchestration, Google Gemini for LLM interactions, and Streamlit for the web interface. Users can input property details via chat or provide a URL for automatic extraction.

## Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run the Streamlit app
streamlit run app.py
```

## Environment Setup

Copy `.env.example` to `.env` and set:
- `GOOGLE_API_KEY`: Google API key for Gemini
- `GEMINI_MODEL`: Model name (referenced in nodes.py)

## Architecture

### LangGraph Workflow (`src/graph/`)

The conversational flow is a linear graph: `extract_info → predict_price → chatbot → END`

- **GraphState** (`state.py`): TypedDict holding `messages`, `features` (PropertyFeatures), `user_input_url`, and `prediction_result`
- **Nodes** (`nodes.py`):
  - `extract_info`: Parses user message for URLs (scraping) or uses LLM structured output to extract property features
  - `predict_price`: Runs prediction if minimum data exists (area_name + size/dimensions)
  - `chatbot`: Generates Vietnamese response using Gemini with system prompt containing collected features and prediction status

### Data Model (`src/models.py`)

`PropertyFeatures` is a Pydantic model with ~20 optional fields for Vietnamese real estate (location, property type, dimensions, amenities, legal status). All fields are optional to allow incremental collection.

### Supporting Modules

- `src/ml/placeholder_model.py`: Mock price predictor using simple heuristics (base price × location factor × size). To be replaced with trained Scikit-learn/XGBoost model.
- `src/utils/scraper.py`: URL scraper for Vietnamese real estate sites. Currently basic implementation with TODO for domain-specific parsing.

### Frontend (`app.py`)

Streamlit chat interface maintaining session state for messages, features, and the compiled LangGraph. Sidebar displays collected property information as JSON.

## Valid Feature Values

See `docs/u_features.md` for complete list of valid categorical values for features like `area_name`, `category_name`, `direction_name`, etc. These are specific to Ho Chi Minh City districts and Vietnamese real estate terminology.
