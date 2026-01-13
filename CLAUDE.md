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
- `GOOGLE_MAPS_API_KEY`: Google Maps API key for geocoding
- `FIRECRAWL_API_KEY`: Firecrawl API key for web scraping (get from firecrawl.dev)

## Architecture

### LangGraph Workflow (`src/graph/`)

The conversational flow is a linear graph: `extract_info → predict_price → chatbot → END`

- **GraphState** (`state.py`): TypedDict holding `messages`, `features` (PropertyFeatures), `user_input_url`, `prediction_result`, `price_comparison`, and `unknown_fields`
- **Nodes** (`nodes.py`):
  - `extract_info`: Parses user message for URLs (uses Firecrawl for scraping) or uses LLM structured output to extract property features
  - `predict_price`: Runs prediction if minimum data exists (area_name + size/dimensions), also calculates price comparison if actual_price exists
  - `chatbot`: Generates Vietnamese response using Gemini with system prompt containing collected features, prediction status, and tool instructions

### Data Model (`src/models.py`)

`PropertyFeatures` is a Pydantic model with ~20 optional fields for Vietnamese real estate (location, property type, dimensions, amenities, legal status). All fields are optional to allow incremental collection.

### Supporting Modules

- `src/ml/real_estate_predictor.py`: XGBoost-based price predictor with SHAP explanations and confidence intervals
- `src/utils/scraper.py`: Basic URL scraper (fallback for when Firecrawl is unavailable)
- `src/utils/price_comparison.py`: Utility for comparing predicted vs actual prices with Vietnamese explanations

### Tools (`src/tools/`)

- `geocoding.py`: LangChain tool for getting coordinates from addresses using Google Maps API
- `property_scraper.py`: LangChain tool using Firecrawl to extract property info (including actual price) from Vietnamese real estate listing URLs. Supports batdongsan.com.vn, chotot.com, alonhadat.com.vn, mogi.vn, homedy.com

### Frontend (`app.py`)

Streamlit chat interface maintaining session state for messages, features, and the compiled LangGraph. Sidebar displays:
- Collected property information as JSON
- Prediction results with SHAP explanation
- Actual price (from listing or user input)
- Price comparison metrics (predicted vs actual) with accuracy assessment

## Valid Feature Values

See `docs/u_features.md` for complete list of valid categorical values for features like `area_name`, `category_name`, `direction_name`, etc. These are specific to Ho Chi Minh City districts and Vietnamese real estate terminology.
