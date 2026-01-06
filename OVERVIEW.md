## Real Estate Price Prediction Chatbot - Project Overview

### Project Description
An interactive chatbot application that predicts real estate prices using machine learning, allowing users to either manually input property details or provide a website URL for automatic data extraction and price comparison.

### Technology Stack

**Backend & ML:**
- **LangGraph**: Orchestrate conversational flow and state management
- **LangChain**: Handle LLM interactions and data processing
- **Google Gemini**: LLM for conversational interface (via LangChain)
- **Scikit-learn/XGBoost**: ML model for price prediction (placeholder implementation)
- **Python 3.9+**: Core programming language

**Frontend:**
- **Streamlit**: Interactive web interface with chat components

**Additional Libraries:**
- **BeautifulSoup4**: Web scraping for URL-based input
- **Pandas/NumPy**: Data manipulation
- **Joblib/Pickle**: Model serialization
- **google-generativeai**: Gemini API client

### Core Features

**1. Dual Input Modes**
- **Manual Entry**: Users input property details through conversational interface
- **URL-Based**: Users provide real estate listing URL for automatic extraction

**2. Property Information Collection**
- Transaction type (buy-sell option and rent option)

List of features that need to be collected: in docs/u_features.md

**3. Price Prediction & Analysis**
- Real-time ML-based price prediction (using placeholder model initially)
- Confidence intervals
- Market comparison (for URL mode)
- Price insights and recommendations
