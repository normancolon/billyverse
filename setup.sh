#!/bin/bash

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}Setting up AI Trading System Environment...${NC}"

# Check if Python 3 is installed
if ! command -v python3 &> /dev/null; then
    echo "Python 3 is not installed. Please install Python 3 and try again."
    exit 1
fi

# Create virtual environment
echo -e "${BLUE}Creating virtual environment...${NC}"
python3 -m venv venv

# Activate virtual environment
echo -e "${BLUE}Activating virtual environment...${NC}"
source venv/bin/activate

# Upgrade pip
echo -e "${BLUE}Upgrading pip...${NC}"
python -m pip install --upgrade pip

# Install requirements
echo -e "${BLUE}Installing requirements...${NC}"
pip install -r requirements.txt

# Create .env file if it doesn't exist
if [ ! -f .env ]; then
    echo -e "${BLUE}Creating .env file...${NC}"
    cat > .env << EOL
# Exchange API Keys
EXCHANGE_API_KEY=your_api_key_here
EXCHANGE_API_SECRET=your_api_secret_here

# Trading Parameters
INITIAL_CAPITAL=100000
MAX_POSITION_SIZE=0.2
MAX_LEVERAGE=2.0

# Risk Parameters
RISK_FREE_RATE=0.02
MAX_DRAWDOWN=0.2

# Backtesting Parameters
BACKTEST_START_DATE=2023-01-01
BACKTEST_END_DATE=2023-12-31

# Model Parameters
MODEL_SAVE_PATH=models/saved
LOG_PATH=logs
EOL
fi

# Create necessary directories
echo -e "${BLUE}Creating necessary directories...${NC}"
mkdir -p models/saved
mkdir -p logs
mkdir -p data

echo -e "${GREEN}Setup completed successfully!${NC}"
echo -e "${GREEN}To activate the virtual environment, run: source venv/bin/activate${NC}" 