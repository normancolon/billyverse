@echo off
echo Setting up AI Trading System Environment...

REM Check if Python 3 is installed
python --version 2>NUL
if errorlevel 1 (
    echo Python 3 is not installed. Please install Python 3 and try again.
    exit /b 1
)

REM Create virtual environment
echo Creating virtual environment...
python -m venv venv

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat

REM Upgrade pip
echo Upgrading pip...
python -m pip install --upgrade pip

REM Install requirements
echo Installing requirements...
pip install -r requirements.txt

REM Create .env file if it doesn't exist
if not exist .env (
    echo Creating .env file...
    (
        echo # Exchange API Keys
        echo EXCHANGE_API_KEY=your_api_key_here
        echo EXCHANGE_API_SECRET=your_api_secret_here
        echo.
        echo # Trading Parameters
        echo INITIAL_CAPITAL=100000
        echo MAX_POSITION_SIZE=0.2
        echo MAX_LEVERAGE=2.0
        echo.
        echo # Risk Parameters
        echo RISK_FREE_RATE=0.02
        echo MAX_DRAWDOWN=0.2
        echo.
        echo # Backtesting Parameters
        echo BACKTEST_START_DATE=2023-01-01
        echo BACKTEST_END_DATE=2023-12-31
        echo.
        echo # Model Parameters
        echo MODEL_SAVE_PATH=models/saved
        echo LOG_PATH=logs
    ) > .env
)

REM Create necessary directories
echo Creating necessary directories...
if not exist models\saved mkdir models\saved
if not exist logs mkdir logs
if not exist data mkdir data

echo Setup completed successfully!
echo To activate the virtual environment, run: venv\Scripts\activate.bat 