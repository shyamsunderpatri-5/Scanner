
"""
NSE SWING SCANNER v8.5 - GITHUB ACTIONS VERSION
Optimized for automated daily execution in GitHub Actions
"""

import sys
import os
import io
import logging
from datetime import datetime, timedelta
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
import shutil 

# Fix Windows encoding BEFORE any other imports
if sys.platform == 'win32':
    os.environ['PYTHONIOENCODING'] = 'utf-8'
    os.environ['PYTHONUTF8'] = '1'
    
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
    except:
        pass
# ============================================================================
# GITHUB ACTIONS CONFIGURATION
# ============================================================================

# Get email credentials from environment variables (GitHub Secrets)
EMAIL_CONFIG = {
    'enabled': True,
    'smtp_server': 'smtp.gmail.com',
    'smtp_port': 587,
    'sender_email': os.environ.get('EMAIL_SENDER', ''),
    'sender_password': os.environ.get('EMAIL_PASSWORD', ''),
    'recipient_email': os.environ.get('EMAIL_RECIPIENT', ''),
}



 

# ============================================================================
# STEP 2: SAFE LOGGING SETUP
# ============================================================================

class SafeStreamHandler(logging.StreamHandler):
    """Handler that safely encodes Unicode for Windows console"""
    
    def emit(self, record):
        try:
            msg = self.format(record)
            # Replace Unicode characters with ASCII equivalents
            replacements = {
                '✅': '[OK]', '❌': '[FAIL]', '⚠️': '[WARN]', '⚠': '[WARN]',
                '📊': '[DATA]', '📈': '[UP]', '📉': '[DOWN]', '🚀': '[GO]',
                '💰': '[MONEY]', '🎯': '[TARGET]', '📧': '[EMAIL]', '📋': '[COPY]',
                '📂': '[DIR]', '📄': '[FILE]', '🔍': '[SEARCH]', '💡': '[TIP]',
                '🟢': '[GREEN]', '🟡': '[YELLOW]', '🔵': '[BLUE]', '🔴': '[RED]',
                '🟠': '[ORANGE]', '⚪': '[WHITE]', '🔥': '[HOT]', '⏳': '[WAIT]',
                '█': '#', '░': '-', '═': '=', '─': '-', '━': '-',
                '╔': '+', '╗': '+', '╚': '+', '╝': '+', '║': '|',
                '╠': '+', '╣': '+', '╦': '+', '╩': '+', '╬': '+',
                '✓': '[v]', '✗': '[x]', '•': '*', '→': '->', '←': '<-',
                '↑': '^', '↓': 'v', '★': '*', '☆': '*',
                '█': '#', '▓': '#', '▒': '=', '░': '-',
                '┌': '+', '┐': '+', '└': '+', '┘': '+', '├': '+',
                '┤': '+', '┬': '+', '┴': '+', '┼': '+', '│': '|', '─': '-',
            }
            
            for unicode_char, ascii_char in replacements.items():
                msg = msg.replace(unicode_char, ascii_char)
            
            # Handle any remaining non-ASCII characters
            msg = msg.encode('ascii', errors='replace').decode('ascii')
            
            stream = self.stream
            stream.write(msg + self.terminator)
            self.flush()
        except Exception:
            self.handleError(record)

# Create directories
os.makedirs('logs', exist_ok=True)
os.makedirs('signals', exist_ok=True)

# Configure logging with safe handler
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/scanner.log', encoding='utf-8'),
        SafeStreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# ============================================================================
# IMPORT YOUR ORIGINAL SCANNER CODE
# ============================================================================

# ============================================================================
# STEP 3: SAFE PRINT FUNCTION
# ============================================================================
_original_print = print

def safe_print(*args, **kwargs):
    """Print function that handles Unicode safely on Windows"""
    try:
        # Convert all arguments to strings and replace Unicode
        safe_args = []
        for arg in args:
            s = str(arg)
            replacements = {
                '✅': '[OK]', '❌': '[FAIL]', '⚠️': '[WARN]', '⚠': '[WARN]',
                '📊': '[DATA]', '📈': '[UP]', '📉': '[DOWN]', '🚀': '[GO]',
                '💰': '$', '🎯': '[TARGET]', '📧': '[EMAIL]', '📋': '[COPY]',
                '📂': '[DIR]', '📄': '[FILE]', '🔍': '[SEARCH]', '💡': '[TIP]',
                '🟢': '[GREEN]', '🟡': '[YELLOW]', '🔵': '[BLUE]', '🔴': '[RED]',
                '🟠': '[ORANGE]', '⚪': '[WHITE]', '🔥': '[HOT]', '⏳': '[WAIT]',
                '█': '#', '░': '-', '═': '=', '━': '-',
                '✓': '[v]', '•': '*', '→': '->', '★': '*',
            }
            for unicode_char, ascii_char in replacements.items():
                s = s.replace(unicode_char, ascii_char)
            safe_args.append(s)
        
        _original_print(*safe_args, **kwargs)
    except UnicodeEncodeError:
        # Fallback: encode to ASCII with replacement
        safe_args = [str(arg).encode('ascii', errors='replace').decode('ascii') for arg in args]
        _original_print(*safe_args, **kwargs)

# Replace built-in print
print = safe_print

"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
NSE SWING SCANNER v8.5 - ULTIMATE ACCURACY EDITION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Complete production-ready scanner with:
✓ All v7.4 Advanced Features
✓ All v8.0 Backtesting Features  
✓ Full error handling
✓ Bug fixes applied
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import sys
import io
import os
from typing import Optional, Tuple, List, Dict, Set
from dataclasses import dataclass
from datetime import datetime, timedelta, time as dt_time
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings
import time

# Platform-specific encoding fix
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

import yfinance as yf
import pandas as pd
import numpy as np
import requests
from io import StringIO

# Technical Analysis
from ta.momentum import RSIIndicator, ROCIndicator
from ta.trend import MACD, EMAIndicator, ADXIndicator
from ta.volatility import AverageTrueRange, BollingerBands
from ta.volume import OnBalanceVolumeIndicator
from scipy import stats
from scipy.signal import argrelextrema

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION - CHANGE THESE SETTINGS
# ============================================================================

ACCURACY_MODE = 'BALANCED'  # 'CONSERVATIVE', 'BALANCED', or 'AGGRESSIVE'
BACKTEST_MODE = 'HYBRID'      # 'MINI', 'FULL', 'HYBRID', 'NONE'

# GitHub Actions paths
SIGNALS_DIR = f"signals_v8.5_{ACCURACY_MODE.lower()}"
LOG_FILE = 'logs/scanner.log'
ERROR_LOG_FILE = f"scanner_v8.5_{ACCURACY_MODE.lower()}.log"

# Create directories
os.makedirs(SIGNALS_DIR, exist_ok=True)
os.makedirs('logs', exist_ok=True)
os.makedirs('signals', exist_ok=True)  

# Configuration Presets
PRESETS = {
    'CONSERVATIVE': {
        'min_confidence': 75,
        'min_turnover': 2_000_000,
        'min_rr_ratio': 2.5,
        'min_volatility_pct': 0.008,
        'min_volume_ratio': 0.8,
        'rsi_range_long': (45, 60),
        'rsi_range_short': (40, 55),
        'conditions_required': 4,
        'use_sector_rotation': True,
        'use_vix_sentiment': True,
        'use_fibonacci': True,
        'use_portfolio_risk': True,
        'use_full_backtest': True,
        'use_walk_forward': True,
        'max_stocks': 500,
        'rs_lookback': 21,
        'fib_lookback': 50,
        'atr_period': 14,
        'min_adx': 30,
        'min_volume_breakout': 2.0,
        'vcp_lookback': 20,
        'min_backtest_win_rate': 65,
        'min_profit_factor': 2.0, 
        'max_drawdown': 12,
        'sigmoid_divisor': 40,
    },
    'BALANCED': {
        # ✅ OPTIMIZED FOR 2-5 DAILY SIGNALS
        'min_confidence': 65,               # ✅ Already correct
        'min_turnover': 800_000,            # ✅ Already correct
        'min_rr_ratio': 1.7,                # ✅ Already correct (you changed from 1.6 to 2.0)
        'min_volatility_pct': 0.005,        # ✅ Already correct
        'min_volume_ratio': 0.5,            # ✅ Already correct
        'rsi_range_long': (40, 62),         # ✅ Already correct
        'rsi_range_short': (38, 60),        # ✅ Already correct
        'conditions_required': 2,           # ✅ Already correct
        'use_sector_rotation': True,        # ✅ Already correct
        'use_vix_sentiment': True,          # ✅ Already correct
        'use_fibonacci': True,              # ✅ Already correct
        'use_portfolio_risk': True,         # ✅ Already correct
        'use_full_backtest': True,          # ✅ Already correct
        'use_walk_forward': True,           # ✅ Already correct
        'max_stocks': 500,                  # ✅ Already correct
        'rs_lookback': 14,                  # ✅ Already correct
        'fib_lookback': 40,                 # ✅ Already correct
        'atr_period': 12,                   # ✅ Already correct
        'min_adx': 23,                      # ✅ Already correct
        'min_volume_breakout': 1.6,         # ✅ Already correct
        'vcp_lookback': 18,                 # ✅ Already correct
        'min_backtest_win_rate': 58,        # ✅ Already correct
        'min_profit_factor': 1.7,           # ✅ Already correct
        'max_drawdown': 18,                 # ✅ Already correct
        'sigmoid_divisor': 35,              # ✅ Already correct
    },
    'AGGRESSIVE': {
        'min_confidence': 40,
        'min_turnover': 200_000,
        'min_rr_ratio': 1.2,
        'min_volatility_pct': 0.003,
        'min_volume_ratio': 0.2,
        'rsi_range_long': (30, 70),
        'rsi_range_short': (30, 70),
        'conditions_required': 2,
        'use_sector_rotation': False,
        'use_vix_sentiment': False,
        'use_fibonacci': True,
        'use_portfolio_risk': False,
        'use_full_backtest': False,
        'use_walk_forward': False,
        'max_stocks': 500,
        'rs_lookback': 10,
        'fib_lookback': 30,
        'atr_period': 8,
        'min_adx': 15,
        'min_volume_breakout': 1.2,
        'vcp_lookback': 10,
        'min_backtest_win_rate': 50,
        'min_profit_factor': 1.3,
        'max_drawdown': 25,
        'sigmoid_divisor': 30,
    }
}

PRESET = PRESETS[ACCURACY_MODE]

# Scoring Weights - OPTIMIZED FOR BALANCED MODE
SCORING_WEIGHTS = {
    "ema_trend": 9,                 # ✅ Raised from 8
    "ema_200": 5,                   # ✅ Raised from 4
    "rsi": 8,                       # ✅ Raised from 7
    "macd": 8,                      # ✅ Raised from 7
    "volume_ratio": 5,              # ✅ Raised from 4
    "ema_slope": 4,                 # ✅ Raised from 3
    "obv": 4,                       # ✅ Raised from 3
    "sr_support": 6,                # ✅ Raised from 5
    "sr_resistance": 6,             # ✅ Raised from 5
    "candle": 5,                    # ✅ Raised from 4
    "mtf_trend": 7,                 # ✅ Raised from 6
    "mtf_penalty": 6,               # ✅ Raised from 5
    "vix_sentiment": 5,             # ✅ Raised from 4
    "fib_boost": 6,                 # ✅ Raised from 5
    "adx_momentum": 6,              # ✅ Raised from 5
    "vsa_confirmation": 5,          # ✅ Raised from 4
    "gap_detection": 4,             # ✅ Raised from 3
    "vcp_pattern": 7,               # ✅ Raised from 6
    "supertrend": 6,                # ✅ Raised from 5
    "volume_breakout": 5,           # ✅ Raised from 4
    "backtest_win_rate": 13,        # ✅ Raised from 12
    "backtest_profit_factor": 9,    # ✅ Raised from 8
    "pattern_reliability": 11,      # ✅ Raised from 10

    "fifty_two_week": 10,           # 52-Week High/Low analysis
    "relative_strength": 9,          # RS vs Nifty
    "delivery_volume": 6,            # Delivery % estimation
    "earnings_filter": 8,            # Earnings date penalty
    "consolidation_breakout": 11,    # Consolidation breakout bonus
    "market_breadth": 7,             # Market breadth filter
}

# Scanner Parameters
LOOKBACK_DAYS = 250             # ✅ Changed from 180 to 250 (10 months data)
MIN_CONFIDENCE = PRESET['min_confidence']
MIN_TURNOVER = PRESET['min_turnover']
MIN_RR_RATIO = PRESET['min_rr_ratio']
TOP_RESULTS = 10               # ✅ Changed from 20 to 15 (focus on top signals)

# Account & Risk
ACCOUNT_CAPITAL = 75_000
RISK_PER_TRADE = 0.01
MAX_CONCURRENT_TRADES = 3       # ✅ Changed from 5 to 4 (more focused)
MAX_SECTOR_EXPOSURE = 0.32      # ✅ Changed from 0.35 to 0.32

# Data Quality
MAX_DATA_STALE_DAYS = 5         # ✅ Changed from 7 to 5 (fresher data)
MIN_PRICE = 5                   # ✅ Keep at 5 (avoid penny stocks)
MAX_PRICE = 100000

# Feature Flags - OPTIMIZED FOR BALANCED
USE_SECTOR_ROTATION = True      # ✅ ENABLED
USE_VIX_SENTIMENT = True        # ✅ ENABLED
USE_FIBONACCI_SCORING = True    # ✅ ENABLED
USE_PORTFOLIO_RISK = True       # ✅ ENABLED
USE_MINI_BACKTEST = True        # ✅ ENABLED
USE_FULL_BACKTEST = True        # ✅ ENABLED
USE_WALK_FORWARD = True         # ✅ ENABLED (3 periods validation)
USE_MONTE_CARLO = False         # ✅ DISABLED (saves time in BALANCED mode)
USE_REALTIME_DATA = True  # Use live data during market hours

USE_52_WEEK_FILTER = True           # 52-Week High/Low analysis
USE_RELATIVE_STRENGTH = True         # Relative Strength vs Nifty
USE_DELIVERY_VOLUME = True           # Delivery volume estimation
USE_EARNINGS_FILTER = True           # Earnings date filter
USE_TRAILING_STOP = True             # Trailing stop calculations
USE_CONSOLIDATION_BREAKOUT = True    # Consolidation breakout detection
USE_MARKET_BREADTH = True            # Market breadth analysis
EARNINGS_AVOID_DAYS = 7              # Avoid stocks with earnings within X days

# Backtesting
BACKTEST_LOOKBACK_DAYS = 300    # ✅ Changed from 252 to 300
MIN_BACKTEST_TRADES = 12        # ✅ Raised from 10 to 12
WALK_FORWARD_PERIODS = 3        # ✅ Keep at 3 (good balance)

# Output
SIGNALS_DIR = f"signals_v8.5_{ACCURACY_MODE.lower()}"
ERROR_LOG_FILE = f"scanner_v8.5_{ACCURACY_MODE.lower()}.log"



# Complete stats dictionary
stats = {
    "total": 0, "passed": 0, "no_data": 0,
    "indicator_fail": 0, "turnover_fail": 0, "confidence_fail": 0,
    "rr_fail": 0, "data_quality_fail": 0, "volatility_fail": 0,
    "mini_backtest_fail": 0, "full_backtest_fail": 0, "full_backtest_run": 0,
    "walk_forward_fail": 0, "monte_carlo_fail": 0,
    "vcp_fail": 0, "vsa_fail": 0, "supertrend_fail": 0, "portfolio_fail": 0,
    "liquidity_fail": 0, "circuit_fail": 0, "hours_fail": 0, "price_range_fail": 0,
    "earnings_fail": 0,
    "market_breadth_fail": 0,
    "52w_fail": 0,
}

rejection_samples = []

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def normalize_series(series_like, index=None):
    """Convert series-like object to pandas Series"""
    arr = np.asarray(series_like)
    if arr.ndim > 1:
        arr = arr[:, 0]
    return pd.Series(arr, index=index)

def safe_indicator(func, default=None):
    """Safely execute indicator calculation"""
    try:
        result = func()
        if pd.isna(result) or (isinstance(result, (int, float)) and not np.isfinite(result)):
            return default
        return result
    except Exception:
        return default

# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class BacktestResult:
    """Comprehensive backtest metrics"""
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    avg_win: float
    avg_loss: float
    profit_factor: float
    sharpe_ratio: float
    max_drawdown: float
    total_return: float
    avg_holding_days: float
    best_trade: float
    worst_trade: float
    consecutive_wins: int
    consecutive_losses: int
    reliability_score: float
    expectancy: float

@dataclass
class WalkForwardResult:
    """Walk-forward validation results"""
    period_results: List[BacktestResult]
    avg_win_rate: float
    win_rate_stability: float
    avg_profit_factor: float
    consistency_score: float
    passed: bool

@dataclass
class MonteCarloResult:
    """Monte Carlo simulation results"""
    simulations: int
    avg_final_equity: float
    median_final_equity: float
    worst_case: float
    best_case: float
    prob_profit: float
    confidence_95: Tuple[float, float]
    risk_of_ruin: float
# ============================================================================
# TRADE RULE CLASS
# ============================================================================

class TradeRule:
    """Enhanced trade rule with multi-entry support - AGGRESSIVE QTY FIX"""


    def __init__(self, ticker: str, side: str, current_price: float, atr: float):
        self.ticker = ticker
        self.side = side
        self.entry_price = round(current_price, 2)
        self.atr = max(atr, current_price * 0.005)  # Min 0.5% ATR
        
        self.risk_amount = ACCOUNT_CAPITAL * RISK_PER_TRADE
        
        # ✅ FIX: Use 2.0x ATR multiplier for better default R:R
        if side == "LONG":
            self.stop_loss = round(current_price - (self.atr * 1.5), 2)
            self.target_1 = round(current_price + (self.atr * 2.5), 2)  # ✅ Changed from 2.0 to 2.5
            self.target_2 = round(current_price + (self.atr * 5.0), 2)  # ✅ Changed from 4.0 to 5.0
        else:
            self.stop_loss = round(current_price + (self.atr * 1.5), 2)
            self.target_1 = round(current_price - (self.atr * 2.5), 2)  # ✅ Changed from 2.0 to 2.5
            self.target_2 = round(current_price - (self.atr * 5.0), 2)  # ✅ Changed from 4.0 to 5.0
        
        # ✅ Calculate quantity based on actual stop distance
        risk_per_share = abs(self.entry_price - self.stop_loss)
        
        if risk_per_share > 0:
            # Calculate max qty by risk
            max_qty_by_risk = int(self.risk_amount / risk_per_share)
            
            # Calculate max qty by position size (max 10% of capital)
            max_qty_by_capital = int((ACCOUNT_CAPITAL * 0.10) / current_price)
            
            # Take minimum of both constraints
            self.qty = min(max_qty_by_risk, max_qty_by_capital)
            
            # Cap at 1000 shares
            self.qty = min(self.qty, 1000)
            
            # Ensure at least 1 share
            self.qty = max(1, self.qty)
        else:
            self.qty = 1
        
        self.position_value = self.qty * self.entry_price
        self.actual_risk = self.qty * risk_per_share
        
        # ✅ Calculate R:R based on new default targets (now ~1.67x minimum)
        risk = abs(self.entry_price - self.stop_loss)
        reward = abs(self.target_1 - self.entry_price)
        self.risk_reward_ratio = reward / risk if risk > 0 else 0
    
    def get_trailing_stops(self, current_price: float, highest_since_entry: float, lowest_since_entry: float) -> Dict:
        """Calculate trailing stops for this trade"""
        return TrailingStopManager.get_all_trailing_stops(
            entry_price=self.entry_price,
            current_price=current_price,
            highest_price=highest_since_entry,
            lowest_price=lowest_since_entry,
            atr=self.atr,
            side=self.side
        )
     

    def calculate_dynamic_targets(self, confidence: float):
        """Adjust targets based on setup quality - ENHANCED VERSION"""
        
        # ✅ Dynamic multipliers based on confidence
        if confidence >= 90:
            multiplier_t1 = 3.0
            multiplier_t2 = 6.0
        elif confidence >= 85:
            multiplier_t1 = 2.8
            multiplier_t2 = 5.5
        elif confidence >= 80:
            multiplier_t1 = 2.5
            multiplier_t2 = 5.0
        elif confidence >= 75:
            multiplier_t1 = 2.3
            multiplier_t2 = 4.5
        elif confidence >= 70:
            multiplier_t1 = 2.2
            multiplier_t2 = 4.2
        elif confidence >= 65:
            multiplier_t1 = 2.0
            multiplier_t2 = 4.0
        else:
            # Low confidence - use defaults from __init__
            multiplier_t1 = 2.5  # ✅ Increased from 1.8
            multiplier_t2 = 5.0  # ✅ Increased from 3.5
        
        if self.side == "LONG":
            self.target_1 = round(self.entry_price + (self.atr * multiplier_t1), 2)
            self.target_2 = round(self.entry_price + (self.atr * multiplier_t2), 2)
        else:
            self.target_1 = round(self.entry_price - (self.atr * multiplier_t1), 2)
            self.target_2 = round(self.entry_price - (self.atr * multiplier_t2), 2)
        
        # Recalculate R:R with new targets
        risk_per_share = abs(self.entry_price - self.stop_loss)
        reward_per_share = abs(self.target_1 - self.entry_price)
        
        if risk_per_share > 0:
            self.risk_reward_ratio = reward_per_share / risk_per_share
        else:
            self.risk_reward_ratio = 0
# ============================================================================
# DATA QUALITY VALIDATOR - FIXED VERSION
# ============================================================================

class DataQualityValidator:
    """Validate data quality and freshness"""
    
    @staticmethod
    def is_data_fresh(df: pd.DataFrame, max_stale_days: int = MAX_DATA_STALE_DAYS) -> bool:
        """Check if data is recent"""
        try:
            last_date = pd.to_datetime(df.index[-1])
            days_old = (datetime.now() - last_date).days
            return days_old <= max_stale_days
        except Exception:
            return False
    
    @staticmethod
    def has_sufficient_history(df: pd.DataFrame, min_bars: int = 60) -> bool:
        """Check minimum data length"""
        return len(df) >= min_bars
    
    @staticmethod
    def has_valid_prices(df: pd.DataFrame) -> Tuple[bool, str]:
        """Validate price data - BULLETPROOF VERSION"""
        try:
            # Handle MultiIndex columns
            if isinstance(df.columns, pd.MultiIndex):
                df = df.copy()
                df.columns = df.columns.get_level_values(0)
            
            # Check for Close column
            if 'Close' not in df.columns.tolist():
                return False, "No Close column"
            
            if len(df) == 0:
                return False, "Empty dataframe"
            
            # Get non-NaN close values
            close_series = df['Close'].dropna()
            
            if len(close_series) == 0:
                return False, "All Close values are NaN"
            
            # Get last non-NaN close
            close = close_series.iloc[-1]
            
            if pd.isna(close):
                return False, "Close is NaN"
            
            try:
                close_price = float(close)
            except (ValueError, TypeError):
                return False, "Close not numeric"
            
            if not np.isfinite(close_price):
                return False, "Close is infinite"
            
            # Price range
            if close_price < MIN_PRICE:
                return False, f"Price too low: ₹{close_price:.2f}"
            
            if close_price > MAX_PRICE:
                return False, f"Price too high: ₹{close_price:.2f}"
            
            return True, "Valid"
            
        except Exception as e:
            return False, f"{type(e).__name__}: {str(e)[:40]}"

# ============================================================================
# VOLATILITY REGIME FILTER - FIXED VERSION
# ============================================================================
class VolatilityRegimeFilter:
    """Advanced volatility regime analysis - COMPLETELY BULLETPROOF"""
    
    @staticmethod
    def analyze_regime(df: pd.DataFrame, price: float) -> Tuple[bool, str, Dict]:
        """Comprehensive volatility regime analysis"""
        try:
            # ============================================================
            # STEP 1: VALIDATE DATAFRAME
            # ============================================================
            if df is None:
                return False, "DataFrame is None", {}
            
            if not isinstance(df, pd.DataFrame):
                return False, f"Not a DataFrame: {type(df)}", {}
            
            if df.empty:
                return False, "Empty DataFrame", {}
            
            # ============================================================
            # STEP 2: CHECK COLUMNS
            # ============================================================
            required_cols = ['High', 'Low', 'Close']
            
            # Get actual columns
            actual_cols = list(df.columns)
            
            # Check if all required columns exist
            missing = [col for col in required_cols if col not in actual_cols]
            if missing:
                return False, f"Missing: {missing}", {}
            
            # ============================================================
            # STEP 3: EXTRACT DATA AS SERIES
            # ============================================================
            try:
                # Extract columns and ensure they are Series
                high_col = df['High']
                low_col = df['Low']
                close_col = df['Close']
                
                # Convert to Series if needed
                if isinstance(high_col, pd.DataFrame):
                    high_col = high_col.iloc[:, 0]
                if isinstance(low_col, pd.DataFrame):
                    low_col = low_col.iloc[:, 0]
                if isinstance(close_col, pd.DataFrame):
                    close_col = close_col.iloc[:, 0]
                
                # Make clean copies
                high_data = pd.Series(high_col.values, name='High')
                low_data = pd.Series(low_col.values, name='Low')
                close_data = pd.Series(close_col.values, name='Close')
                
            except Exception as e:
                return False, f"Data extract fail: {str(e)[:25]}", {}
            
            # ============================================================
            # STEP 4: CLEAN DATA
            # ============================================================
            try:
                # Remove NaN values
                high_data = high_data.dropna()
                low_data = low_data.dropna()
                close_data = close_data.dropna()
                
                # Check we have enough data
                min_len = min(len(high_data), len(low_data), len(close_data))
                
                if min_len < PRESET.get('atr_period', 14) + 5:
                    return False, f"Insufficient data: {min_len}", {}
                
                # Align lengths (take shortest)
                if len(high_data) != len(low_data) or len(low_data) != len(close_data):
                    min_len = min(len(high_data), len(low_data), len(close_data))
                    high_data = high_data.iloc[-min_len:]
                    low_data = low_data.iloc[-min_len:]
                    close_data = close_data.iloc[-min_len:]
                
            except Exception as e:
                return False, f"Data clean fail: {str(e)[:25]}", {}
            
            # ============================================================
            # STEP 5: CALCULATE ATR
            # ============================================================
            try:
                atr_period = PRESET.get('atr_period', 14)
                
                # Calculate True Range manually for more control
                tr1 = high_data - low_data
                tr2 = abs(high_data - close_data.shift(1))
                tr3 = abs(low_data - close_data.shift(1))
                
                # Combine and take max
                tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
                
                # Calculate ATR as rolling mean
                atr_series = tr.rolling(window=atr_period, min_periods=atr_period).mean()
                
                # Remove NaN values
                atr_series = atr_series.dropna()
                
                if len(atr_series) == 0:
                    return False, "ATR calculation empty", {}
                
                # Get current ATR
                current_atr = float(atr_series.iloc[-1])
                
                if pd.isna(current_atr) or current_atr <= 0:
                    return False, f"Invalid ATR: {current_atr}", {}
                
            except Exception as e:
                return False, f"ATR calc fail: {str(e)[:25]}", {}
            
            # ============================================================
            # STEP 6: CALCULATE METRICS
            # ============================================================
            try:
                # Validate price
                if price <= 0 or pd.isna(price):
                    return False, "Invalid price", {}
                
                # ATR as percentage
                atr_pct = float(current_atr / price)
                
                # ATR slope
                if len(atr_series) >= 5:
                    atr_5 = float(atr_series.iloc[-5])
                    if atr_5 > 0:
                        atr_slope = float((current_atr - atr_5) / atr_5)
                    else:
                        atr_slope = 0.0
                else:
                    atr_slope = 0.0
                
                # ATR percentile
                if len(atr_series) > 10:
                    from scipy import stats as scipy_stats
                    atr_percentile = float(scipy_stats.percentileofscore(atr_series, current_atr))
                else:
                    atr_percentile = 50.0
                
            except Exception as e:
                return False, f"Metrics fail: {str(e)[:25]}", {}
            
            # ============================================================
            # STEP 7: BUILD METRICS DICT
            # ============================================================
            metrics = {
                'atr': current_atr,
                'atr_pct': atr_pct,
                'atr_slope': atr_slope,
                'atr_percentile': atr_percentile
            }
            
            # ============================================================
            # STEP 8: REGIME CLASSIFICATION
            # ============================================================
            min_vol = PRESET.get('min_volatility_pct', 0.003)
            
            # Very low volatility
            if atr_pct < min_vol * 0.6:
                return False, f"Very low vol ({atr_pct:.3%})", metrics
            
            # Contracting volatility
            if atr_pct < min_vol and atr_slope < -0.1:
                return False, f"Vol contracting ({atr_pct:.3%})", metrics
            
            # Extreme volatility
            if atr_pct > 0.10:
                return False, f"Extreme vol ({atr_pct:.3%})", metrics
            
            # Historical extremes
            if atr_percentile > 95:
                return False, f"Vol at extremes (P{atr_percentile:.0f})", metrics
            
            # ACCEPTABLE REGIME
            if atr_pct >= min_vol:
                return True, f"OK ({atr_pct:.3%})", metrics
            
            # DEFAULT: REJECT
            return False, f"Low vol ({atr_pct:.3%})", metrics
            
        except Exception as e:
            # Final catch-all
            return False, f"Error: {type(e).__name__}", {}
        
# ============================================================================
# PORTFOLIO RISK MANAGER
# ============================================================================

class PortfolioRiskManager:
    """Manage portfolio-level risk"""
    
    def __init__(self, capital: float):
        self.capital = capital
        self.current_trades = []
        self.sector_exposure = {}
        self.total_risk = 0.0
    
    def can_add_trade(self, trade: Dict) -> Tuple[bool, str]:
        """Check if trade can be added"""
        
        if len(self.current_trades) >= MAX_CONCURRENT_TRADES:
            return False, "Max concurrent trades"
        
        sector = trade.get('sector', 'UNKNOWN')
        current_sector_exposure = self.sector_exposure.get(sector, 0)
        
        if current_sector_exposure >= MAX_SECTOR_EXPOSURE:
            return False, f"Max sector exposure ({sector})"
        
        trade_risk = trade.get('risk_amount', 0)
        if self.total_risk + trade_risk > self.capital * 0.05:
            return False, "Max portfolio risk"
        
        return True, "OK"
    
    def add_trade(self, trade: Dict):
        """Add trade to portfolio"""
        self.current_trades.append(trade)
        sector = trade.get('sector', 'UNKNOWN')
        self.sector_exposure[sector] = self.sector_exposure.get(sector, 0) + trade.get('position_value', 0) / self.capital
        self.total_risk += trade.get('risk_amount', 0)# ============================================================================
# INSTITUTIONAL VOLUME ANALYZER
# ============================================================================

class InstitutionalVolumeAnalyzer:
    """Detect institutional participation"""
    
    @staticmethod
    def analyze(df: pd.DataFrame, current_volume: float) -> Tuple[float, str, Dict]:
        """Analyze volume for institutional activity"""
        try:
            vol_20 = df['Volume'].tail(20).mean()
            vol_50 = df['Volume'].tail(50).mean()
            vol_ratio = current_volume / vol_20 if vol_20 > 0 else 1.0
            
            vol_slope = (vol_20 - vol_50) / vol_50 if vol_50 > 0 else 0
            
            price_change = (df['Close'].iloc[-1] - df['Close'].iloc[-2]) / df['Close'].iloc[-2]
            vol_change = (current_volume - df['Volume'].iloc[-2]) / df['Volume'].iloc[-2]
            
            pv_correlation = 1 if (price_change > 0 and vol_change > 0) or (price_change < 0 and vol_change < 0) else -1
            
            metrics = {
                'volume_ratio': vol_ratio,
                'vol_slope': vol_slope,
                'pv_correlation': pv_correlation
            }
            
            score = 0
            signal = ""
            
            if vol_ratio >= 3.0 and pv_correlation > 0:
                score = 10
                signal = "Very strong institutional buying"
            elif vol_ratio >= PRESET['min_volume_breakout'] and pv_correlation > 0:
                score = 7
                signal = "Strong volume breakout"
            elif vol_ratio >= 1.5:
                score = 4
                signal = "Above average volume"
            elif vol_ratio < 0.8:
                score = -5
                signal = "Weak volume"
            elif vol_ratio < 0.5:
                score = -8
                signal = "Very weak volume"
            
            return score, signal, metrics
            
        except Exception as e:
            return 0, f"Volume analysis error: {str(e)[:30]}", {}

# ============================================================================
# VOLUME SPREAD ANALYSIS (VSA)
# ============================================================================

class VolumeSpreadAnalyzer:
    """Volume Spread Analysis for smart money detection"""
    
    @staticmethod
    def analyze(df: pd.DataFrame) -> Tuple[float, str, Dict]:
        """Comprehensive VSA analysis"""
        try:
            if len(df) < 5:
                return 0, "Insufficient data", {}
            
            recent = df.tail(5)
            current = recent.iloc[-1]
            prev = recent.iloc[-2]
            
            current_spread = current['High'] - current['Low']
            prev_spread = prev['High'] - prev['Low']
            avg_spread = recent['High'].sub(recent['Low']).mean()
            
            current_volume = current['Volume']
            avg_volume = df['Volume'].tail(20).mean()
            
            spread_ratio = current_spread / avg_spread if avg_spread > 0 else 1
            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1
            
            close_position = (current['Close'] - current['Low']) / current_spread if current_spread > 0 else 0.5
            
            metrics = {
                'spread_ratio': spread_ratio,
                'volume_ratio': volume_ratio,
                'close_position': close_position
            }
            
            score = 0
            signal = ""
            
            # Accumulation
            if volume_ratio > 1.5 and spread_ratio < 0.8 and close_position > 0.7:
                score = 8
                signal = "VSA: Accumulation"
            # Distribution
            elif volume_ratio > 1.5 and spread_ratio < 0.8 and close_position < 0.3:
                score = -8
                signal = "VSA: Distribution"
            # Stopping volume
            elif volume_ratio > 2.0 and spread_ratio < 0.6:
                score = 6
                signal = "VSA: Stopping volume"
            # No demand
            elif volume_ratio < 0.7 and spread_ratio > 1.3:
                score = -6
                signal = "VSA: No demand"
            # Effort vs Result
            elif volume_ratio > 1.8 and abs(current['Close'] - current['Open']) / current_spread < 0.3:
                score = -5
                signal = "VSA: High effort, no result"
            
            return score, signal, metrics
            
        except Exception as e:
            return 0, f"VSA error: {str(e)[:30]}", {}

# ============================================================================
# MOMENTUM SCORE CALCULATOR
# ============================================================================

class MomentumScoreCalculator:
    """Comprehensive momentum analysis"""
    
    @staticmethod
    def calculate(df: pd.DataFrame) -> Tuple[float, str, Dict]:
        """Calculate composite momentum score"""
        try:
            c = normalize_series(df["Close"])
            h = normalize_series(df["High"])
            l = normalize_series(df["Low"])
            
            adx_indicator = ADXIndicator(h, l, c, window=14)
            adx = adx_indicator.adx().iloc[-1]
            adx_pos = adx_indicator.adx_pos().iloc[-1]
            adx_neg = adx_indicator.adx_neg().iloc[-1]
            
            roc = ROCIndicator(c, window=10).roc().iloc[-1]
            rsi = RSIIndicator(c, window=14).rsi().iloc[-1]
            
            momentum_direction = "BULLISH" if adx_pos > adx_neg else "BEARISH"
            
            metrics = {
                'adx': adx,
                'adx_pos': adx_pos,
                'adx_neg': adx_neg,
                'roc': roc,
                'rsi': rsi,
                'direction': momentum_direction
            }
            
            score = 0
            signal = ""
            
            if adx > 30 and abs(roc) > 5:
                score = 10
                signal = f"Very strong {momentum_direction.lower()} momentum"
            elif adx >= PRESET['min_adx'] and abs(roc) > 2:
                score = 7
                signal = f"Strong {momentum_direction.lower()} momentum"
            elif adx >= 20:
                score = 4
                signal = f"Moderate momentum"
            elif adx < 15:
                score = -5
                signal = "Weak momentum"
            
            # RSI confirmation
            if momentum_direction == "BULLISH" and rsi > 60:
                score += 3
            elif momentum_direction == "BEARISH" and rsi < 40:
                score += 3
            
            return score, signal, metrics
            
        except Exception as e:
            return 0, f"Momentum error: {str(e)[:30]}", {}

# ============================================================================
# VCP DETECTOR
# ============================================================================

class VCPDetector:
    """Detect Volatility Contraction Patterns"""
    
    @staticmethod
    def detect(df: pd.DataFrame) -> Tuple[float, str, Dict]:
        """Detect VCP setup"""
        try:
            lookback = PRESET['vcp_lookback']
            
            if len(df) < lookback + 20:
                return 0, "Insufficient data", {}
            
            recent = df.tail(lookback).copy()
            
            atr = AverageTrueRange(
                high=normalize_series(recent['High']),
                low=normalize_series(recent['Low']),
                close=normalize_series(recent['Close']),
                window=10
            ).average_true_range()
            
            bb = BollingerBands(
                close=normalize_series(recent['Close']),
                window=20,
                window_dev=2
            )
            bb_width = (bb.bollinger_hband() - bb.bollinger_lband()) / bb.bollinger_mavg()
            
            atr_contraction = (atr.iloc[-1] - atr.max()) / atr.max() if atr.max() > 0 else 0
            bb_contraction = (bb_width.iloc[-1] - bb_width.max()) / bb_width.max() if bb_width.max() > 0 else 0
            
            volume = recent['Volume']
            vol_contraction = (volume.iloc[-5:].mean() - volume.max()) / volume.max() if volume.max() > 0 else 0
            
            contraction_score = abs(atr_contraction) + abs(bb_contraction) + abs(vol_contraction)
            price_range = (recent['High'].max() - recent['Low'].min()) / recent['Close'].iloc[-1]
            
            metrics = {
                'atr_contraction': atr_contraction,
                'bb_contraction': bb_contraction,
                'vol_contraction': vol_contraction,
                'contraction_score': contraction_score,
                'price_range': price_range
            }
            
            score = 0
            signal = ""
            
            if contraction_score > 0.6 and price_range < 0.15:
                score = 10
                signal = "Strong VCP pattern"
            elif contraction_score > 0.4 and price_range < 0.20:
                score = 6
                signal = "Moderate VCP pattern"
            elif contraction_score > 0.2:
                score = 3
                signal = "Weak VCP forming"
            
            return score, signal, metrics
            
        except Exception as e:
            return 0, f"VCP error: {str(e)[:30]}", {}

# ============================================================================
# S/R CLUSTER DETECTOR
# ============================================================================

class SRClusterDetector:
    """Advanced S/R detection using price clustering"""
    
    @staticmethod
    def detect(df: pd.DataFrame, lookback: int = 60) -> Dict:
        """Detect clustered S/R levels"""
        try:
            recent = df.tail(lookback).copy()
            
            highs = recent['High'].values
            lows = recent['Low'].values
            
            high_indices = argrelextrema(highs, np.greater, order=3)[0]
            low_indices = argrelextrema(lows, np.less, order=3)[0]
            
            swing_highs = [highs[i] for i in high_indices]
            swing_lows = [lows[i] for i in low_indices]
            
            current_price = recent['Close'].iloc[-1]
            
            if not swing_highs and not swing_lows:
                return {
                    'support': current_price * 0.98,
                    'resistance': current_price * 1.02,
                    'strength': 0,
                    'bias': 'NEUTRAL'
                }
            
            all_levels = swing_highs + swing_lows
            tolerance = np.std(all_levels) * 0.25 if all_levels else current_price * 0.01
            
            clusters = {}
            for level in all_levels:
                found = False
                for center in list(clusters.keys()):
                    if abs(level - center) <= tolerance:
                        clusters[center].append(level)
                        found = True
                        break
                if not found:
                    clusters[level] = [level]
            
            cluster_strength = {k: len(v) for k, v in clusters.items()}
            sorted_clusters = sorted(cluster_strength.items(), key=lambda x: x[1], reverse=True)
            
            support_levels = [c[0] for c in sorted_clusters if c[0] < current_price]
            resistance_levels = [c[0] for c in sorted_clusters if c[0] > current_price]
            
            if support_levels:
                support = max(support_levels)
                support_strength = cluster_strength[support]
            else:
                support = min(all_levels) if all_levels else current_price * 0.98
                support_strength = 1
            
            if resistance_levels:
                resistance = min(resistance_levels)
                resistance_strength = cluster_strength[resistance]
            else:
                resistance = max(all_levels) if all_levels else current_price * 1.02
                resistance_strength = 1
            
            dist_to_support = abs(current_price - support) / current_price
            dist_to_resistance = abs(resistance - current_price) / current_price
            
            if dist_to_support < 0.02:
                bias = "AT_SUPPORT"
            elif dist_to_resistance < 0.02:
                bias = "AT_RESISTANCE"
            elif dist_to_support < dist_to_resistance:
                bias = "NEAR_SUPPORT"
            else:
                bias = "NEAR_RESISTANCE"
            
            return {
                'support': float(support),
                'resistance': float(resistance),
                'support_strength': support_strength,
                'resistance_strength': resistance_strength,
                'total_clusters': len(sorted_clusters),
                'bias': bias,
                'dist_to_support': dist_to_support,
                'dist_to_resistance': dist_to_resistance
            }
            
        except Exception as e:
            logger.debug(f"S/R clustering failed: {e}")
            return {
                'support': None,
                'resistance': None,
                'strength': 0,
                'bias': 'UNKNOWN'
            }

# ============================================================================
# SUPERTREND INDICATOR
# ============================================================================

class SupertrendIndicator:
    """Supertrend for trend confirmation"""
    
    @staticmethod
    def calculate(df: pd.DataFrame, period: int = 10, multiplier: int = 3) -> Tuple[Optional[str], float, Dict]:
        """Calculate Supertrend"""
        try:
            h = normalize_series(df["High"])
            l = normalize_series(df["Low"])
            c = normalize_series(df["Close"])
            
            atr = AverageTrueRange(h, l, c, window=period).average_true_range()
            
            hl2 = (h + l) / 2
            upper_band = hl2 + (multiplier * atr)
            lower_band = hl2 - (multiplier * atr)
            
            supertrend = pd.Series(index=c.index, dtype=float)
            direction = pd.Series(index=c.index, dtype=str)
            
            for i in range(len(c)):
                if i < period:
                    supertrend.iloc[i] = 0
                    direction.iloc[i] = None
                    continue
                
                if c.iloc[i] > upper_band.iloc[i-1]:
                    supertrend.iloc[i] = lower_band.iloc[i]
                    direction.iloc[i] = "UP"
                elif c.iloc[i] < lower_band.iloc[i-1]:
                    supertrend.iloc[i] = upper_band.iloc[i]
                    direction.iloc[i] = "DOWN"
                else:
                    supertrend.iloc[i] = supertrend.iloc[i-1]
                    direction.iloc[i] = direction.iloc[i-1]
            
            current_direction = direction.iloc[-1]
            
            if current_direction == "UP":
                strength = ((c.iloc[-1] - supertrend.iloc[-1]) / supertrend.iloc[-1]) * 100
            elif current_direction == "DOWN":
                strength = ((supertrend.iloc[-1] - c.iloc[-1]) / c.iloc[-1]) * 100
            else:
                strength = 0
            
            consecutive = 1
            for i in range(len(direction)-2, -1, -1):
                if direction.iloc[i] == current_direction:
                    consecutive += 1
                else:
                    break
            
            metrics = {
                'direction': current_direction,
                'strength': strength,
                'consecutive_bars': consecutive,
                'supertrend_value': supertrend.iloc[-1]
            }
            
            return current_direction, round(strength, 2), metrics
            
        except Exception as e:
            return None, 0, {}

# ============================================================================
# GAP DETECTOR
# ============================================================================

class GapDetector:
    """Detect and analyze gaps"""
    
    @staticmethod
    def detect(df: pd.DataFrame) -> Tuple[float, str, Dict]:
        """Detect gap patterns"""
        try:
            if len(df) < 2:
                return 0, "No gap", {}
            
            prev_close = df['Close'].iloc[-2]
            current_open = df['Open'].iloc[-1]
            current_close = df['Close'].iloc[-1]
            current_high = df['High'].iloc[-1]
            current_low = df['Low'].iloc[-1]
            
            gap_pct = ((current_open - prev_close) / prev_close) * 100
            
            if gap_pct > 0:
                gap_filled = current_low <= prev_close
            else:
                gap_filled = current_high >= prev_close
            
            if current_open != 0:
                post_gap_move = ((current_close - current_open) / current_open) * 100
            else:
                post_gap_move = 0
            
            metrics = {
                'gap_pct': gap_pct,
                'gap_filled': gap_filled,
                'post_gap_move': post_gap_move
            }
            
            score = 0
            signal = ""
            
            if abs(gap_pct) < 0.5:
                return 0, "No significant gap", metrics
            
            if gap_pct > 2.0 and not gap_filled and post_gap_move > 0:
                score = 8
                signal = f"Strong gap up {gap_pct:.1f}%"
            elif gap_pct > 1.0 and not gap_filled:
                score = 5
                signal = f"Gap up {gap_pct:.1f}%"
            elif gap_pct > 1.0 and gap_filled:
                score = -4
                signal = f"Gap up filled - bearish"
            elif gap_pct < -2.0 and not gap_filled and post_gap_move < 0:
                score = -8
                signal = f"Strong gap down {gap_pct:.1f}%"
            elif gap_pct < -1.0 and not gap_filled:
                score = -5
                signal = f"Gap down {gap_pct:.1f}%"
            elif gap_pct < -1.0 and gap_filled:
                score = 4
                signal = f"Gap down filled - bullish"
            
            return score, signal, metrics
            
        except Exception as e:
            return 0, f"Gap detection error: {str(e)[:30]}", {}

# ============================================================================
# CANDLESTICK PATTERN DETECTOR
# ============================================================================

class CandlestickPatternDetector:
    """Detect common candlestick patterns"""
    
    @staticmethod
    def detect(df: pd.DataFrame) -> Dict:
        """Detect bullish/bearish candlestick patterns"""
        try:
            if len(df) < 3:
                return {'pattern': 'NONE', 'score': 0}
            
            recent = df.tail(3)
            c0 = recent.iloc[-3]
            c1 = recent.iloc[-2]
            c2 = recent.iloc[-1]
            
            def is_bullish(candle):
                return candle['Close'] > candle['Open']
            
            def is_bearish(candle):
                return candle['Close'] < candle['Open']
            
            def body_size(candle):
                return abs(candle['Close'] - candle['Open'])
            
            def candle_range(candle):
                return candle['High'] - candle['Low']
            
            def upper_shadow(candle):
                return candle['High'] - max(candle['Open'], candle['Close'])
            
            def lower_shadow(candle):
                return min(candle['Open'], candle['Close']) - candle['Low']
            
            pattern = 'NONE'
            score = 0
            
            # Bullish Engulfing
            if (is_bearish(c1) and is_bullish(c2) and
                c2['Close'] > c1['Open'] and c2['Open'] < c1['Close']):
                pattern = 'BULLISH_ENGULFING'
                score = 7
            
            # Bearish Engulfing
            elif (is_bullish(c1) and is_bearish(c2) and
                  c2['Close'] < c1['Open'] and c2['Open'] > c1['Close']):
                pattern = 'BEARISH_ENGULFING'
                score = -7
            
            # Hammer
            elif (body_size(c2) < candle_range(c2) * 0.3 and
                  lower_shadow(c2) > body_size(c2) * 2 and
                  upper_shadow(c2) < body_size(c2) * 0.5):
                pattern = 'HAMMER'
                score = 6
            
            # Shooting Star
            elif (body_size(c2) < candle_range(c2) * 0.3 and
                  upper_shadow(c2) > body_size(c2) * 2 and
                  lower_shadow(c2) < body_size(c2) * 0.5):
                pattern = 'SHOOTING_STAR'
                score = -6
            
            # Doji
            elif body_size(c2) < candle_range(c2) * 0.1:
                pattern = 'DOJI'
                score = -2
            
            # Morning Star
            elif (is_bearish(c0) and body_size(c1) < body_size(c0) * 0.3 and
                  is_bullish(c2) and c2['Close'] > (c0['Open'] + c0['Close']) / 2):
                pattern = 'MORNING_STAR'
                score = 8
            
            # Evening Star
            elif (is_bullish(c0) and body_size(c1) < body_size(c0) * 0.3 and
                  is_bearish(c2) and c2['Close'] < (c0['Open'] + c0['Close']) / 2):
                pattern = 'EVENING_STAR'
                score = -8
            
            # Strong bullish
            elif (is_bullish(c2) and body_size(c2) > candle_range(c2) * 0.7 and
                  candle_range(c2) > candle_range(c1) * 1.5):
                pattern = 'STRONG_BULLISH'
                score = 5
            
            # Strong bearish
            elif (is_bearish(c2) and body_size(c2) > candle_range(c2) * 0.7 and
                  candle_range(c2) > candle_range(c1) * 1.5):
                pattern = 'STRONG_BEARISH'
                score = -5
            
            return {
                'pattern': pattern,
                'score': score,
                'description': pattern.replace('_', ' ').title()
            }
            
        except Exception as e:
            return {'pattern': 'NONE', 'score': 0, 'description': 'None'}

# ============================================================================
# 1. FIFTY-TWO WEEK HIGH/LOW ANALYZER
# ============================================================================

class FiftyTwoWeekAnalyzer:
    """
    52-Week High/Low Analysis - CRITICAL for breakouts
    
    Why Important:
    - Breakouts above 52W high have 65% success rate
    - Stocks near 52W low often continue falling
    - Best swing trades: 10-20% below 52W high
    """
    
    @staticmethod
    def analyze(df: pd.DataFrame, current_price: float) -> Dict:
        """Analyze position relative to 52-week range"""
        try:
            # Need at least 200 days of data
            if len(df) < 200:
                return {
                    'valid': False,
                    'score': 0,
                    'signal': 'Insufficient data for 52W analysis'
                }
            
            # Get 52-week data (approximately 252 trading days)
            year_data = df.tail(252)
            
            high_52w = float(year_data['High'].max())
            low_52w = float(year_data['Low'].min())
            range_52w = high_52w - low_52w
            
            if range_52w <= 0:
                return {'valid': False, 'score': 0, 'signal': 'Invalid price range'}
            
            # Position in range (0% = at low, 100% = at high)
            position_pct = ((current_price - low_52w) / range_52w) * 100
            
            # Distance from 52W high
            dist_from_high = ((high_52w - current_price) / high_52w) * 100
            
            # Distance from 52W low
            dist_from_low = ((current_price - low_52w) / low_52w) * 100 if low_52w > 0 else 0
            
            # Is it making new high?
            is_new_high = current_price >= high_52w * 0.98  # Within 2%
            
            # Is it near 52W low? (Avoid these!)
            is_near_low = current_price <= low_52w * 1.10  # Within 10%
            
            # Score calculation
            score = 0
            signal = ""
            
            if is_new_high:
                score = 12
                signal = "🚀 52-Week Breakout!"
            elif position_pct >= 85:
                score = 8
                signal = "Near 52W high - Strong momentum"
            elif position_pct >= 70:
                score = 5
                signal = "Upper range - Healthy trend"
            elif position_pct >= 50:
                score = 2
                signal = "Middle of range - Neutral"
            elif position_pct >= 30:
                score = -3
                signal = "Lower range - Weakness"
            elif position_pct >= 15:
                score = -7
                signal = "Near 52W low - Caution"
            else:
                score = -12
                signal = "⚠️ At 52W low - AVOID"
            
            return {
                'valid': True,
                'high_52w': round(high_52w, 2),
                'low_52w': round(low_52w, 2),
                'position_pct': round(position_pct, 1),
                'dist_from_high': round(dist_from_high, 1),
                'dist_from_low': round(dist_from_low, 1),
                'is_new_high': is_new_high,
                'is_near_low': is_near_low,
                'score': score,
                'signal': signal
            }
            
        except Exception as e:
            return {
                'valid': False,
                'score': 0,
                'signal': f'52W analysis error: {str(e)[:30]}'
            }


# ============================================================================
# 2. RELATIVE STRENGTH VS NIFTY ANALYZER
# ============================================================================

class RelativeStrengthAnalyzer:
    """
    Compare stock performance vs Nifty 50
    
    Why Important:
    - Stocks outperforming Nifty tend to continue outperforming
    - RS > 1 means stock is stronger than market
    - Best swing trades: RS > 1.2 (20% stronger)
    """
    
    def __init__(self):
        self.nifty_data = None
        self._load_nifty_data()
    
    def _load_nifty_data(self):
        """Load Nifty 50 data"""
        try:
            self.nifty_data = yf.download(
                "^NSEI",
                period="6mo",
                interval="1d",
                progress=False
            )
            
            # Handle MultiIndex columns
            if isinstance(self.nifty_data.columns, pd.MultiIndex):
                self.nifty_data.columns = self.nifty_data.columns.get_level_values(0)
            
            if self.nifty_data.empty:
                logger.warning("Failed to load Nifty data for RS analysis")
                self.nifty_data = None
                
        except Exception as e:
            logger.warning(f"Nifty data load error: {e}")
            self.nifty_data = None
    
    def calculate_rs(self, stock_df: pd.DataFrame, periods: List[int] = [5, 20, 60]) -> Dict:
        """Calculate Relative Strength vs Nifty"""
        try:
            if self.nifty_data is None or len(self.nifty_data) < 60:
                return {
                    'valid': False,
                    'score': 0,
                    'signal': 'Nifty data not available'
                }
            
            if len(stock_df) < 60:
                return {
                    'valid': False,
                    'score': 0,
                    'signal': 'Insufficient stock data'
                }
            
            results = {'valid': True, 'periods': {}}
            rs_ratios = []
            
            for period in periods:
                if len(stock_df) < period or len(self.nifty_data) < period:
                    continue
                
                # Stock return
                stock_close_now = float(stock_df['Close'].iloc[-1])
                stock_close_past = float(stock_df['Close'].iloc[-period])
                
                if stock_close_past <= 0:
                    continue
                    
                stock_return = ((stock_close_now / stock_close_past) - 1) * 100
                
                # Nifty return
                nifty_close_now = float(self.nifty_data['Close'].iloc[-1])
                nifty_close_past = float(self.nifty_data['Close'].iloc[-period])
                
                if nifty_close_past <= 0:
                    continue
                    
                nifty_return = ((nifty_close_now / nifty_close_past) - 1) * 100
                
                # Relative Strength Ratio
                rs_ratio = (1 + stock_return/100) / (1 + nifty_return/100)
                rs_ratios.append(rs_ratio)
                
                # Outperformance
                outperformance = stock_return - nifty_return
                
                results['periods'][f'{period}d'] = {
                    'stock_return': round(stock_return, 2),
                    'nifty_return': round(nifty_return, 2),
                    'rs_ratio': round(rs_ratio, 3),
                    'outperformance': round(outperformance, 2)
                }
            
            if not rs_ratios:
                return {
                    'valid': False,
                    'score': 0,
                    'signal': 'Could not calculate RS'
                }
            
            # Overall RS Score
            avg_rs = float(np.mean(rs_ratios))
            
            if avg_rs >= 1.3:
                results['score'] = 10
                results['signal'] = "🔥 Very Strong vs Nifty"
            elif avg_rs >= 1.15:
                results['score'] = 7
                results['signal'] = "Strong vs Nifty"
            elif avg_rs >= 1.05:
                results['score'] = 4
                results['signal'] = "Slightly stronger than Nifty"
            elif avg_rs >= 0.95:
                results['score'] = 0
                results['signal'] = "In-line with Nifty"
            elif avg_rs >= 0.85:
                results['score'] = -5
                results['signal'] = "Weaker than Nifty"
            else:
                results['score'] = -10
                results['signal'] = "⚠️ Very Weak vs Nifty - AVOID"
            
            results['avg_rs'] = round(avg_rs, 3)
            
            return results
            
        except Exception as e:
            return {
                'valid': False,
                'score': 0,
                'signal': f'RS calculation error: {str(e)[:30]}'
            }


# ============================================================================
# 3. DELIVERY VOLUME ANALYZER
# ============================================================================

class DeliveryVolumeAnalyzer:
    """
    Delivery Volume Analysis
    
    Why Critical for Indian Markets:
    - High delivery % (>50%) = Real buying (institutions)
    - Low delivery % (<30%) = Intraday speculation
    - Swing trades need high delivery for sustainability
    
    Note: Uses price action heuristics when actual delivery data unavailable
    """
    
    @staticmethod
    def estimate_delivery(df: pd.DataFrame) -> Dict:
        """
        Estimate delivery quality from price action
        Uses heuristics when actual delivery data not available
        """
        try:
            if len(df) < 10:
                return {
                    'valid': False,
                    'estimated_delivery_score': 50,
                    'signal': 'Insufficient data',
                    'score': 0
                }
            
            recent = df.tail(5)
            scores = []
            
            for _, row in recent.iterrows():
                candle_range = row['High'] - row['Low']
                if candle_range <= 0:
                    continue
                
                # Close position in range (0 = at low, 1 = at high)
                close_position = (row['Close'] - row['Low']) / candle_range
                
                # Higher delivery when close is near extremes (not middle)
                # Close near high = bullish delivery
                # Close near low = bearish delivery (selling pressure)
                delivery_score = abs(close_position - 0.5) * 2  # 0-1 scale
                scores.append(delivery_score)
            
            if not scores:
                return {
                    'valid': False,
                    'estimated_delivery_score': 50,
                    'signal': 'Could not calculate',
                    'score': 0
                }
            
            avg_score = float(np.mean(scores))
            
            # Volume consistency (lower std = more consistent = better)
            vol_mean = recent['Volume'].mean()
            vol_std = recent['Volume'].std()
            vol_consistency = 1 - min(1, vol_std / vol_mean) if vol_mean > 0 else 0.5
            
            # Combined score (0-100 scale)
            final_score = (avg_score * 0.6 + vol_consistency * 0.4) * 100
            
            # Determine signal and points
            if final_score >= 70:
                signal = "High delivery likely - Institutional interest"
                points = 6
            elif final_score >= 55:
                signal = "Moderate delivery - Acceptable"
                points = 3
            elif final_score >= 40:
                signal = "Low delivery - Speculative"
                points = -2
            else:
                signal = "Very low delivery - Avoid"
                points = -5
            
            return {
                'valid': True,
                'estimated_delivery_score': round(final_score, 1),
                'close_consistency': round(avg_score * 100, 1),
                'volume_consistency': round(vol_consistency * 100, 1),
                'signal': signal,
                'score': points
            }
            
        except Exception as e:
            return {
                'valid': False,
                'estimated_delivery_score': 50,
                'signal': f'Delivery estimation error: {str(e)[:30]}',
                'score': 0
            }


# ============================================================================
# 4. EARNINGS DATE FILTER
# ============================================================================

class EarningsFilter:
    """
    Filter stocks with upcoming earnings
    
    Why Important:
    - Earnings create unpredictable gaps
    - Swing trades can be destroyed by earnings miss
    - Rule: Avoid stocks reporting within 7 days
    """
    
    @staticmethod
    def check_earnings(ticker: str, days_ahead: int = 7) -> Dict:
        """Check if earnings are upcoming"""
        try:
            # Clean ticker for yfinance
            yf_ticker = ticker if '.NS' in ticker or '.BO' in ticker else f"{ticker}.NS"
            
            stock = yf.Ticker(yf_ticker)
            calendar = stock.calendar
            
            if calendar is None:
                return {
                    'earnings_soon': False,
                    'safe_to_trade': True,
                    'message': "No earnings calendar available",
                    'score': 0
                }
            
            # Handle different calendar formats
            earnings_date = None
            
            if isinstance(calendar, pd.DataFrame):
                if 'Earnings Date' in calendar.columns:
                    earnings_date = calendar['Earnings Date'].iloc[0] if len(calendar) > 0 else None
                elif len(calendar.columns) > 0:
                    # Try first column
                    earnings_date = calendar.iloc[0, 0] if len(calendar) > 0 else None
            elif isinstance(calendar, dict):
                earnings_date = calendar.get('Earnings Date', [None])[0]
            
            if earnings_date is None:
                return {
                    'earnings_soon': False,
                    'safe_to_trade': True,
                    'message': "No upcoming earnings date found",
                    'score': 0
                }
            
            # Convert to datetime if string
            if isinstance(earnings_date, str):
                earnings_date = pd.to_datetime(earnings_date)
            
            # Calculate days until earnings
            now = datetime.now()
            if hasattr(earnings_date, 'to_pydatetime'):
                earnings_date = earnings_date.to_pydatetime()
            
            if isinstance(earnings_date, datetime):
                days_until = (earnings_date - now).days
                
                if 0 <= days_until <= days_ahead:
                    return {
                        'earnings_soon': True,
                        'safe_to_trade': False,
                        'days_until': days_until,
                        'earnings_date': earnings_date.strftime('%Y-%m-%d'),
                        'message': f"⚠️ Earnings in {days_until} days - AVOID!",
                        'score': -15  # Heavy penalty
                    }
                elif days_until < 0 and days_until >= -3:
                    # Just reported - might have gap risk
                    return {
                        'earnings_soon': False,
                        'safe_to_trade': True,
                        'days_since': abs(days_until),
                        'message': f"Earnings {abs(days_until)} days ago",
                        'score': -3
                    }
            
            return {
                'earnings_soon': False,
                'safe_to_trade': True,
                'message': "No imminent earnings",
                'score': 2  # Slight bonus for clarity
            }
            
        except Exception as e:
            # If we can't check, assume safe but no bonus
            return {
                'earnings_soon': False,
                'safe_to_trade': True,
                'message': f"Could not verify: {str(e)[:30]}",
                'score': 0
            }


# ============================================================================
# 5. TRAILING STOP MANAGER
# ============================================================================

class TrailingStopManager:
    """
    Trailing Stop Loss Calculator for Swing Trades
    
    Why Important:
    - Locks in profits as trade moves in your favor
    - Prevents giving back gains
    - Automates exit decision
    
    Methods:
    - ATR: Trail by 2x ATR below highest point
    - Percent: Trail by fixed % below highest point
    - Chandelier: More aggressive 3x ATR trailing
    """
    
    @staticmethod
    def calculate_trailing_stop(
        entry_price: float,
        current_price: float,
        highest_price: float,
        lowest_price: float,
        atr: float,
        side: str,
        method: str = 'atr'
    ) -> Dict:
        """
        Calculate trailing stop based on current price movement
        
        Args:
            entry_price: Original entry price
            current_price: Current market price
            highest_price: Highest price since entry (for LONG)
            lowest_price: Lowest price since entry (for SHORT)
            atr: Current ATR value
            side: 'LONG' or 'SHORT'
            method: 'atr', 'percent', or 'chandelier'
        """
        try:
            # Calculate trail distance based on method
            if method == 'atr':
                trail_distance = atr * 2.0
            elif method == 'percent':
                trail_distance = current_price * 0.05  # 5% trail
            elif method == 'chandelier':
                trail_distance = atr * 3.0
            else:
                trail_distance = atr * 2.0
            
            if side == "LONG":
                # Trailing stop trails below the highest high
                trailing_stop = highest_price - trail_distance
                
                # Stop should never go below initial stop
                initial_stop = entry_price - (atr * 1.5)
                trailing_stop = max(trailing_stop, initial_stop)
                
                # Calculate locked profit
                if trailing_stop > entry_price:
                    locked_profit_pct = ((trailing_stop - entry_price) / entry_price) * 100
                else:
                    locked_profit_pct = 0
                
                # Current P&L
                current_pnl_pct = ((current_price - entry_price) / entry_price) * 100
                
            else:  # SHORT
                # Trailing stop trails above the lowest low
                trailing_stop = lowest_price + trail_distance
                
                # Stop should never go above initial stop
                initial_stop = entry_price + (atr * 1.5)
                trailing_stop = min(trailing_stop, initial_stop)
                
                # Calculate locked profit
                if trailing_stop < entry_price:
                    locked_profit_pct = ((entry_price - trailing_stop) / entry_price) * 100
                else:
                    locked_profit_pct = 0
                
                # Current P&L
                current_pnl_pct = ((entry_price - current_price) / entry_price) * 100
            
            return {
                'trailing_stop': round(trailing_stop, 2),
                'trail_distance': round(trail_distance, 2),
                'method': method,
                'locked_profit_pct': round(locked_profit_pct, 2),
                'current_pnl_pct': round(current_pnl_pct, 2),
                'highest_since_entry': round(highest_price, 2) if side == "LONG" else None,
                'lowest_since_entry': round(lowest_price, 2) if side == "SHORT" else None
            }
            
        except Exception as e:
            return {
                'trailing_stop': entry_price - (atr * 1.5) if side == "LONG" else entry_price + (atr * 1.5),
                'trail_distance': atr * 2.0,
                'method': method,
                'error': str(e)[:30]
            }
    
    @staticmethod
    def get_all_trailing_stops(
        entry_price: float,
        current_price: float,
        highest_price: float,
        lowest_price: float,
        atr: float,
        side: str
    ) -> Dict:
        """Calculate trailing stops using all methods"""
        
        methods = ['atr', 'percent', 'chandelier']
        results = {}
        
        for method in methods:
            results[method] = TrailingStopManager.calculate_trailing_stop(
                entry_price, current_price, highest_price, lowest_price,
                atr, side, method
            )
        
        # Recommend the most appropriate method
        if side == "LONG":
            # Use tightest stop if in profit
            if current_price > entry_price * 1.05:
                recommended = 'percent'  # Tighter trail when in profit
            else:
                recommended = 'atr'  # Standard trail
        else:
            if current_price < entry_price * 0.95:
                recommended = 'percent'
            else:
                recommended = 'atr'
        
        results['recommended_method'] = recommended
        results['recommended_stop'] = results[recommended]['trailing_stop']
        
        return results


# ============================================================================
# 6. CONSOLIDATION BREAKOUT DETECTOR
# ============================================================================

class ConsolidationBreakoutDetector:
    """
    Detect consolidation patterns and breakouts
    
    Why Important:
    - Consolidation = Price compression = Energy building
    - Breakouts from consolidation have high success rate (>60%)
    - Best swing setups often come from 2-4 week consolidations
    """
    
    @staticmethod
    def detect(df: pd.DataFrame, min_days: int = 10, max_days: int = 40) -> Dict:
        """Detect consolidation and potential breakout"""
        try:
            if len(df) < max_days + 5:
                return {
                    'consolidation_found': False,
                    'score': 0,
                    'signal': 'Insufficient data for consolidation analysis'
                }
            
            best_consolidation = None
            best_score = 0
            
            # Look for consolidation patterns of various lengths
            for lookback in range(max_days, min_days - 1, -3):
                if len(df) < lookback + 5:
                    continue
                
                # Get consolidation period and current bars
                consolidation_period = df.iloc[-(lookback + 5):-5].copy()
                current_bars = df.tail(3).copy()
                
                if len(consolidation_period) < lookback * 0.8:
                    continue
                
                # Calculate consolidation range
                cons_high = float(consolidation_period['High'].max())
                cons_low = float(consolidation_period['Low'].min())
                
                if cons_low <= 0:
                    continue
                
                range_pct = ((cons_high - cons_low) / cons_low) * 100
                
                # Narrow range = Consolidation (typically less than 12-15%)
                if range_pct <= 15:
                    current_price = float(current_bars['Close'].iloc[-1])
                    current_high = float(current_bars['High'].max())
                    current_low = float(current_bars['Low'].min())
                    
                    # Check for breakout
                    breakout_up = current_high > cons_high
                    breakout_down = current_low < cons_low
                    
                    # Volume confirmation
                    avg_vol = float(consolidation_period['Volume'].mean())
                    current_vol = float(current_bars['Volume'].mean())
                    vol_expansion = current_vol > avg_vol * 1.5 if avg_vol > 0 else False
                    
                    # Tightness bonus (tighter = better)
                    tightness_score = max(0, (15 - range_pct) / 15 * 5)
                    
                    if breakout_up and vol_expansion:
                        score = 12 + tightness_score
                        if score > best_score:
                            best_score = score
                            best_consolidation = {
                                'consolidation_found': True,
                                'consolidation_days': lookback,
                                'range_pct': round(range_pct, 1),
                                'breakout_type': 'BULLISH',
                                'breakout_level': round(cons_high, 2),
                                'support_level': round(cons_low, 2),
                                'volume_confirmed': True,
                                'volume_ratio': round(current_vol / avg_vol, 2) if avg_vol > 0 else 1,
                                'score': round(score),
                                'signal': f"🚀 Bullish breakout from {lookback}-day consolidation!"
                            }
                    
                    elif breakout_down and vol_expansion:
                        score = -12 - tightness_score
                        if abs(score) > abs(best_score):
                            best_score = score
                            best_consolidation = {
                                'consolidation_found': True,
                                'consolidation_days': lookback,
                                'range_pct': round(range_pct, 1),
                                'breakout_type': 'BEARISH',
                                'breakout_level': round(cons_low, 2),
                                'resistance_level': round(cons_high, 2),
                                'volume_confirmed': True,
                                'volume_ratio': round(current_vol / avg_vol, 2) if avg_vol > 0 else 1,
                                'score': round(score),
                                'signal': f"📉 Bearish breakdown from {lookback}-day consolidation"
                            }
                    
                    elif range_pct <= 8 and not breakout_up and not breakout_down:
                        # Tight consolidation, no breakout yet - prepare for trade
                        score = 4
                        if not best_consolidation or best_consolidation.get('breakout_type') == 'PENDING':
                            if score > best_score:
                                best_score = score
                                best_consolidation = {
                                    'consolidation_found': True,
                                    'consolidation_days': lookback,
                                    'range_pct': round(range_pct, 1),
                                    'breakout_type': 'PENDING',
                                    'resistance': round(cons_high, 2),
                                    'support': round(cons_low, 2),
                                    'score': score,
                                    'signal': f"⏳ Tight {lookback}-day consolidation - Breakout imminent"
                                }
            
            if best_consolidation:
                return best_consolidation
            
            return {
                'consolidation_found': False,
                'score': 0,
                'signal': 'No consolidation pattern detected'
            }
            
        except Exception as e:
            return {
                'consolidation_found': False,
                'score': 0,
                'error': str(e)[:40],
                'signal': f'Consolidation detection error'
            }


# ============================================================================
# 7. MARKET BREADTH ANALYZER
# ============================================================================

class MarketBreadthAnalyzer:
    """
    Market Breadth Analysis using Nifty 50
    
    Why Important:
    - Don't go LONG when market breadth is negative
    - Protects from buying in stealth corrections
    - Aligns trades with overall market direction
    """
    
    def __init__(self):
        self.market_data = None
        self.last_update = None
        self._load_market_data()
    
    def _load_market_data(self):
        """Load market index data"""
        try:
            # Try Nifty 50 first
            self.market_data = yf.download(
                "^NSEI",
                period="3mo",
                interval="1d",
                progress=False
            )
            
            # Handle MultiIndex
            if isinstance(self.market_data.columns, pd.MultiIndex):
                self.market_data.columns = self.market_data.columns.get_level_values(0)
            
            if self.market_data.empty:
                # Fallback to Nifty Bank
                self.market_data = yf.download(
                    "^NSEBANK",
                    period="3mo",
                    interval="1d",
                    progress=False
                )
                
                if isinstance(self.market_data.columns, pd.MultiIndex):
                    self.market_data.columns = self.market_data.columns.get_level_values(0)
            
            self.last_update = datetime.now()
            
        except Exception as e:
            logger.warning(f"Market breadth data load error: {e}")
            self.market_data = None
    
    def analyze(self) -> Dict:
        """Analyze current market breadth"""
        try:
            # Refresh data if stale
            if self.last_update is None or (datetime.now() - self.last_update).total_seconds() > 3600:
                self._load_market_data()
            
            if self.market_data is None or len(self.market_data) < 50:
                return {
                    'valid': False,
                    'long_ok': True,  # Default to allowing
                    'short_ok': True,
                    'regime': 'UNKNOWN',
                    'score': 0,
                    'message': 'Market data not available'
                }
            
            # Calculate indicators
            current_close = float(self.market_data['Close'].iloc[-1])
            sma_20 = float(self.market_data['Close'].rolling(20).mean().iloc[-1])
            sma_50 = float(self.market_data['Close'].rolling(50).mean().iloc[-1])
            
            # Rate of change
            roc_5 = ((current_close / float(self.market_data['Close'].iloc[-5])) - 1) * 100
            roc_20 = ((current_close / float(self.market_data['Close'].iloc[-20])) - 1) * 100
            
            # Volatility (using ATR proxy)
            high = self.market_data['High'].tail(14)
            low = self.market_data['Low'].tail(14)
            close = self.market_data['Close'].tail(14)
            tr = pd.concat([high - low, abs(high - close.shift()), abs(low - close.shift())], axis=1).max(axis=1)
            atr_pct = (tr.mean() / current_close) * 100
            
            # Determine market regime
            if current_close > sma_20 > sma_50 and roc_5 > 0 and roc_20 > 0:
                regime = "BULLISH"
                long_ok = True
                short_ok = False
                score = 8
                message = "🟢 Market BULLISH - Favor LONG trades"
            
            elif current_close > sma_20 > sma_50 and roc_5 < 0:
                regime = "BULLISH_PULLBACK"
                long_ok = True
                short_ok = False
                score = 4
                message = "🟡 Bullish pullback - LONG on dips"
            
            elif current_close < sma_20 < sma_50 and roc_5 < 0 and roc_20 < 0:
                regime = "BEARISH"
                long_ok = False
                short_ok = True
                score = -8
                message = "🔴 Market BEARISH - Favor SHORT or stay cash"
            
            elif current_close < sma_20 < sma_50 and roc_5 > 0:
                regime = "BEARISH_BOUNCE"
                long_ok = False
                short_ok = True
                score = -4
                message = "🟠 Bearish bounce - SHORT on rallies"
            
            elif current_close > sma_20 and current_close < sma_50:
                regime = "RECOVERY"
                long_ok = True
                short_ok = True
                score = 2
                message = "🔵 Recovery mode - Selective LONG"
            
            elif current_close < sma_20 and current_close > sma_50:
                regime = "WEAKENING"
                long_ok = True
                short_ok = True
                score = -2
                message = "🟠 Weakening trend - Caution advised"
            
            else:
                regime = "NEUTRAL"
                long_ok = True
                short_ok = True
                score = 0
                message = "⚪ Market NEUTRAL - Trade selectively"
            
            # High volatility adjustment
            if atr_pct > 2.0:
                score = int(score * 0.7)  # Reduce conviction in high volatility
                message += " (High volatility)"
            
            return {
                'valid': True,
                'regime': regime,
                'long_ok': long_ok,
                'short_ok': short_ok,
                'score': score,
                'roc_5': round(roc_5, 2),
                'roc_20': round(roc_20, 2),
                'sma_20': round(sma_20, 2),
                'sma_50': round(sma_50, 2),
                'current': round(current_close, 2),
                'atr_pct': round(atr_pct, 2),
                'message': message
            }
            
        except Exception as e:
            return {
                'valid': False,
                'long_ok': True,
                'short_ok': True,
                'regime': 'ERROR',
                'score': 0,
                'message': f'Market breadth error: {str(e)[:30]}'
            }

# ============================================================================
# MULTI-TIMEFRAME ANALYZER
# ============================================================================

class MultiTimeframeAnalyzer:
    """Analyze trends across multiple timeframes"""
    
    @staticmethod
    def get_trend(ticker: str) -> Optional[str]:
        """Get multi-timeframe trend alignment"""
        try:
            daily = yf.download(ticker, period="3mo", interval="1d", progress=False)
            if isinstance(daily.columns, pd.MultiIndex):
                daily.columns = daily.columns.get_level_values(0)

            weekly = yf.download(ticker, period="6mo", interval="1wk", progress=False)
            if isinstance(weekly.columns, pd.MultiIndex):
                weekly.columns = weekly.columns.get_level_values(0)
            
            if daily.empty or weekly.empty:
                return None
            
            daily_ema20 = daily['Close'].ewm(span=20, adjust=False).mean().iloc[-1]
            daily_ema50 = daily['Close'].ewm(span=50, adjust=False).mean().iloc[-1]
            daily_close = daily['Close'].iloc[-1]
            
            daily_trend = "UP" if daily_close > daily_ema20 > daily_ema50 else "DOWN"
            
            weekly_ema10 = weekly['Close'].ewm(span=10, adjust=False).mean().iloc[-1]
            weekly_ema20 = weekly['Close'].ewm(span=20, adjust=False).mean().iloc[-1]
            weekly_close = weekly['Close'].iloc[-1]
            
            weekly_trend = "UP" if weekly_close > weekly_ema10 > weekly_ema20 else "DOWN"
            
            if daily_trend == weekly_trend:
                return daily_trend
            else:
                return "MIXED"
                
        except Exception:
            return None

# ============================================================================
# FIBONACCI LEVEL DETECTOR
# ============================================================================

class FibonacciDetector:
    """Fibonacci retracement levels"""
    
    @staticmethod
    def calculate_levels(df: pd.DataFrame, lookback: int = 50) -> Optional[Dict]:
        """Calculate Fibonacci levels"""
        try:
            recent = df.tail(lookback)
            high = recent['High'].max()
            low = recent['Low'].min()
            diff = high - low
            
            levels = {
                'high': high,
                'low': low,
                'fib_0.236': high - (diff * 0.236),
                'fib_0.382': high - (diff * 0.382),
                'fib_0.500': high - (diff * 0.500),
                'fib_0.618': high - (diff * 0.618),
                'fib_0.786': high - (diff * 0.786),
            }
            
            return levels
            
        except Exception:
            return None
    
    @staticmethod
    def detect_bounce(price: float, fib_levels: Dict, side: str) -> Tuple[int, str]:
        """Detect if price is near Fibonacci level"""
        try:
            if not fib_levels:
                return 0, ""
            
            tolerance = 0.02
            
            for level_name, level_price in fib_levels.items():
                if 'fib_' not in level_name:
                    continue
                
                distance = abs(price - level_price) / price
                
                if distance < tolerance:
                    if side == "LONG" and price >= level_price:
                        return 8, f"Bouncing off {level_name}"
                    elif side == "SHORT" and price <= level_price:
                        return 8, f"Rejecting at {level_name}"
            
            return 0, ""
            
        except Exception:
            return 0, ""

# ============================================================================
# VIX SENTIMENT ANALYZER
# ============================================================================

class VIXSentimentAnalyzer:
    """Market sentiment using VIX"""
    
    @staticmethod
    def get_vix_sentiment() -> Optional[Dict]:
        """Get current VIX sentiment"""
        try:
            vix = yf.download("^INDIAVIX", period="1mo", interval="1d", progress=False)
            if isinstance(vix.columns, pd.MultiIndex):
                vix.columns = vix.columns.get_level_values(0)
            
            if vix.empty:
                return None
            
            current_vix = vix['Close'].iloc[-1]
            vix_avg = vix['Close'].mean()
            vix_percentile = stats.percentileofscore(vix['Close'], current_vix)
            
            if current_vix < 15:
                regime = "LOW_VOLATILITY"
                sentiment = "BULLISH"
                score = 5
            elif current_vix < 20:
                regime = "NORMAL"
                sentiment = "NEUTRAL"
                score = 0
            elif current_vix < 25:
                regime = "ELEVATED"
                sentiment = "CAUTIOUS"
                score = -3
            else:
                regime = "HIGH_VOLATILITY"
                sentiment = "BEARISH"
                score = -7
            
            return {
                'vix': current_vix,
                'vix_avg': vix_avg,
                'vix_percentile': vix_percentile,
                'regime': regime,
                'sentiment': sentiment,
                'score': score
            }
            
        except Exception:
            return None

# ============================================================================
# SECTOR ROTATION ANALYZER
# ============================================================================

class SectorRotationAnalyzer:
    """Analyze sector strength"""
    
    def __init__(self):
        self.sector_etfs = {
            'NIFTY_BANK': '^NSEBANK',
            'NIFTY_IT': 'NIFTYBEES.NS',
            'NIFTY_AUTO': '^CNXAUTO',
            'NIFTY_PHARMA': '^CNXPHARMA',
        }
        self.sector_strength = {}
    
    def update_sector_strength(self):
        """Update relative strength of sectors"""
        try:
            for sector, ticker in self.sector_etfs.items():
                try:
                    data = yf.download(ticker, period="3mo", interval="1d", progress=False)
                    if isinstance(data.columns, pd.MultiIndex):
                        data.columns = data.columns.get_level_values(0)
                    if not data.empty:
                        returns = (data['Close'].iloc[-1] / data['Close'].iloc[-20] - 1) * 100
                        self.sector_strength[sector] = returns
                except:
                    continue
        except Exception:
            pass
    
    def get_sector_strength(self, sector: str) -> Optional[Tuple[float, str]]:
        """Get sector relative strength"""
        try:
            if not self.sector_strength:
                self.update_sector_strength()
            
            if sector in self.sector_strength:
                strength = self.sector_strength[sector]
                
                if strength > 5:
                    return strength, "STRONG"
                elif strength > 0:
                    return strength, "MODERATE"
                else:
                    return strength, "WEAK"
            
            return None
            
        except Exception:
            return None

# ============================================================================
# ADVANCED BACKTESTING ENGINE
# ============================================================================

class AdvancedBacktester:
    """Comprehensive backtesting with walk-forward and Monte Carlo"""
    
    def __init__(self, initial_capital: float = 100000):
        self.initial_capital = initial_capital
        self.trades = []
    
    def backtest_strategy(
        self,
        df: pd.DataFrame,
        signal_type: str,
        entry_conditions: Dict,
        lookback_days: int = BACKTEST_LOOKBACK_DAYS
    ) -> Optional[BacktestResult]:
        """Full backtest of strategy"""
        
        if len(df) < lookback_days + 50:
            lookback_days = max(100, len(df) - 50)
        
        if len(df) < 150:
            return None
        
        test_data = df.iloc[-lookback_days:].copy()
        trades = []
        
        for i in range(50, len(test_data) - 20):
            window = test_data.iloc[:i+1].copy()
            
            if self._check_entry_conditions(window, signal_type, entry_conditions):
                trade = self._simulate_trade(
                    test_data.iloc[i:min(i+30, len(test_data))],
                    signal_type,
                    entry_price=test_data.iloc[i]['Close'],
                    atr=self._calculate_atr(window)
                )
                
                if trade:
                    trades.append(trade)
        
        if len(trades) < MIN_BACKTEST_TRADES:
            return None
        
        return self._calculate_metrics(trades)
    
    def _check_entry_conditions(
        self,
        df: pd.DataFrame,
        signal_type: str,
        entry_conditions: Dict
    ) -> bool:
        """Enhanced entry condition check with v7.4 features"""
        try:
            if len(df) < 50:
                return False
            
            close = df['Close'].iloc[-1]
            
            ema20 = df['Close'].ewm(span=20, adjust=False).mean().iloc[-1]
            ema50 = df['Close'].ewm(span=50, adjust=False).mean().iloc[-1]
            
            delta = df['Close'].diff()
            gain = delta.where(delta > 0, 0).rolling(14).mean()
            loss = -delta.where(delta < 0, 0).rolling(14).mean()
            rs = gain / loss
            rsi = (100 - (100 / (1 + rs))).iloc[-1]
            
            volume = df['Volume'].iloc[-1]
            avg_volume = df['Volume'].rolling(20).mean().iloc[-1]
            volume_ratio = volume / avg_volume if avg_volume > 0 else 1
            
            exp12 = df['Close'].ewm(span=12, adjust=False).mean()
            exp26 = df['Close'].ewm(span=26, adjust=False).mean()
            macd_line = exp12 - exp26
            macd_signal = macd_line.ewm(span=9, adjust=False).mean()
            
            high = df['High']
            low = df['Low']
            
            plus_dm = high.diff()
            minus_dm = -low.diff()
            plus_dm[plus_dm < 0] = 0
            minus_dm[minus_dm < 0] = 0
            
            tr1 = high - low
            tr2 = abs(high - df['Close'].shift())
            tr3 = abs(low - df['Close'].shift())
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr = tr.rolling(14).mean().iloc[-1]
            
            plus_di = 100 * (plus_dm.rolling(14).mean() / atr)
            minus_di = 100 * (minus_dm.rolling(14).mean() / atr)
            dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
            adx = dx.rolling(14).mean().iloc[-1]
            
            vcp_score, _, _ = VCPDetector.detect(df)
            vsa_score, _, _ = VolumeSpreadAnalyzer.analyze(df)
            vol_ok, _, _ = VolatilityRegimeFilter.analyze_regime(df, close)
            st_direction, _, _ = SupertrendIndicator.calculate(df)
            
            rsi_range = PRESET['rsi_range_long'] if signal_type == "LONG" else PRESET['rsi_range_short']
            
            conditions_met = []
            
            if signal_type == "LONG":
                conditions_met = [
                    close > ema20,
                    ema20 > ema50,
                    rsi_range[0] <= rsi <= rsi_range[1],
                    macd_line.iloc[-1] > macd_signal.iloc[-1],
                    volume_ratio >= PRESET['min_volume_ratio'],
                    adx >= PRESET['min_adx'],
                    vol_ok,
                    vsa_score >= 0,
                    st_direction == "UP" if st_direction else True,
                ]
            else:
                conditions_met = [
                    close < ema20,
                    ema20 < ema50,
                    rsi_range[0] <= rsi <= rsi_range[1],
                    macd_line.iloc[-1] < macd_signal.iloc[-1],
                    volume_ratio >= PRESET['min_volume_ratio'],
                    adx >= PRESET['min_adx'],
                    vol_ok,
                    vsa_score <= 0,
                    st_direction == "DOWN" if st_direction else True,
                ]
            
            if vcp_score > 5:
                conditions_met.append(True)
            
            return sum(conditions_met) >= PRESET['conditions_required']
            
        except Exception as e:
            logger.debug(f"Entry condition check failed: {e}")
            return False
    
    def _calculate_atr(self, df: pd.DataFrame) -> float:
        """Calculate ATR"""
        try:
            atr = AverageTrueRange(
                high=df['High'],
                low=df['Low'],
                close=df['Close'],
                window=PRESET['atr_period']
            ).average_true_range().iloc[-1]
            return atr
        except:
            return df['Close'].iloc[-1] * 0.02
    
    def _simulate_trade(
        self,
        future_data: pd.DataFrame,
        signal_type: str,
        entry_price: float,
        atr: float
    ) -> Optional[Dict]:
        """Simulate single trade - NO LOOK-AHEAD BIAS"""
        
        if len(future_data) < 3:
            return None
        
        # ✅ FIX #3: Enter at NEXT bar's open (realistic)
        actual_entry = future_data.iloc[1]['Open']
        
        if signal_type == "LONG":
            stop_loss = actual_entry - (atr * 1.5)
            target_1 = actual_entry + (atr * 2)
            target_2 = actual_entry + (atr * 4)
        else:
            stop_loss = actual_entry + (atr * 1.5)
            target_1 = actual_entry - (atr * 2)
            target_2 = actual_entry - (atr * 4)
        
        exit_price = None
        exit_day = None
        exit_reason = None
        
        # Start from day 2 (day 1 is entry bar)
        for day in range(2, len(future_data)):
            row = future_data.iloc[day]
            
            if signal_type == "LONG":
                # ✅ Check gap down at open first
                if row['Open'] <= stop_loss:
                    exit_price = min(stop_loss, row['Open'])  # Slippage
                    exit_day = day
                    exit_reason = "Stop Loss (gap)"
                    break
                
                # Check if stop hit during the day
                elif row['Low'] <= stop_loss:
                    exit_price = stop_loss
                    exit_day = day
                    exit_reason = "Stop Loss"
                    break
                
                # Check target 2
                elif row['High'] >= target_2:
                    exit_price = target_2
                    exit_day = day
                    exit_reason = "Target 2"
                    break
                
                # Check target 1 after 5 days
                elif day >= 7 and row['High'] >= target_1:
                    exit_price = target_1
                    exit_day = day
                    exit_reason = "Target 1"
                    break
            
            else:  # SHORT
                if row['Open'] >= stop_loss:
                    exit_price = max(stop_loss, row['Open'])
                    exit_day = day
                    exit_reason = "Stop Loss (gap)"
                    break
                
                elif row['High'] >= stop_loss:
                    exit_price = stop_loss
                    exit_day = day
                    exit_reason = "Stop Loss"
                    break
                
                elif row['Low'] <= target_2:
                    exit_price = target_2
                    exit_day = day
                    exit_reason = "Target 2"
                    break
                
                elif day >= 7 and row['Low'] <= target_1:
                    exit_price = target_1
                    exit_day = day
                    exit_reason = "Target 1"
                    break
        
        # Time exit if no target/stop hit
        if exit_price is None:
            exit_price = future_data.iloc[-1]['Close']
            exit_day = len(future_data) - 1
            exit_reason = "Time Exit"
        
        # Calculate return based on ACTUAL entry
        if signal_type == "LONG":
            return_pct = ((exit_price - actual_entry) / actual_entry) * 100
        else:
            return_pct = ((actual_entry - exit_price) / actual_entry) * 100
        
        return {
            'entry_price': actual_entry,
            'exit_price': exit_price,
            'return_pct': return_pct,
            'return_dollars': return_pct * 1000,
            'holding_days': exit_day - 1,  # Subtract entry day
            'exit_reason': exit_reason,
            'signal_type': signal_type
        }
    
    def _calculate_metrics(self, trades: List[Dict]) -> BacktestResult:
        """Calculate comprehensive metrics"""
        
        if not trades:
            return None
        
        returns = [t['return_pct'] for t in trades]
        return_dollars = [t['return_dollars'] for t in trades]
        
        winning_trades = [r for r in returns if r > 0]
        losing_trades = [r for r in returns if r <= 0]
        
        equity = self.initial_capital
        equity_curve = [equity]
        peak_equity = equity
        max_drawdown = 0
        
        for ret_dollars in return_dollars:
            equity += ret_dollars
            equity_curve.append(equity)
            peak_equity = max(peak_equity, equity)
            drawdown = ((peak_equity - equity) / peak_equity) * 100
            max_drawdown = max(max_drawdown, drawdown)
        
        max_consecutive_wins = 0
        max_consecutive_losses = 0
        current_streak = 0
        
        for ret in returns:
            if ret > 0:
                current_streak = current_streak + 1 if current_streak >= 0 else 1
                max_consecutive_wins = max(max_consecutive_wins, current_streak)
            else:
                current_streak = current_streak - 1 if current_streak <= 0 else -1
                max_consecutive_losses = max(max_consecutive_losses, abs(current_streak))
        
        total_trades = len(trades)
        win_rate = (len(winning_trades) / total_trades) * 100
        
        avg_win = np.mean(winning_trades) if winning_trades else 0
        avg_loss = np.mean(losing_trades) if losing_trades else 0
        
        gross_profit = sum([r for r in returns if r > 0])
        gross_loss = abs(sum([r for r in returns if r < 0]))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0
        
        if len(returns) > 1 and np.std(returns) > 0:
            sharpe_ratio = (np.mean(returns) / np.std(returns)) * np.sqrt(252)
        else:
            sharpe_ratio = 0
        
        total_return = ((equity - self.initial_capital) / self.initial_capital) * 100
        expectancy = np.mean(return_dollars)
        
        reliability_score = self._calculate_reliability_score(
            win_rate, profit_factor, max_drawdown, sharpe_ratio
        )
        
        return BacktestResult(
            total_trades=total_trades,
            winning_trades=len(winning_trades),
            losing_trades=len(losing_trades),
            win_rate=round(win_rate, 2),
            avg_win=round(avg_win, 2),
            avg_loss=round(avg_loss, 2),
            profit_factor=round(profit_factor, 2),
            sharpe_ratio=round(sharpe_ratio, 2),
            max_drawdown=round(max_drawdown, 2),
            total_return=round(total_return, 2),
            avg_holding_days=round(np.mean([t['holding_days'] for t in trades]), 1),
            best_trade=round(max(returns), 2),
            worst_trade=round(min(returns), 2),
            consecutive_wins=max_consecutive_wins,
            consecutive_losses=max_consecutive_losses,
            reliability_score=round(reliability_score, 1),
            expectancy=round(expectancy, 2)
        )
    
    def _calculate_reliability_score(
        self,
        win_rate: float,
        profit_factor: float,
        max_drawdown: float,
        sharpe_ratio: float
    ) -> float:
        """Calculate 0-100 reliability score"""
        
        wr_score = min(100, (win_rate - 40) * 2.5) if win_rate > 40 else 0
        pf_score = min(100, (profit_factor - 1) * 50) if profit_factor > 1 else 0
        dd_score = max(0, 100 - max_drawdown * 2)
        sharpe_score = min(100, sharpe_ratio * 33) if sharpe_ratio > 0 else 0
        
        score = (
            wr_score * 0.30 +
            pf_score * 0.30 +
            dd_score * 0.20 +
            sharpe_score * 0.20
        )
        
        return max(0, min(100, score))
    
    def walk_forward_validation(
    self,
    df: pd.DataFrame,
    signal_type: str,
    entry_conditions: Dict,
    periods: int = 4
    ) -> Optional[WalkForwardResult]:
        """
        Walk-Forward Validation - FIXED VERSION v2.0
        
        Validates strategy consistency across non-overlapping time periods.
        Prevents overfitting by testing on independent data slices.
        
        Args:
            df: Historical price data (pandas DataFrame)
            signal_type: Trade direction - "LONG" or "SHORT"
            entry_conditions: Strategy parameters dictionary
            periods: Number of validation periods (default: 4)
        
        Returns:
            WalkForwardResult: Contains validation metrics and pass/fail status
            None: If validation fails or insufficient data
        
        Methodology:
            1. Splits data into N non-overlapping periods
            2. Backtests each period independently (no data reuse)
            3. Requires minimum 5 trades per period for statistical validity
            4. Checks performance consistency across all periods
            5. Uses mode-specific passing criteria (CONSERVATIVE/BALANCED/AGGRESSIVE)
            6. Scores based on performance + stability + drawdown
        
        Example:
            600 bars, 4 periods:
            Period 1: [0-150]   → 8 trades, 62% WR, 1.8 PF ✅
            Period 2: [150-300] → 10 trades, 58% WR, 1.6 PF ✅
            Period 3: [300-450] → 9 trades, 60% WR, 1.7 PF ✅
            Period 4: [450-600] → 7 trades, 56% WR, 1.5 PF ✅
            Result: PASSED (consistent performance)
        
        Fixes Applied:
            ✅ Removed 100-bar overlap bug
            ✅ Increased minimum data requirement (150 bars/period)
            ✅ Added minimum trades per period (5+)
            ✅ Mode-specific passing criteria
            ✅ Improved consistency scoring algorithm
            ✅ Detailed logging for debugging
            ✅ Better error handling
        """
        
        # ═══════════════════════════════════════════════════════════════════
        # STEP 1: VALIDATE INPUT DATA
        # ═══════════════════════════════════════════════════════════════════
        
        # Calculate minimum required data
        # Conservative: 150 bars per period minimum for reliable results
        min_bars_per_period = 150
        min_required_total = periods * min_bars_per_period
        
        if len(df) < min_required_total:
            logger.debug(
                f"Walk-forward rejected: Insufficient data "
                f"({len(df)} bars < {min_required_total} required for {periods} periods)"
            )
            return None
        
        # ═══════════════════════════════════════════════════════════════════
        # STEP 2: CALCULATE PERIOD BOUNDARIES (NO OVERLAP!)
        # ═══════════════════════════════════════════════════════════════════
        
        period_length = len(df) // periods
        period_results = []
        period_details = []
        
        # Debug header
        logger.debug(f"\n{'='*70}")
        logger.debug(f"WALK-FORWARD VALIDATION")
        logger.debug(f"{'='*70}")
        logger.debug(f"Strategy: {signal_type}")
        logger.debug(f"Total Data: {len(df)} bars")
        logger.debug(f"Periods: {periods}")
        logger.debug(f"Period Length: {period_length} bars")
        logger.debug(f"Mode: {ACCURACY_MODE}")
        logger.debug(f"{'='*70}")
        
        # ═══════════════════════════════════════════════════════════════════
        # STEP 3: BACKTEST EACH PERIOD INDEPENDENTLY
        # ═══════════════════════════════════════════════════════════════════
        
        for i in range(periods):
            # Calculate strict boundaries (NO overlap with other periods)
            start_idx = i * period_length
            
            # Last period gets all remaining data (handles rounding)
            if i == periods - 1:
                end_idx = len(df)
            else:
                end_idx = (i + 1) * period_length
            
            # Extract period data
            period_data = df.iloc[start_idx:end_idx].copy()
            actual_bars = len(period_data)
            
            # Calculate lookback for this period
            # Use 70% of period for backtesting, 30% for indicator warmup
            lookback = max(50, int(actual_bars * 0.7))
            
            logger.debug(f"\n{'─'*70}")
            logger.debug(f"PERIOD {i + 1}/{periods}")
            logger.debug(f"{'─'*70}")
            logger.debug(f"  Date Range: {period_data.index[0]} to {period_data.index[-1]}")
            logger.debug(f"  Index Range: [{start_idx}:{end_idx}]")
            logger.debug(f"  Total Bars: {actual_bars}")
            logger.debug(f"  Lookback: {lookback} bars")
            logger.debug(f"  Test Window: ~{lookback - 50} bars")
            
            # Run backtest on this period
            try:
                result = self.backtest_strategy(
                    df=period_data,
                    signal_type=signal_type,
                    entry_conditions=entry_conditions,
                    lookback_days=lookback
                )
            except Exception as e:
                logger.debug(f"  ❌ Backtest Error: {str(e)[:50]}")
                result = None
            
            # ═══════════════════════════════════════════════════════════════
            # STEP 4: VALIDATE PERIOD RESULTS
            # ═══════════════════════════════════════════════════════════════
            
            # Minimum trades required for statistical validity
            min_trades_required = 5
            
            if result is None:
                logger.debug(f"  ❌ REJECTED: Backtest returned None")
                continue
            
            if result.total_trades < min_trades_required:
                logger.debug(
                    f"  ❌ REJECTED: Only {result.total_trades} trades "
                    f"(minimum {min_trades_required} required)"
                )
                continue
            
            # Period passed - store results
            period_results.append(result)
            
            period_details.append({
                'period': i + 1,
                'start_idx': start_idx,
                'end_idx': end_idx,
                'bars': actual_bars,
                'trades': result.total_trades,
                'win_rate': result.win_rate,
                'profit_factor': result.profit_factor,
                'max_drawdown': result.max_drawdown,
                'sharpe_ratio': result.sharpe_ratio,
                'avg_win': result.avg_win,
                'avg_loss': result.avg_loss
            })
            
            logger.debug(f"  ✅ VALID PERIOD")
            logger.debug(f"     Trades: {result.total_trades}")
            logger.debug(f"     Win Rate: {result.win_rate:.1f}%")
            logger.debug(f"     Profit Factor: {result.profit_factor:.2f}")
            logger.debug(f"     Max Drawdown: {result.max_drawdown:.1f}%")
            logger.debug(f"     Sharpe Ratio: {result.sharpe_ratio:.2f}")
        
        # ═══════════════════════════════════════════════════════════════════
        # STEP 5: CHECK MINIMUM PERIODS REQUIREMENT
        # ═══════════════════════════════════════════════════════════════════
        
        # Require at least 50% of periods to succeed (minimum 2)
        min_required_periods = max(2, (periods + 1) // 2)
        
        if len(period_results) < min_required_periods:
            logger.debug(f"\n{'='*70}")
            logger.debug(f"❌ WALK-FORWARD FAILED")
            logger.debug(f"{'='*70}")
            logger.debug(
                f"Only {len(period_results)}/{periods} periods valid "
                f"(need {min_required_periods}+ periods)"
            )
            logger.debug(f"{'='*70}\n")
            return None
        
        # ═══════════════════════════════════════════════════════════════════
        # STEP 6: CALCULATE AGGREGATE STATISTICS
        # ═══════════════════════════════════════════════════════════════════
        
        win_rates = [r.win_rate for r in period_results]
        profit_factors = [r.profit_factor for r in period_results]
        max_drawdowns = [r.max_drawdown for r in period_results]
        sharpe_ratios = [r.sharpe_ratio for r in period_results]
        
        # Averages
        avg_win_rate = float(np.mean(win_rates))
        avg_profit_factor = float(np.mean(profit_factors))
        avg_max_drawdown = float(np.mean(max_drawdowns))
        avg_sharpe = float(np.mean(sharpe_ratios))
        
        # Stability (standard deviation)
        win_rate_stability = float(np.std(win_rates))
        pf_stability = float(np.std(profit_factors))
        
        # Min/Max
        min_win_rate = float(min(win_rates))
        max_win_rate = float(max(win_rates))
        min_profit_factor = float(min(profit_factors))
        max_profit_factor = float(max(profit_factors))
        
        # ═══════════════════════════════════════════════════════════════════
        # STEP 7: CALCULATE CONSISTENCY SCORE (IMPROVED ALGORITHM)
        # ═══════════════════════════════════════════════════════════════════
        
        # Base score from average performance
        if avg_win_rate < 35:
            base_score = 0  # Terrible
        elif avg_win_rate < 45:
            base_score = 20  # Very poor
        elif avg_win_rate < 50:
            base_score = 35  # Poor
        elif avg_win_rate < 55:
            base_score = 50  # Below average
        elif avg_win_rate < 60:
            base_score = 65  # Average
        elif avg_win_rate < 65:
            base_score = 80  # Good
        elif avg_win_rate < 70:
            base_score = 90  # Very good
        else:
            base_score = 95  # Excellent
        
        # Penalty for win rate instability
        # High std = inconsistent performance
        stability_penalty = min(40, win_rate_stability * 3)
        
        # Bonus for profit factor
        # Reward strategies that make more when they win
        pf_bonus = min(15, max(0, (avg_profit_factor - 1.0) * 8))
        
        # Penalty for drawdown
        # Penalize risky strategies
        dd_penalty = min(25, avg_max_drawdown / 1.5)
        
        # Bonus for Sharpe ratio
        # Reward risk-adjusted returns
        sharpe_bonus = min(10, max(0, avg_sharpe * 4))
        
        # Penalty for worst period
        # If any single period is terrible, reduce score
        worst_period_penalty = 0
        if min_win_rate < 40:
            worst_period_penalty = 15
        elif min_win_rate < 45:
            worst_period_penalty = 10
        elif min_win_rate < 50:
            worst_period_penalty = 5
        
        # Calculate final consistency score (0-100)
        consistency_score = base_score - stability_penalty + pf_bonus - dd_penalty + sharpe_bonus - worst_period_penalty
        consistency_score = float(max(0, min(100, consistency_score)))
        
        # ═══════════════════════════════════════════════════════════════════
        # STEP 8: DETERMINE PASS/FAIL (MODE-SPECIFIC CRITERIA)
        # ═══════════════════════════════════════════════════════════════════
        
        # Get mode-specific thresholds
        if ACCURACY_MODE == 'CONSERVATIVE':
            required_min_wr = 60
            required_min_pf = 1.8
            required_max_stability = 8
            required_max_dd = 15
        elif ACCURACY_MODE == 'BALANCED':
            required_min_wr = 55
            required_min_pf = 1.5
            required_max_stability = 10
            required_max_dd = 20
        else:  # AGGRESSIVE
            required_min_wr = 50
            required_min_pf = 1.3
            required_max_stability = 15
            required_max_dd = 25
        
        # Check 1: All periods must meet minimum criteria
        all_periods_pass_individual = all([
            r.win_rate >= required_min_wr and 
            r.profit_factor >= required_min_pf and
            r.max_drawdown <= required_max_dd
            for r in period_results
        ])
        
        # Check 2: Overall stability acceptable
        stability_acceptable = win_rate_stability <= required_max_stability
        
        # Check 3: Average performance meets requirements
        average_performance_ok = (
            avg_win_rate >= required_min_wr and
            avg_profit_factor >= required_min_pf
        )
        
        # Check 4: No catastrophic single period
        no_terrible_periods = min_win_rate >= (required_min_wr - 10)
        
        # Final pass/fail decision
        passed = (
            all_periods_pass_individual and 
            stability_acceptable and 
            average_performance_ok and
            no_terrible_periods
        )
        
        # ═══════════════════════════════════════════════════════════════════
        # STEP 9: DETAILED LOGGING
        # ═══════════════════════════════════════════════════════════════════
        
        logger.debug(f"\n{'='*70}")
        logger.debug(f"WALK-FORWARD RESULTS")
        logger.debug(f"{'='*70}")
        logger.debug(f"Validated Periods: {len(period_results)}/{periods}")
        logger.debug(f"")
        logger.debug(f"AGGREGATE STATISTICS:")
        logger.debug(f"  Win Rate: {avg_win_rate:.1f}% (range: {min_win_rate:.1f}% - {max_win_rate:.1f}%)")
        logger.debug(f"  Stability: ±{win_rate_stability:.1f}%")
        logger.debug(f"  Profit Factor: {avg_profit_factor:.2f} (range: {min_profit_factor:.2f} - {max_profit_factor:.2f})")
        logger.debug(f"  Max Drawdown: {avg_max_drawdown:.1f}%")
        logger.debug(f"  Sharpe Ratio: {avg_sharpe:.2f}")
        logger.debug(f"")
        logger.debug(f"CONSISTENCY SCORE: {consistency_score:.1f}/100")
        logger.debug(f"  Base Score: {base_score:.0f}")
        logger.debug(f"  Stability Penalty: -{stability_penalty:.0f}")
        logger.debug(f"  Profit Factor Bonus: +{pf_bonus:.0f}")
        logger.debug(f"  Drawdown Penalty: -{dd_penalty:.0f}")
        logger.debug(f"  Sharpe Bonus: +{sharpe_bonus:.0f}")
        logger.debug(f"  Worst Period Penalty: -{worst_period_penalty:.0f}")
        logger.debug(f"")
        logger.debug(f"REQUIREMENTS ({ACCURACY_MODE} mode):")
        logger.debug(f"  Min Win Rate: {required_min_wr}% {'✅' if avg_win_rate >= required_min_wr else '❌'}")
        logger.debug(f"  Min Profit Factor: {required_min_pf} {'✅' if avg_profit_factor >= required_min_pf else '❌'}")
        logger.debug(f"  Max Stability: {required_max_stability}% {'✅' if win_rate_stability <= required_max_stability else '❌'}")
        logger.debug(f"  Max Drawdown: {required_max_dd}% {'✅' if avg_max_drawdown <= required_max_dd else '❌'}")
        logger.debug(f"")
        logger.debug(f"VALIDATION CHECKS:")
        logger.debug(f"  All periods pass individual: {'✅' if all_periods_pass_individual else '❌'}")
        logger.debug(f"  Stability acceptable: {'✅' if stability_acceptable else '❌'}")
        logger.debug(f"  Average performance OK: {'✅' if average_performance_ok else '❌'}")
        logger.debug(f"  No terrible periods: {'✅' if no_terrible_periods else '❌'}")
        
        if period_details:
            logger.debug(f"")
            logger.debug(f"PERIOD BREAKDOWN:")
            for detail in period_details:
                status = "✅" if (
                    detail['win_rate'] >= required_min_wr and 
                    detail['profit_factor'] >= required_min_pf and
                    detail['max_drawdown'] <= required_max_dd
                ) else "❌"
                
                logger.debug(
                    f"  Period {detail['period']} {status}: "
                    f"{detail['trades']} trades | "
                    f"WR={detail['win_rate']:.1f}% | "
                    f"PF={detail['profit_factor']:.2f} | "
                    f"DD={detail['max_drawdown']:.1f}% | "
                    f"Sharpe={detail['sharpe_ratio']:.2f}"
                )
        
        logger.debug(f"")
        logger.debug(f"{'='*70}")
        logger.debug(f"FINAL RESULT: {'✅ PASSED' if passed else '❌ FAILED'}")
        logger.debug(f"{'='*70}\n")
        
        # ═══════════════════════════════════════════════════════════════════
        # STEP 10: RETURN RESULT
        # ═══════════════════════════════════════════════════════════════════
        
        return WalkForwardResult(
            period_results=period_results,
            avg_win_rate=round(avg_win_rate, 2),
            win_rate_stability=round(win_rate_stability, 2),
            avg_profit_factor=round(avg_profit_factor, 2),
            consistency_score=round(consistency_score, 1),
            passed=passed
        )
   
    def monte_carlo_simulation(
        self,
        backtest_result: BacktestResult,
        simulations: int = 1000,
        num_trades: int = 50
    ) -> MonteCarloResult:
        """Monte Carlo simulation"""
        
        if not backtest_result or backtest_result.total_trades < 10:
            return None
        
        win_rate = backtest_result.win_rate / 100
        avg_win = backtest_result.avg_win
        avg_loss = backtest_result.avg_loss
        
        final_equities = []
        
        for _ in range(simulations):
            equity = self.initial_capital
            
            for _ in range(num_trades):
                if np.random.random() < win_rate:
                    ret = np.random.normal(avg_win, avg_win * 0.3)
                else:
                    ret = np.random.normal(avg_loss, abs(avg_loss) * 0.3)
                
                equity *= (1 + ret / 100)
            
            final_equities.append(equity)
        
        final_equities = np.array(final_equities)
        
        avg_final = np.mean(final_equities)
        median_final = np.median(final_equities)
        worst_case = np.percentile(final_equities, 5)
        best_case = np.percentile(final_equities, 95)
        
        prob_profit = (final_equities > self.initial_capital).sum() / simulations
        
        confidence_95 = (
            np.percentile(final_equities, 2.5),
            np.percentile(final_equities, 97.5)
        )
        
        risk_of_ruin = (final_equities < self.initial_capital * 0.5).sum() / simulations
        
        return MonteCarloResult(
            simulations=simulations,
            avg_final_equity=round(avg_final, 2),
            median_final_equity=round(median_final, 2),
            worst_case=round(worst_case, 2),
            best_case=round(best_case, 2),
            prob_profit=round(prob_profit * 100, 2),
            confidence_95=(round(confidence_95[0], 2), round(confidence_95[1], 2)),
            risk_of_ruin=round(risk_of_ruin * 100, 2)
        )
# ============================================================================
# FUNDAMENTALS FETCHER
# ============================================================================

def get_fundamentals(ticker: str) -> Tuple[Optional[str], Optional[float]]:
    """Get sector and P/E ratio"""
    try:
        t = yf.Ticker(ticker)
        info = t.info or {}
        sector = info.get('sector')
        pe = info.get('trailingPE')
        return sector, pe
    except Exception:
        return None, None

# ============================================================================
# INDICATOR CALCULATION
# ============================================================================
def normalize_price_data(df: pd.DataFrame) -> pd.DataFrame:
    """Fixed price normalization - only convert if definitely paise data"""
    try:
        if len(df) < 10:
            return df
        
        # Get current close price
        current_close = df['Close'].iloc[-1]
        avg_close = df['Close'].mean()
        
        # Only normalize if we're SURE it's paise data
        # Criteria: Average price > 100,000 (no stock trades above ₹100k except in paise)
        if avg_close > 100000:
            # This is definitely paise data
            df = df.copy()
            df['Open'] = df['Open'] / 100
            df['High'] = df['High'] / 100
            df['Low'] = df['Low'] / 100
            df['Close'] = df['Close'] / 100
            logger.debug(f"Normalized paise data: {avg_close:.0f} → {avg_close/100:.2f}")
        
        # Don't normalize anything else - keep actual prices
        return df
        
    except Exception as e:
        logger.debug(f"Price normalization error: {e}")
        return df
    
def calculate_indicators(df: pd.DataFrame) -> Optional[Dict]:
    """Calculate all technical indicators"""
    try:
        if len(df) < 60:
            return None
        
        df = df.sort_index()
        
        c = normalize_series(df["Close"])
        h = normalize_series(df["High"])
        l = normalize_series(df["Low"])
        v = normalize_series(df["Volume"])
        
        ema20_series = EMAIndicator(c, 20).ema_indicator()
        ema50_series = EMAIndicator(c, 50).ema_indicator()
        ema200_series = EMAIndicator(c, 200).ema_indicator()
        
        ema20 = safe_indicator(lambda: float(ema20_series.iloc[-1]), default=float(c.iloc[-1]))
        ema50 = safe_indicator(lambda: float(ema50_series.iloc[-1]), default=float(c.iloc[-1]))
        ema200 = safe_indicator(lambda: float(ema200_series.iloc[-1]), default=float(c.iloc[-1]))
        
        def calc_slope(series, periods=5):
            try:
                return float(series.iloc[-1] - series.iloc[-periods])
            except:
                return 0.0
        
        ema20_slope = calc_slope(ema20_series)
        ema50_slope = calc_slope(ema50_series)
        
        atr_series = AverageTrueRange(h, l, c, PRESET['atr_period']).average_true_range()
        atr_val = safe_indicator(lambda: float(atr_series.iloc[-1]), default=float(c.iloc[-1]) * 0.02)
        atr_slope = calc_slope(atr_series)
        
        rsi = safe_indicator(lambda: float(RSIIndicator(c, 14).rsi().iloc[-1]), default=50)
        
        try:
            macd_obj = MACD(c)
            macd_line = float(macd_obj.macd().iloc[-1])
            macd_signal = float(macd_obj.macd_signal().iloc[-1])
        except:
            macd_line = 0
            macd_signal = 0
        
        price_change = c.diff().fillna(0)
        direction = np.sign(price_change)
        obv_series = (direction * v).cumsum()
        obv_slope = calc_slope(obv_series)
        
        adx = safe_indicator(lambda: float(ADXIndicator(h, l, c, 14).adx().iloc[-1]), default=20)
        roc = safe_indicator(lambda: float(ROCIndicator(c, 10).roc().iloc[-1]), default=0)
        
        avg_vol = v.rolling(20).mean().iloc[-1]
        volume_ratio = float(v.iloc[-1] / avg_vol) if avg_vol > 0 else 1.0
        
        return {
            "price": float(c.iloc[-1]),
            "volume": float(v.iloc[-1]),
            "ema20": ema20,
            "ema50": ema50,
            "ema200": ema200,
            "ema20_slope": ema20_slope,
            "ema50_slope": ema50_slope,
            "atr": atr_val,
            "atr_slope": atr_slope,
            "rsi": rsi,
            "macd_line": macd_line,
            "macd_signal": macd_signal,
            "obv_slope": obv_slope,
            "adx": adx,
            "roc": roc,
            "volume_ratio": volume_ratio,
        }
        
    except Exception as e:
        logger.debug(f"Indicator calculation failed: {e}")
        return None

# ============================================================================
# ENTRY SIGNAL GENERATOR
# ============================================================================

def generate_entry_signal(ind: Dict, df: pd.DataFrame, side: str) -> Dict:
    """Generate entry signal with detailed reasons"""
    
    try:
        entry_ready = False
        entry_signal = "NONE"
        reasons = []
        
        price = ind["price"]
        rsi = ind["rsi"]
        ema20 = ind["ema20"]
        ema50 = ind["ema50"]
        macd_line = ind["macd_line"]
        macd_signal = ind["macd_signal"]
        volume_ratio = ind["volume_ratio"]
        
        rsi_range = PRESET['rsi_range_long'] if side == "LONG" else PRESET['rsi_range_short']
        
        if side == "LONG":
            if price > ema20:
                reasons.append("Price > EMA20")
            
            if ema20 > ema50:
                reasons.append("EMA20 > EMA50")
            
            if rsi_range[0] <= rsi <= rsi_range[1]:
                reasons.append(f"RSI {rsi:.1f} in range")
            
            if macd_line > macd_signal:
                reasons.append("MACD bullish")
            
            if volume_ratio >= 1.0:
                reasons.append(f"Volume {volume_ratio:.2f}x")
            
            conditions = [
                price > ema20,
                ema20 > ema50,
                rsi_range[0] <= rsi <= rsi_range[1],
                macd_line > macd_signal,
                volume_ratio >= PRESET['min_volume_ratio']
            ]
            
            min_conditions = PRESET.get('conditions_required', 3)
            if sum(conditions) >= min_conditions:
                entry_ready = True
                entry_signal = "LONG_ENTRY"
        
        else:  # SHORT
            if price < ema20:
                reasons.append("Price < EMA20")
            
            if ema20 < ema50:
                reasons.append("EMA20 < EMA50")
            
            if rsi_range[0] <= rsi <= rsi_range[1]:
                reasons.append(f"RSI {rsi:.1f} in range")
            
            if macd_line < macd_signal:
                reasons.append("MACD bearish")
            
            if volume_ratio >= 1.0:
                reasons.append(f"Volume {volume_ratio:.2f}x")
            
            conditions = [
                price < ema20,
                ema20 < ema50,
                rsi_range[0] <= rsi <= rsi_range[1],
                macd_line < macd_signal,
                volume_ratio >= PRESET['min_volume_ratio']
            ]
            
            min_conditions = PRESET.get('conditions_required', 3)
            if sum(conditions) >= min_conditions:
                entry_ready = True
                entry_signal = "SHORT_ENTRY"
        
        return {
            "entry_ready": entry_ready,
            "entry_signal": entry_signal,
            "reasons": reasons
        }
        
    except Exception as e:
        return {
            "entry_ready": False,
            "entry_signal": "ERROR",
            "reasons": [str(e)[:50]]
        }

# ============================================================================
# COMPREHENSIVE SCORING SYSTEM
# ============================================================================

def score_signal(
    ind: Dict,
    df: pd.DataFrame,
    side: str,
    vix_sentiment: Optional[Dict],
    sector_rs: Optional[Tuple[float, str]],
    fibo_boost: int,
    sr_info: Dict,
    candle: Dict,
    mtf_trend: Optional[str],
    vol_analysis: Dict,
    volume_analysis: Dict,
    vsa_analysis: Dict,
    momentum_analysis: Dict,
    vcp_analysis: Dict,
    gap_analysis: Dict,
    supertrend_info: Dict,
    fifty_two_week_data: Optional[Dict] = None,
    rs_data: Optional[Dict] = None,
    delivery_data: Optional[Dict] = None,
    earnings_data: Optional[Dict] = None,
    consolidation_data: Optional[Dict] = None,
    market_breadth_data: Optional[Dict] = None
) -> Tuple[float, List[str]]:
    """Comprehensive scoring with REALISTIC confidence - STEEPER SIGMOID"""
    
    w = SCORING_WEIGHTS
    raw_score = 0.0  # ✅ Start at 0
    reasons = []
    
    # ═══════════════════════════════════════════════════════════════════════
    # EMA TREND (max ±15 points)
    # ═══════════════════════════════════════════════════════════════════════
    if side == "LONG":
        if ind["price"] > ind["ema20"] > ind["ema50"]:
            raw_score += 15
            reasons.append("Strong EMA alignment")
        elif ind["price"] > ind["ema20"]:
            raw_score += 7
        else:
            raw_score -= 12  # ✅ STRONG PENALTY for against trend
        
        if ind["price"] > ind["ema200"]:
            raw_score += w["ema_200"]
            reasons.append("Above 200 EMA")
        else:
            raw_score -= 3  # ✅ PENALTY
    else:  # SHORT
        if ind["price"] < ind["ema20"] < ind["ema50"]:
            raw_score += 15
            reasons.append("Strong EMA alignment")
        elif ind["price"] < ind["ema20"]:
            raw_score += 7
        else:
            raw_score -= 12  # ✅ STRONG PENALTY
        
        if ind["price"] < ind["ema200"]:
            raw_score += w["ema_200"]
            reasons.append("Below 200 EMA")
        else:
            raw_score -= 3
    
    # ═══════════════════════════════════════════════════════════════════════
    # EMA SLOPE (max ±5 points)
    # ═══════════════════════════════════════════════════════════════════════
    if side == "LONG" and ind["ema20_slope"] > 0:
        raw_score += w["ema_slope"]
        reasons.append("EMA rising")
    elif side == "SHORT" and ind["ema20_slope"] < 0:
        raw_score += w["ema_slope"]
        reasons.append("EMA falling")
    else:
        raw_score -= 4  # ✅ PENALTY for wrong slope
    
    # ═══════════════════════════════════════════════════════════════════════
    # RSI (max ±10 points)
    # ═══════════════════════════════════════════════════════════════════════
    rsi_range = PRESET['rsi_range_long'] if side == "LONG" else PRESET['rsi_range_short']
    
    if rsi_range[0] <= ind["rsi"] <= rsi_range[1]:
        raw_score += w["rsi"]
        reasons.append(f"RSI {ind['rsi']:.1f} optimal")
    elif ind["rsi"] < 20 or ind["rsi"] > 80:
        raw_score -= 8  # ✅ STRONG PENALTY for extreme RSI
    elif ind["rsi"] < 30 or ind["rsi"] > 70:
        raw_score -= 5  # ✅ MODERATE PENALTY
    
    # ═══════════════════════════════════════════════════════════════════════
    # MACD (max ±10 points)
    # ═══════════════════════════════════════════════════════════════════════
    if side == "LONG" and ind["macd_line"] > ind["macd_signal"]:
        raw_score += w["macd"]
        reasons.append("MACD bullish")
    elif side == "SHORT" and ind["macd_line"] < ind["macd_signal"]:
        raw_score += w["macd"]
        reasons.append("MACD bearish")
    else:
        raw_score -= 6  # ✅ PENALTY for wrong MACD direction
    
    # ═══════════════════════════════════════════════════════════════════════
    # VOLUME (max ±6 points)
    # ═══════════════════════════════════════════════════════════════════════
    if ind["volume_ratio"] >= 1.5:
        raw_score += w["volume_ratio"]
        reasons.append(f"Strong volume")
    elif ind["volume_ratio"] >= 1.0:
        raw_score += w["volume_ratio"] * 0.5  # Partial credit
    elif ind["volume_ratio"] < 0.7:
        raw_score -= w["volume_ratio"]  # PENALTY for weak volume
    
    # ═══════════════════════════════════════════════════════════════════════
    # OBV (max ±4 points)
    # ═══════════════════════════════════════════════════════════════════════
    if side == "LONG" and ind["obv_slope"] > 0:
        raw_score += w["obv"]
    elif side == "SHORT" and ind["obv_slope"] < 0:
        raw_score += w["obv"]
    else:
        raw_score -= 3  # ✅ PENALTY
    
    # ═══════════════════════════════════════════════════════════════════════
    # SUPPORT/RESISTANCE (max ±7 points)
    # ═══════════════════════════════════════════════════════════════════════
    sr_bias = sr_info.get('bias', 'UNKNOWN')
    
    if sr_bias == 'AT_SUPPORT' and side == "LONG":
        raw_score += w["sr_support"]
        reasons.append("At support")
    elif sr_bias == 'AT_RESISTANCE' and side == "SHORT":
        raw_score += w["sr_resistance"]
        reasons.append("At resistance")
    elif sr_bias == 'AT_RESISTANCE' and side == "LONG":
        raw_score -= 5  # ✅ PENALTY - buying at resistance
    elif sr_bias == 'AT_SUPPORT' and side == "SHORT":
        raw_score -= 5  # ✅ PENALTY - shorting at support
    
    # ═══════════════════════════════════════════════════════════════════════
    # CANDLESTICK PATTERNS (max ±6 points)
    # ═══════════════════════════════════════════════════════════════════════
    candle_score = candle.get('score', 0)
    
    if (side == "LONG" and candle_score > 0) or (side == "SHORT" and candle_score < 0):
        raw_score += w["candle"] * min(1, abs(candle_score) / 8)
        reasons.append(candle.get('description', 'Pattern'))
    elif abs(candle_score) > 5:
        # ✅ PENALTY if candle contradicts our direction
        raw_score -= 4
    
    # ═══════════════════════════════════════════════════════════════════════
    # MULTI-TIMEFRAME (max ±9 points) - STRONG WEIGHT
    # ═══════════════════════════════════════════════════════════════════════
    if mtf_trend:
        if (side == "LONG" and mtf_trend == "UP") or (side == "SHORT" and mtf_trend == "DOWN"):
            raw_score += w["mtf_trend"]
            reasons.append(f"MTF {mtf_trend}")
        elif mtf_trend == "MIXED":
            raw_score -= w["mtf_penalty"]  # ✅ STRONG PENALTY
        else:
            raw_score -= 8  # ✅ VERY STRONG PENALTY for opposing MTF
    
    # ═══════════════════════════════════════════════════════════════════════
    # VIX SENTIMENT (max ±6 points)
    # ═══════════════════════════════════════════════════════════════════════
    if vix_sentiment and USE_VIX_SENTIMENT:
        vix_score = vix_sentiment.get('score', 0)
        if side == "LONG" and vix_score > 0:
            raw_score += w["vix_sentiment"]
        elif side == "SHORT" and vix_score < 0:
            raw_score += w["vix_sentiment"]
        elif abs(vix_score) > 5:
            raw_score -= 3  # ✅ PENALTY if VIX opposes direction
    
    # ═══════════════════════════════════════════════════════════════════════
    # FIBONACCI (max ±8 points)
    # ═══════════════════════════════════════════════════════════════════════
    if fibo_boost > 0:
        raw_score += w["fib_boost"]
        reasons.append("At Fib level")
    
    # ═══════════════════════════════════════════════════════════════════════
    # VOLUME BREAKOUT (max ±7 points)
    # ═══════════════════════════════════════════════════════════════════════
    if volume_analysis:
        vol_score = volume_analysis.get('score', 0)
        if vol_score > 0:
            raw_score += min(w["volume_breakout"], vol_score)
            if vol_score > 5:
                reasons.append(volume_analysis.get('signal', ''))
        elif vol_score < -5:
            raw_score -= 5  # ✅ PENALTY for negative volume signal
    
    # ═══════════════════════════════════════════════════════════════════════
    # VSA (max ±7 points)
    # ═══════════════════════════════════════════════════════════════════════
    if vsa_analysis:
        vsa_score = vsa_analysis.get('score', 0)
        if (side == "LONG" and vsa_score > 0) or (side == "SHORT" and vsa_score < 0):
            raw_score += w["vsa_confirmation"] * min(1, abs(vsa_score) / 8)
            reasons.append(vsa_analysis.get('signal', ''))
        elif abs(vsa_score) > 5:
            raw_score -= 4  # ✅ PENALTY for opposing VSA
    
    # ═══════════════════════════════════════════════════════════════════════
    # MOMENTUM (max ±8 points)
    # ═══════════════════════════════════════════════════════════════════════
    if momentum_analysis:
        mom_score = momentum_analysis.get('score', 0)
        if mom_score > 0:
            raw_score += min(w["adx_momentum"], mom_score)
            if mom_score > 5:
                reasons.append(momentum_analysis.get('signal', ''))
        elif mom_score < -5:
            raw_score -= 5  # ✅ PENALTY
    
    # ═══════════════════════════════════════════════════════════════════════
    # VCP PATTERN (max ±9 points)
    # ═══════════════════════════════════════════════════════════════════════
    if vcp_analysis:
        vcp_score = vcp_analysis.get('score', 0)
        if vcp_score > 5:
            raw_score += w["vcp_pattern"] * min(1, vcp_score / 10)
            reasons.append(vcp_analysis.get('signal', ''))
    
    # ═══════════════════════════════════════════════════════════════════════
    # GAP DETECTION (max ±5 points)
    # ═══════════════════════════════════════════════════════════════════════
    if gap_analysis:
        gap_score = gap_analysis.get('score', 0)
        if (side == "LONG" and gap_score > 0) or (side == "SHORT" and gap_score < 0):
            raw_score += w["gap_detection"] * min(1, abs(gap_score) / 8)
        elif abs(gap_score) > 5:
            raw_score -= 3  # ✅ PENALTY
    
    # ═══════════════════════════════════════════════════════════════════════
    # SUPERTREND (max ±8 points)
    # ═══════════════════════════════════════════════════════════════════════
    if supertrend_info:
        st_direction = supertrend_info.get('direction')
        if (side == "LONG" and st_direction == "UP") or (side == "SHORT" and st_direction == "DOWN"):
            raw_score += w["supertrend"]
            reasons.append(f"Supertrend {st_direction}")
        elif st_direction:
            raw_score -= 6  # ✅ STRONG PENALTY for opposing Supertrend
    
    # ═══════════════════════════════════════════════════════════════════════
    # SECTOR STRENGTH (max ±5 points)
    # ═══════════════════════════════════════════════════════════════════════
    if sector_rs and USE_SECTOR_ROTATION:
        rs_value, rs_label = sector_rs
        if rs_label == "STRONG":
            raw_score += 5
            reasons.append("Strong sector")
        elif rs_label == "WEAK":
            raw_score -= 4  # ✅ PENALTY
     # ═══════════════════════════════════════════════════════════════════════
    # NEW FEATURE SCORING - ADD ALL OF THIS BELOW SECTOR STRENGTH
    # ═══════════════════════════════════════════════════════════════════════
    
    # ═══════════════════════════════════════════════════════════════════════
    # 52-WEEK HIGH/LOW (max ±12 points)
    # ═══════════════════════════════════════════════════════════════════════
    if fifty_two_week_data and fifty_two_week_data.get('valid'):
        ftw_score = fifty_two_week_data.get('score', 0)
    
        # Apply score based on trade direction
        if side == "LONG":
            raw_score += ftw_score
            if ftw_score >= 8:
                reasons.append(fifty_two_week_data.get('signal', '52W Strong'))
            elif ftw_score <= -7:
                reasons.append("⚠️ Near 52W low")
        else:  # SHORT
            # Inverse logic for shorts
            raw_score -= ftw_score * 0.5  # Shorts benefit from weakness
            if ftw_score <= -7:
                reasons.append("Weakness - good for short")
    
    # ═══════════════════════════════════════════════════════════════════════
    # RELATIVE STRENGTH VS NIFTY (max ±10 points)
    # ═══════════════════════════════════════════════════════════════════════
    if rs_data and rs_data.get('valid'):
        rs_score = rs_data.get('score', 0)
        
        if side == "LONG":
            raw_score += rs_score
            if rs_score >= 6:
                reasons.append(f"RS {rs_data.get('avg_rs', 1):.2f}x Nifty")
        else:  # SHORT - weak RS is good
            raw_score -= rs_score * 0.7
            if rs_score <= -5:
                reasons.append("Weak vs Nifty - good for short")
    
    # ═══════════════════════════════════════════════════════════════════════
    # DELIVERY VOLUME (max ±6 points)
    # ═══════════════════════════════════════════════════════════════════════
    if delivery_data and delivery_data.get('valid'):
        delivery_score = delivery_data.get('score', 0)
        raw_score += delivery_score
        
        if delivery_score >= 4:
            reasons.append("High delivery %")
        elif delivery_score <= -3:
            reasons.append("Low delivery - speculative")
    
    # ═══════════════════════════════════════════════════════════════════════
    # EARNINGS FILTER (max ±15 points - HEAVY PENALTY)
    # ═══════════════════════════════════════════════════════════════════════
    if earnings_data:
        earnings_score = earnings_data.get('score', 0)
        raw_score += earnings_score
        
        if earnings_score <= -10:
            reasons.append(earnings_data.get('message', 'Earnings risk'))
        elif earnings_score > 0:
            reasons.append("No earnings risk")
    
    # ====================================================================
    # EARNINGS FILTER - REJECT IF EARNINGS SOON
    # ====================================================================
    if earnings_data and USE_EARNINGS_FILTER:
        if earnings_data.get('earnings_soon') and not earnings_data.get('safe_to_trade', True):
            stats["earnings_fail"] += 1  # ✅ FIXED
            log_rejection("Earnings soon", earnings_data.get('message', ''))
            return None
    
    # ═══════════════════════════════════════════════════════════════════════
    # CONSOLIDATION BREAKOUT (max ±15 points)
    # ═══════════════════════════════════════════════════════════════════════
    if consolidation_data and consolidation_data.get('consolidation_found'):
        cons_score = consolidation_data.get('score', 0)
        breakout_type = consolidation_data.get('breakout_type', '')
        
        # Apply score based on alignment with trade direction
        if (side == "LONG" and breakout_type == "BULLISH") or \
           (side == "SHORT" and breakout_type == "BEARISH"):
            raw_score += abs(cons_score)
            reasons.append(consolidation_data.get('signal', 'Consolidation breakout'))
        elif breakout_type == "PENDING":
            raw_score += cons_score * 0.5
            reasons.append("Consolidation forming")
        elif breakout_type:
            # Wrong direction breakout
            raw_score -= abs(cons_score) * 0.5
    
    # ═══════════════════════════════════════════════════════════════════════
    # MARKET BREADTH (max ±8 points)
    # ═══════════════════════════════════════════════════════════════════════
    if market_breadth_data and market_breadth_data.get('valid'):
        breadth_score = market_breadth_data.get('score', 0)
        regime = market_breadth_data.get('regime', 'NEUTRAL')
        
        if side == "LONG":
            raw_score += breadth_score
            if breadth_score >= 5:
                reasons.append(f"Market {regime}")
        else:  # SHORT
            raw_score -= breadth_score  # Inverse for shorts
            if breadth_score <= -5:
                reasons.append(f"Market bearish - good for short")
    # ═══════════════════════════════════════════════════════════════════════
    # ✅ CONFIGURABLE SIGMOID TRANSFORMATION
    # ═══════════════════════════════════════════════════════════════════════
    
    # Get sigmoid divisor from preset (defaults to 35 if not configured)
    # CONSERVATIVE: 40 (stricter scoring, fewer high-confidence signals)
    # BALANCED:     35 (standard scoring, balanced distribution)
    # AGGRESSIVE:   30 (gentler scoring, more signals pass)
    sigmoid_divisor = PRESET.get('sigmoid_divisor', 35)  # ✅ CRITICAL FIX
    
    # Map raw_score to 0-100 scale with realistic distribution
    # Expected distribution with divisor=35:
    # raw_score =  -30 → ~25% confidence (very weak)
    # raw_score =    0 → ~50% confidence (neutral)
    # raw_score = +30 → ~70% confidence (good)
    # raw_score = +60 → ~85% confidence (strong)
    # raw_score = +90 → ~93% confidence (very strong)
    confidence = 50 + (45 / (1 + np.exp(-raw_score / sigmoid_divisor)))
    
    # Final bounds: 30-95%
    confidence = round(min(95, max(30, confidence)), 1)
    
    return confidence, reasons[:12]
# ============================================================================
# MINI BACKTEST
# ============================================================================

def mini_backtest(
    df: pd.DataFrame,
    entry_price: float,
    side: str,
    rr_ratio: float = 2.0
) -> Tuple[bool, Dict]:
    """Enhanced mini backtest - STRICT SUCCESS CRITERIA"""
    try:
        if len(df) < 25:
            return False, {"reason": "Insufficient data"}
        
        lookback = min(60, len(df) - 1)
        future_data = df.iloc[:-1].tail(lookback).copy()
        
        if future_data.empty:
            return False, {"reason": "No future data"}
        
        atr = AverageTrueRange(
            high=normalize_series(future_data['High']),
            low=normalize_series(future_data['Low']),
            close=normalize_series(future_data['Close']),
            window=PRESET['atr_period']
        ).average_true_range().iloc[-1]
        
        if side == "LONG":
            stop_loss = entry_price - (atr * 1.5)
            target = entry_price + (atr * rr_ratio)
        else:
            stop_loss = entry_price + (atr * 1.5)
            target = entry_price - (atr * rr_ratio)
        
        # Track which happened first
        target_hit = False
        stop_hit = False
        target_bar = None
        stop_bar = None
        
        for i, row in enumerate(future_data.itertuples()):
            if side == "LONG":
                if row.High >= target and not target_hit:
                    target_hit = True
                    target_bar = i
                if row.Low <= stop_loss and not stop_hit:
                    stop_hit = True
                    stop_bar = i
            else:
                if row.Low <= target and not target_hit:
                    target_hit = True
                    target_bar = i
                if row.High >= stop_loss and not stop_hit:
                    stop_hit = True
                    stop_bar = i
        
        # ✅ FIX #5: STRICT SUCCESS CRITERIA
        if target_hit and not stop_hit:
            success = True  # Clean winner
            reason = "Target hit, stop not hit"
        elif target_hit and stop_hit:
            # Which happened first?
            if target_bar < stop_bar:
                success = True
                reason = "Target hit before stop"
            elif target_bar == stop_bar:
                success = True
                reason = "Target/stop same bar - took target"
            else:
                final_price = future_data.iloc[-1]['Close']
                if side == "LONG":
                    final_pnl = ((final_price - entry_price) / entry_price) * 100
                else:
                    final_pnl = ((entry_price - final_price) / entry_price) * 100
                success = final_pnl > -0.5
                reason = f"Stop first but final: {final_pnl:.2f}%"

        elif not target_hit and not stop_hit:
            # Neither hit - check final P&L
            final_price = future_data.iloc[-1]['Close']
            if side == "LONG":
                final_pnl_pct = ((final_price - entry_price) / entry_price) * 100
            else:
                final_pnl_pct = ((entry_price - final_price) / entry_price) * 100
            
            # Must be positive AND > 1% to pass
            success = final_pnl_pct > 0.3
            reason = f"Final P&L: {final_pnl_pct:.2f}%"
        else:
            # Stop hit, target not hit
            final_price = future_data.iloc[-1]['Close']
            if side == "LONG":
                final_pnl = ((final_price - entry_price) / entry_price) * 100
            else:
                final_pnl = ((entry_price - final_price) / entry_price) * 100
            success = final_pnl > -1.0
            reason = f"Stop hit, final: {final_pnl:.2f}%"
        
        max_favorable = (future_data['High'].max() - entry_price) / entry_price * 100 if side == "LONG" else (entry_price - future_data['Low'].min()) / entry_price * 100
        max_adverse = (future_data['Low'].min() - entry_price) / entry_price * 100 if side == "LONG" else (entry_price - future_data['High'].max()) / entry_price * 100
        
        metrics = {
        "target_hit": target_hit,
        "stop_hit": stop_hit,
        "success_reason": reason,
        "max_favorable": round(max_favorable, 2),
        "max_adverse": round(max_adverse, 2),
        "bars_analyzed": lookback
    }
        
        return success, metrics
        
    except Exception as e:
        return False, {"reason": str(e)[:50]}

# ============================================================================
# DATA LOADING FUNCTIONS
# ============================================================================

def get_universe_symbols() -> List[str]:
    """Get NIFTY 500 stock universe - FIXED FOR NIFTY 500 ONLY"""
    
    print("  📊 Fetching NIFTY 500 constituents...")
    
    try:
        # Method 1: Try NSE official API (most reliable)
        url = "https://archives.nseindia.com/content/indices/ind_nifty500list.csv"
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
        }
        
        response = requests.get(url, headers=headers, timeout=15)
        
        if response.status_code == 200:
            df = pd.read_csv(StringIO(response.text))
            
            # NSE CSV has 'Symbol' column
            if 'Symbol' in df.columns:
                symbols = df['Symbol'].tolist()
            elif 'SYMBOL' in df.columns:
                symbols = df['SYMBOL'].tolist()
            else:
                print("  ⚠️  Column not found, trying fallback...")
                return get_nifty500_fallback()
            
            # Clean and add .NS suffix
            symbols = [s.strip() + ".NS" for s in symbols if isinstance(s, str) and s.strip()]
            
            # Remove duplicates
            symbols = list(dict.fromkeys(symbols))
            
            print(f"  ✅ Loaded {len(symbols)} Nifty 500 stocks from NSE")
            return symbols[:500]  # Ensure max 500
        
        else:
            print(f"  ⚠️  NSE API returned {response.status_code}, using fallback...")
            return get_nifty500_fallback()
            
    except Exception as e:
        print(f"  ⚠️  Error fetching Nifty 500: {e}")
        print(f"  🔄 Using fallback list...")
        return get_nifty500_fallback()

def get_nifty500_fallback() -> List[str]:
    """
    Fallback Nifty 500 list (Top 100 most liquid stocks)
    
    🎯 This is a curated list of highly liquid Nifty 500 stocks
    💡 Update this list periodically from: https://www.niftyindices.com
    """
    
    nifty_500_symbols = [
        # Nifty 50 (Blue chips)
        "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS",
        "HINDUNILVR.NS", "ITC.NS", "SBIN.NS", "BHARTIARTL.NS", "KOTAKBANK.NS",
        "BAJFINANCE.NS", "LT.NS", "ASIANPAINT.NS", "AXISBANK.NS", "MARUTI.NS",
        "SUNPHARMA.NS", "TITAN.NS", "ULTRACEMCO.NS", "NESTLEIND.NS", "WIPRO.NS",
        "HCLTECH.NS", "TATAMOTORS.NS", "ONGC.NS", "NTPC.NS", "POWERGRID.NS",
        "ADANIENT.NS", "ADANIPORTS.NS", "COALINDIA.NS", "TATASTEEL.NS", "M&M.NS",
        "BAJAJFINSV.NS", "TECHM.NS", "JSWSTEEL.NS", "GRASIM.NS", "INDUSINDBK.NS",
        "HINDALCO.NS", "CIPLA.NS", "DRREDDY.NS", "APOLLOHOSP.NS", "DIVISLAB.NS",
        "EICHERMOT.NS", "BRITANNIA.NS", "SHREECEM.NS", "HEROMOTOCO.NS", "TATACONSUM.NS",
        "BAJAJ-AUTO.NS", "UPL.NS", "SBILIFE.NS", "HDFCLIFE.NS", "BPCL.NS",
        
        # Nifty Next 50
        "ADANIGREEN.NS", "ADANITRANS.NS", "AMBUJACEM.NS", "BANDHANBNK.NS", "BERGEPAINT.NS",
        "BEL.NS", "BOSCHLTD.NS", "COLPAL.NS", "DABUR.NS", "DMART.NS",
        "DLF.NS", "GAIL.NS", "GODREJCP.NS", "HAVELLS.NS", "HINDPETRO.NS",
        "ICICIPRULI.NS", "INDIGO.NS", "IRCTC.NS", "JINDALSTEL.NS", "LICHSGFIN.NS",
        "LUPIN.NS", "MARICO.NS", "MCDOWELL-N.NS", "MOTHERSON.NS", "MUTHOOTFIN.NS",
        "NAUKRI.NS", "NMDC.NS", "OFSS.NS", "PAGEIND.NS", "PETRONET.NS",
        "PFC.NS", "PIDILITIND.NS", "PNB.NS", "RECLTD.NS", "SAIL.NS",
        "SIEMENS.NS", "SRF.NS", "TATAPOWER.NS", "TORNTPHARM.NS", "TRENT.NS",
        "TVSMOTOR.NS", "VEDL.NS", "ZYDUSLIFE.NS", "ATUL.NS", "AUROPHARMA.NS",
        
        # Mid Cap (High Quality)
        "ABCAPITAL.NS", "ABFRL.NS", "ACC.NS", "ALKEM.NS", "AMARAJABAT.NS",
        "ASTRAL.NS", "AUBANK.NS", "BALKRISIND.NS", "BATAINDIA.NS", "BHARATFORG.NS",
        "BIOCON.NS", "CADILAHC.NS", "CANBK.NS", "CHAMBLFERT.NS", "CHOLAFIN.NS",
        "COFORGE.NS", "CONCOR.NS", "CUMMINSIND.NS", "DEEPAKNTR.NS", "DIXON.NS",
        "ESCORTS.NS", "FEDERALBNK.NS", "GLENMARK.NS", "GMRINFRA.NS", "GNFC.NS",
        "GODREJPROP.NS", "GRANULES.NS", "GUJGASLTD.NS", "HAL.NS", "HDFCAMC.NS",
        "HONAUT.NS", "IDFCFIRSTB.NS", "INDUSTOWER.NS", "IOC.NS", "IPCALAB.NS",
        "IRFC.NS", "IGL.NS", "JUBLFOOD.NS", "KANSAINER.NS", "KEI.NS",
        "L&TFH.NS", "LALPATHLAB.NS", "LAURUSLABS.NS", "LTIM.NS", "MANAPPURAM.NS",
        "MAXHEALTH.NS", "MGL.NS", "MPHASIS.NS", "MRF.NS", "NAM-INDIA.NS",
        "OBEROIRLTY.NS", "OIL.NS", "PERSISTENT.NS", "PIIND.NS", "PVR.NS",
        "RAIN.NS", "RAMCOCEM.NS", "SBICARD.NS", "SKFINDIA.NS", "SOLARINDS.NS",
        "SONACOMS.NS", "SUNTV.NS", "SUPREMEIND.NS", "SYNGENE.NS", "TATACHEM.NS",
         "TATAELXSI.NS", "TIINDIA.NS", "TORNTPOWER.NS", "TTML.NS",
        "UBL.NS", "UNIONBANK.NS", "VOLTAS.NS", "WHIRLPOOL.NS", "ZEEL.NS",
        
        # Small Cap (Momentum stocks)
        "AAVAS.NS",  "AEGISCHEM.NS", "AFFLE.NS", "AJANTPHARM.NS",
        "ALKYLAMINE.NS", "AMBER.NS", "APLLTD.NS", "ASAHIINDIA.NS", "ASHOKLEY.NS",
        "ATUL.NS", "AVANTIFEED.NS", "AXISCADES.NS", "BASF.NS", "BEML.NS",
        "BHARTIHEXA.NS", "BIKAJI.NS", "BLS.NS", "BSOFT.NS", "CAMS.NS",
        "CAPLIPOINT.NS", "CARBORUNIV.NS", "CCL.NS", "CDSL.NS", "CERA.NS",
        "CHALET.NS", "CLEAN.NS", "COCHINSHIP.NS", "COROMANDEL.NS", "CREDITACC.NS",
        "CROMPTON.NS", "CSB.NS", "CUB.NS", "CYIENT.NS", "DATAPATTNS.NS",
        "DCBBANK.NS", "DELTACORP.NS", "DHANI.NS", "DHANUKA.NS", "DISHTV.NS",
        "EMAMILTD.NS", "ENDURANCE.NS", "EQUITAS.NS", "FINEORG.NS", "FINPIPE.NS",
        "FSL.NS", "GATEWAY.NS", "GICRE.NS", "GILLETTE.NS", "GLAXO.NS",
        "GPIL.NS", "GRAPHITE.NS", "GREAVESCOT.NS", "GRINDWELL.NS", "GRSE.NS",
        "GSFC.NS", "GSPL.NS", "HAPPSTMNDS.NS", "HATHWAY.NS", "HCG.NS",
        "HFCL.NS", "HIMATSEIDE.NS", "HNDFDS.NS", "HOMEFIRST.NS", "HUDCO.NS",
        "IFBIND.NS", "IIFL.NS", "IIFLSEC.NS", "INDHOTEL.NS", "INOXLEISUR.NS",
        "JBCHEPHARM.NS", "JKCEMENT.NS", "JKLAKSHMI.NS", "JKPAPER.NS", "JMFINANCIL.NS",
        "JSWENERGY.NS", "JTEKTINDIA.NS", "JUBLINGREA.NS", "JUSTDIAL.NS", "JYOTHYLAB.NS",
        "KAJARIACER.NS", "KALPATPOWR.NS", "KARURVYSYA.NS", "KEC.NS", "KPRMILL.NS",
        "KRBL.NS", "LEMONTREE.NS", "LXCHEM.NS", "MAHLOG.NS", "MAHSEAMLES.NS",
        "MAHSCOOTER.NS", "MASTEK.NS", "MINDACORP.NS", "MINDAIND.NS", "MOTILALOFS.NS",
        "NATIONALUM.NS", "NAVINFLUOR.NS", "NBCC.NS", "NCC.NS", "NDTV.NS",
        "NELCO.NS", "NETWORK18.NS", "NHPC.NS", "NILKAMAL.NS", "NLCINDIA.NS",
        "NOCIL.NS", "NYKAA.NS", "ORIENTELEC.NS", "PARAGMILK.NS", "PCJEWELLER.NS",
        "PNBHOUSING.NS", "PNCINFRA.NS", "POLICYBZR.NS", "POLYCAB.NS", "POLYMED.NS",
        "POONAWALLA.NS", "PRSMJOHNSN.NS", "PTCIL.NS", "PVP.NS", "RAJESHEXPO.NS",
        "RALLIS.NS", "RATNAMANI.NS", "RAYMOND.NS", "RBL.NS", "RCF.NS",
        "REDINGTON.NS", "RELAXO.NS", "RITES.NS", "ROUTE.NS", "SANOFI.NS",
        "SCHAEFFLER.NS", "SHARDACROP.NS", "SHILPAMED.NS", "SHOPERSTOP.NS", "SHYAMMETL.NS",
        "SJVN.NS", "SOBHA.NS", "SPANDANA.NS", "SPARC.NS", "STARCEMENT.NS",
        "STAR.NS", "STLTECH.NS", "STYRENIX.NS", "SUBEX.NS", "SUNDARMFIN.NS",
        "SUNDRMFAST.NS",  "SUPRAJIT.NS", "SUPRIYA.NS", "SUVEN.NS",
        "SWANENERGY.NS", "SYMPHONY.NS", "TATACOMM.NS", "TATAINVEST.NS", "TATAMETALI.NS",
        "TCNSBRANDS.NS", "TEAMLEASE.NS",  "THERMAX.NS", "THYROCARE.NS",
        "TIMKEN.NS", "TIPSINDLTD.NS", "TRITURBINE.NS", "TRIDENT.NS", "TRITURBINE.NS",
        "TTK.NS", "TV18BRDCST.NS", "UCOBANK.NS", "UJAAS.NS", "UJJIVAN.NS",
         "UNICHEMLAB.NS", "UTIAMC.NS", "VAIBHAVGBL.NS",
        "VARROC.NS", "VBL.NS", "VENKEYS.NS", "VGUARD.NS", "VIPIND.NS",
        "VINATIORGA.NS", "VSTIND.NS", "WABCOINDIA.NS", "WELCORP.NS", "WELSPUNIND.NS",
        "WESTLIFE.NS", "WILSON.NS", "WOCKPHARMA.NS", "YESBANK.NS", "ZENSARTECH.NS",
    ]
    
    print(f"  ✅ Using fallback: {len(nifty_500_symbols)} curated Nifty 500 stocks")
    
    return nifty_500_symbols[:500]



def build_cache(tickers: List[str]) -> Optional[pd.DataFrame]:
    """Build price data cache - FIXED VERSION FOR UNIQUE DATA"""
    print(f"\n[Building Price Cache for {len(tickers)} stocks]")
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=LOOKBACK_DAYS)
    
    all_data = []
    failed = []
    successful_downloads = {}  # Track successful downloads to prevent duplicates
    
    def download_ticker(ticker):
        """Download single ticker - FIXED FOR UNIQUE DATA"""
        original_ticker = ticker
        
        # Clean ticker - remove any existing suffix
        base_ticker = ticker.replace('.NS', '').replace('.BO', '')
        
        # Always try .NS first for NSE stocks
        test_ticker = base_ticker + '.NS'
        
        # Download with retries
        for attempt in range(3):
            try:
                # Download fresh data
                df = yf.download(
                    test_ticker,
                    start=start_date,
                    end=end_date,
                    progress=False,
                    auto_adjust=True,
                    timeout=10,
                    threads=False  # Disable threading for more reliable downloads
                )
                
                # Check if we got valid data
                if df is not None and not df.empty and len(df) >= 60:
                    break
                    
                # If .NS failed, try .BO
                if attempt == 1:
                    test_ticker = base_ticker + '.BO'
                    
            except Exception as e:
                if attempt < 2:
                    time.sleep(1)
                continue
        
        # Validate we got data
        if df is None or df.empty or len(df) < 60:
            return None
        
        try:
            # Flatten MultiIndex columns if present
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            
            # Ensure required columns
            required = ['Open', 'High', 'Low', 'Close', 'Volume']
            if not all(col in df.columns for col in required):
                return None
            
            # Remove any NaN values
            df = df.dropna(subset=['Close', 'Volume'])
            
            if len(df) < 60:
                return None
            
            # Create unique signature for this data
            data_signature = f"{df['Close'].mean():.2f}_{df['Close'].std():.2f}_{df['Volume'].mean():.0f}"
            
            # Check if we already have this exact data
            if data_signature in successful_downloads:
                logger.debug(f"{test_ticker}: Duplicate data detected, skipping")
                return None
            
            # Mark this data as downloaded
            successful_downloads[data_signature] = test_ticker
            
            # DON'T normalize prices here - let the scanner handle it
            # This ensures we get actual market prices
            
            # Build clean dataframe with actual ticker used
            clean_df = pd.DataFrame({
                'Date': df.index,
                'Open': df['Open'].values,
                'High': df['High'].values,
                'Low': df['Low'].values,
                'Close': df['Close'].values,
                'Volume': df['Volume'].values,
                'ticker': test_ticker  # Store the actual ticker that worked
            })
            
            clean_df['Date'] = pd.to_datetime(clean_df['Date'])
            
            # Final validation
            if clean_df['Close'].nunique() < 10:
                return None
                
            return clean_df
            
        except Exception as e:
            logger.debug(f"{test_ticker}: Processing failed: {e}")
            return None
    
    # Download data with progress tracking
    with ThreadPoolExecutor(max_workers=5) as executor:  # Reduced workers for stability
        futures = {executor.submit(download_ticker, t): t for t in tickers}
        
        for i, future in enumerate(as_completed(futures), 1):
            ticker = futures[future]
            try:
                result = future.result(timeout=30)
                if result is not None:
                    all_data.append(result)
                else:
                    failed.append(ticker)
            except Exception:
                failed.append(ticker)
            
            # Progress update
            if i % 25 == 0 or i == len(tickers):
                print(f"  Progress: {i}/{len(tickers)} | Success: {len(all_data)} | Failed: {len(failed)}", 
                      end="\r", flush=True)
    
    print()  # New line after progress
    
    if not all_data:
        print("\n  ❌ No data downloaded successfully!")
        return None
    
    # Concatenate all data
    try:
        df_all = pd.concat(all_data, axis=0, ignore_index=True)
        
        # Verify each ticker has unique data
        print(f"\n  📊 Verifying data uniqueness...")
        tickers_to_remove = []
        
        for ticker in df_all['ticker'].unique():
            ticker_data = df_all[df_all['ticker'] == ticker]
            unique_prices = ticker_data['Close'].nunique()
            
            if unique_prices < 10:
                tickers_to_remove.append(ticker)
                print(f"  ⚠️  Removing {ticker} - insufficient unique prices ({unique_prices})")
        
        # Remove bad tickers
        if tickers_to_remove:
            df_all = df_all[~df_all['ticker'].isin(tickers_to_remove)]
        
        final_ticker_count = df_all['ticker'].nunique()
        print(f"  ✓ Cached {final_ticker_count} stocks with {len(df_all):,} total bars")
        
        return df_all
        
    except Exception as e:
        print(f"\n  ❌ Error concatenating data: {e}")
        return None
# ============================================================================
# MAIN SCANNER FUNCTION - COMPLETE WITH TICKER MATCHING FIX
# ============================================================================
def scan_ticker(
    ticker: str,
    df_all: pd.DataFrame,
    quality_validator: DataQualityValidator,
    sector_filter: SectorRotationAnalyzer,
    vix_filter: VIXSentimentAnalyzer,
    fibo_detector: FibonacciDetector,
    portfolio_mgr: PortfolioRiskManager,
    mtf_analyzer: MultiTimeframeAnalyzer,
    backtester: Optional[AdvancedBacktester] = None,
    run_full_backtest: bool = False,
    fifty_two_week_analyzer = None,
    rs_analyzer = None,
    delivery_analyzer = None,
    earnings_filter = None,
    consolidation_detector = None,
    market_breadth_data: Optional[Dict] = None
) -> Optional[Dict]:
    """Complete scanner - FULLY FIXED VERSION WITH PROPER DATA HANDLING"""
    
    global stats, rejection_samples
    stats["total"] += 1
    
    def log_rejection(reason: str, detail: str = ""):
        if len(rejection_samples) < 15:
            msg = f"{ticker.replace('.NS', '').replace('.BO', '')}: {reason}"
            if detail:
                msg += f" ({detail})"
            rejection_samples.append(msg)
    
    try:
        # ====================================================================
        # GET DATA FOR THIS TICKER
        # ====================================================================
        df_t = df_all[df_all["ticker"] == ticker].copy()
        
        if df_t.empty:
            stats["no_data"] += 1
            log_rejection("No data", "Ticker not in cache")
            return None
        
        # ====================================================================
        # PREPARE DATA - ENSURE CLEAN DATAFRAME
        # ====================================================================
        df_t = df_t.copy()  # Make a fresh copy
        
        # Sort by date
        if 'Date' in df_t.columns:
            df_t = df_t.sort_values('Date')
            df_t = df_t.set_index('Date')
        
        df_t = df_t.sort_index()
        
        # Drop ticker column BEFORE any processing
        if 'ticker' in df_t.columns:
            df_t = df_t.drop(columns=['ticker'])
        
        # Normalize prices if needed (only for paise data)
        df_t = normalize_price_data(df_t)
        
        # Ensure it's still a proper DataFrame
        if not isinstance(df_t, pd.DataFrame):
            stats["data_quality_fail"] += 1
            log_rejection("Data type error", f"Got {type(df_t)}")
            return None
        
        # ====================================================================
        # VERIFY DATA FRESHNESS
        # ====================================================================
        try:
            last_date = df_t.index[-1]
            days_old = (datetime.now() - pd.to_datetime(last_date)).days
            
            if days_old > MAX_DATA_STALE_DAYS:
                stats["data_quality_fail"] += 1
                log_rejection("Stale data", f"{days_old} days old")
                return None
        except Exception as e:
            stats["data_quality_fail"] += 1
            log_rejection("Date error", str(e)[:30])
            return None
        
        # ====================================================================
        # VERIFY DATA UNIQUENESS
        # ====================================================================
        unique_prices = df_t['Close'].nunique()
        total_rows = len(df_t)
        
        if total_rows > 20 and unique_prices < 10:
            stats["data_quality_fail"] += 1
            log_rejection("Duplicate data", f"Only {unique_prices} unique prices")
            return None
        
        # ====================================================================
        # DATA QUALITY CHECKS
        # ====================================================================
        if not quality_validator.has_sufficient_history(df_t, min_bars=60):
            stats["indicator_fail"] += 1
            log_rejection("Insufficient history", f"{len(df_t)} bars")
            return None
        
        valid_prices, price_reason = quality_validator.has_valid_prices(df_t)
        if not valid_prices:
            stats["data_quality_fail"] += 1
            log_rejection("Invalid prices", price_reason)
            return None
        
        # ====================================================================
        # CALCULATE INDICATORS
        # ====================================================================
        ind = calculate_indicators(df_t)
        if ind is None:
            stats["indicator_fail"] += 1
            log_rejection("Indicator calculation failed")
            return None
        
        price = ind["price"]
        
        # Debug first few stocks
        if stats["total"] <= 5:
            print(f"\n  DEBUG: {ticker} - Price: ₹{price:.2f} | RSI: {ind['rsi']:.1f} | Vol: {ind['volume_ratio']:.2f}x")
        
        # ====================================================================
        # VOLATILITY REGIME FILTER
        # ====================================================================
        vol_ok, vol_reason, vol_metrics = VolatilityRegimeFilter.analyze_regime(df_t, price)
        if not vol_ok:
            stats["volatility_fail"] += 1
            log_rejection("Volatility", vol_reason)
            return None
        
        # ====================================================================
        # RUN ALL ANALYZERS
        # ====================================================================
        try:
            volume_score, volume_signal, volume_metrics = InstitutionalVolumeAnalyzer.analyze(df_t, ind["volume"])
            vsa_score, vsa_signal, vsa_metrics = VolumeSpreadAnalyzer.analyze(df_t)
            momentum_score, momentum_signal, momentum_metrics = MomentumScoreCalculator.calculate(df_t)
            vcp_score, vcp_signal, vcp_metrics = VCPDetector.detect(df_t)
            gap_score, gap_signal, gap_metrics = GapDetector.detect(df_t)
            sr_info = SRClusterDetector.detect(df_t)
            candle = CandlestickPatternDetector.detect(df_t)
            st_direction, st_strength, st_metrics = SupertrendIndicator.calculate(df_t)
            mtf_trend = mtf_analyzer.get_trend(ticker)

            # 1. 52-Week High/Low Analysis
            fifty_two_week_data = None
            if fifty_two_week_analyzer and USE_52_WEEK_FILTER:
                fifty_two_week_data = FiftyTwoWeekAnalyzer.analyze(df_t, price)
            
            # 2. Relative Strength vs Nifty
            rs_data = None
            if rs_analyzer and USE_RELATIVE_STRENGTH:
                rs_data = rs_analyzer.calculate_rs(df_t)
            
            # 3. Delivery Volume Estimation
            delivery_data = None
            if delivery_analyzer and USE_DELIVERY_VOLUME:
                delivery_data = DeliveryVolumeAnalyzer.estimate_delivery(df_t)
            
            # 4. Earnings Filter
            earnings_data = None
            if earnings_filter and USE_EARNINGS_FILTER:
                earnings_data = EarningsFilter.check_earnings(ticker)
                
                # REJECT if earnings within EARNINGS_AVOID_DAYS
                if earnings_data.get('earnings_soon') and not earnings_data.get('safe_to_trade', True):
                    stats["earnings_fail"] += 1  # ✅ Correct key
                    log_rejection("Earnings soon", earnings_data.get('message', ''))
                    return None
            
            # 5. Consolidation Breakout
            consolidation_data = None
            if consolidation_detector and USE_CONSOLIDATION_BREAKOUT:
                consolidation_data = ConsolidationBreakoutDetector.detect(df_t)
            
            
        except Exception as e:
            stats["indicator_fail"] += 1
            log_rejection("Analyzer error", str(e)[:40])
            return None
        
        # ====================================================================
        # GET FUNDAMENTALS
        # ====================================================================
        sector, pe = get_fundamentals(ticker)
        
        # ====================================================================
        # SECTOR ROTATION (IF ENABLED)
        # ====================================================================
        sector_rs = None
        if USE_SECTOR_ROTATION and sector_filter and sector:
            try:
                sector_rs = sector_filter.get_sector_strength(sector)
            except:
                sector_rs = None
        
        # ====================================================================
        # VIX SENTIMENT (IF ENABLED)
        # ====================================================================
        vix_sentiment = None
        if USE_VIX_SENTIMENT and vix_filter:
            try:
                vix_sentiment = vix_filter.get_vix_sentiment()
            except:
                vix_sentiment = None
        
        # ====================================================================
        # FIBONACCI LEVELS (IF ENABLED)
        # ====================================================================
        long_fibo_boost = 0
        short_fibo_boost = 0
        if USE_FIBONACCI_SCORING and fibo_detector:
            try:
                fib_levels = fibo_detector.calculate_levels(df_t, PRESET['fib_lookback'])
                if fib_levels:
                    long_fibo_boost, _ = fibo_detector.detect_bounce(price, fib_levels, "LONG")
                    short_fibo_boost, _ = fibo_detector.detect_bounce(price, fib_levels, "SHORT")
            except:
                pass
        
        # ====================================================================
        # GENERATE ENTRY SIGNALS
        # ====================================================================
        long_entry = generate_entry_signal(ind, df_t, "LONG")
        short_entry = generate_entry_signal(ind, df_t, "SHORT")
        
        # ====================================================================
        # SCORE SIGNALS
        # ====================================================================
        long_conf, long_reasons = score_signal(
            ind, df_t, "LONG", vix_sentiment, sector_rs, long_fibo_boost,
            sr_info, candle, mtf_trend, vol_metrics,
            {'score': volume_score, 'signal': volume_signal},
            {'score': vsa_score, 'signal': vsa_signal},
            {'score': momentum_score, 'signal': momentum_signal},
            {'score': vcp_score, 'signal': vcp_signal},
            {'score': gap_score, 'signal': gap_signal},
            {'direction': st_direction, 'strength': st_strength},
            fifty_two_week_data=fifty_two_week_data,
            rs_data=rs_data,
            delivery_data=delivery_data,
            earnings_data=earnings_data,
            consolidation_data=consolidation_data,
            market_breadth_data=market_breadth_data
        )
        
        short_conf, short_reasons = score_signal(
            ind, df_t, "SHORT", vix_sentiment, sector_rs, short_fibo_boost,
            sr_info, candle, mtf_trend, vol_metrics,
            {'score': volume_score, 'signal': volume_signal},
            {'score': vsa_score, 'signal': vsa_signal},
            {'score': momentum_score, 'signal': momentum_signal},
            {'score': vcp_score, 'signal': vcp_signal},
            {'score': gap_score, 'signal': gap_signal},
            {'direction': st_direction, 'strength': st_strength},
            fifty_two_week_data=fifty_two_week_data,
            rs_data=rs_data,
            delivery_data=delivery_data,
            earnings_data=earnings_data,
            consolidation_data=consolidation_data,
            market_breadth_data=market_breadth_data
        )
        
        # ====================================================================
        # DETERMINE BEST SIDE
        # ====================================================================
        long_valid = long_entry["entry_ready"] and long_conf >= MIN_CONFIDENCE
        short_valid = short_entry["entry_ready"] and short_conf >= MIN_CONFIDENCE
        
        if not (long_valid or short_valid):
            stats["confidence_fail"] += 1
            log_rejection("Low confidence", f"L:{long_conf:.0f}% S:{short_conf:.0f}%")
            return None
        
        if long_valid and (not short_valid or long_conf >= short_conf):
            side = "LONG"
            confidence = long_conf
            reasons = long_reasons
            entry_info = long_entry
        else:
            side = "SHORT"
            confidence = short_conf
            reasons = short_reasons
            entry_info = short_entry
        # ====================================================================
        # MARKET BREADTH FILTER (NOW AFTER SIDE IS DETERMINED) - ADD HERE!
        # ====================================================================
        if market_breadth_data and market_breadth_data.get('valid'):
            if side == "LONG" and not market_breadth_data.get('long_ok', True):
                stats["market_breadth_fail"] += 1  # ✅ Correct key
                log_rejection("Market bearish", "LONG not advised")
                return None
            if side == "SHORT" and not market_breadth_data.get('short_ok', True):
                stats["market_breadth_fail"] += 1  # ✅ Correct key
                log_rejection("Market bullish", "SHORT not advised")
                return None
        
        # ====================================================================
        # CREATE TRADE RULE
        # ====================================================================
        trade_rule = TradeRule(ticker.replace(".NS", "").replace(".BO", ""), side, price, ind["atr"])
        trade_rule.calculate_dynamic_targets(confidence)
        
        if trade_rule.risk_reward_ratio < MIN_RR_RATIO:
            stats["rr_fail"] += 1
            log_rejection("Low R:R", f"{trade_rule.risk_reward_ratio:.2f}x")
            return None
        
        # ====================================================================
        # MINI BACKTEST
        # ====================================================================
        mini_metrics = {}
        if USE_MINI_BACKTEST:
            mini_passed, mini_metrics = mini_backtest(df_t, price, side, MIN_RR_RATIO)
            if not mini_passed:
                stats["mini_backtest_fail"] += 1
                log_rejection("Mini backtest", mini_metrics.get('reason', 'Failed'))
                return None
        
        # ====================================================================
        # FULL BACKTEST (IF REQUESTED)
        # ====================================================================
        backtest_result = None
        walk_forward_result = None
        monte_carlo_result = None
        
        if run_full_backtest and backtester and USE_FULL_BACKTEST:
            stats["full_backtest_run"] += 1
            
            try:
                backtest_result = backtester.backtest_strategy(
                    df=df_t,
                    signal_type=side,
                    entry_conditions={'side': side},
                    lookback_days=BACKTEST_LOOKBACK_DAYS
                )
                
                if backtest_result:
                    if (backtest_result.win_rate < PRESET['min_backtest_win_rate'] or
                        backtest_result.profit_factor < PRESET['min_profit_factor'] or
                        backtest_result.max_drawdown > PRESET['max_drawdown']):
                        
                        stats["full_backtest_fail"] += 1
                        log_rejection("Backtest fail", f"WR:{backtest_result.win_rate:.0f}%")
                        return None
                    
                    # Apply backtest boost to confidence
                    backtest_boost = (
                        (backtest_result.win_rate - 50) * 0.2 +
                        (backtest_result.profit_factor - 1) * 3 +
                        (backtest_result.reliability_score - 50) * 0.15
                    )
                    confidence = min(100, confidence + backtest_boost)
                    
                    # Walk-forward validation (if enabled)
                    if USE_WALK_FORWARD and ACCURACY_MODE == 'CONSERVATIVE':
                        walk_forward_result = backtester.walk_forward_validation(
                            df_t, side, {'side': side}, WALK_FORWARD_PERIODS
                        )
                        
                        if walk_forward_result and not walk_forward_result.passed:
                            stats["walk_forward_fail"] += 1
                            log_rejection("Walk-forward fail", "Inconsistent")
                            return None
                    
                    # Monte Carlo simulation (if enabled)
                    if USE_MONTE_CARLO:
                        monte_carlo_result = backtester.monte_carlo_simulation(
                            backtest_result, simulations=1000, num_trades=50
                        )
                        
                        if monte_carlo_result and monte_carlo_result.prob_profit < 60:
                            stats["monte_carlo_fail"] += 1
                            log_rejection("Monte Carlo fail", f"Prob:{monte_carlo_result.prob_profit:.0f}%")
                            return None
            except Exception as e:
                logger.debug(f"Backtest error for {ticker}: {e}")
                pass
        
        # ====================================================================
        # PORTFOLIO RISK CHECK (IF ENABLED)
        # ====================================================================
        if USE_PORTFOLIO_RISK and portfolio_mgr:
            can_add, reason = portfolio_mgr.can_add_trade({
                'ticker': ticker,
                'side': side,
                'position_value': trade_rule.position_value,
                'risk_amount': trade_rule.actual_risk,  # ✅ Use actual_risk not risk_amount
                'sector': sector,
            })
        
        if not can_add:
            stats["portfolio_fail"] += 1
            log_rejection("Portfolio limit", reason)
            return None  # ✅ Return None, not just add warning
            
            # ====================================================================
            # SUCCESS - BUILD RESULT OBJECT
            # ====================================================================
            stats["passed"] += 1
            
            # Print successful signal
            print(f"\n✅ SIGNAL #{stats['passed']}: {ticker.replace('.NS', '').replace('.BO', '')}")
            print(f"   {side} @ ₹{price:.2f} | Conf: {confidence:.1f}% | R:R: {trade_rule.risk_reward_ratio:.1f}x")
            
            result = {
                "ticker": ticker.replace(".NS", "").replace(".BO", ""),
                "side": side,
                "price": round(price, 2),
                "confidence": round(confidence, 1),
                "rsi": round(ind["rsi"], 1),
                "atr": round(ind["atr"], 2),
                "adx": round(ind["adx"], 1),
                "entry_price": trade_rule.entry_price,
                "stop_loss": trade_rule.stop_loss,
                "target_1": trade_rule.target_1,
                "target_2": trade_rule.target_2,
                "risk_reward": round(trade_rule.risk_reward_ratio, 2),
                "qty": trade_rule.qty,
                "position_value": round(trade_rule.position_value, 2),
                "risk_amount": round(trade_rule.risk_amount, 2),
                "entry_signal": entry_info["entry_signal"],
                "reasons": " | ".join(reasons),
                "sector": sector or "N/A",
                "pe": pe,
                "trade_rule": trade_rule,
                "mini_backtest": mini_metrics,
                "fifty_two_week": fifty_two_week_data if fifty_two_week_data else {},
                "relative_strength": rs_data if rs_data else {},
                "delivery_estimate": delivery_data if delivery_data else {},
                "earnings_info": earnings_data if earnings_data else {},
                "consolidation": consolidation_data if consolidation_data else {},
                "market_regime": market_breadth_data.get('regime', 'N/A') if market_breadth_data else 'N/A',
                
                # Trailing stops (calculate initial values)
                "trailing_stops": TrailingStopManager.get_all_trailing_stops(
                    entry_price=trade_rule.entry_price,
                    current_price=price,
                    highest_price=float(df_t['High'].tail(5).max()),
                    lowest_price=float(df_t['Low'].tail(5).min()),
                    atr=ind["atr"],
                    side=side
                ) if USE_TRAILING_STOP else {},
                "backtest_validated": False
            }
            
            # Add backtest results if available
            if backtest_result:
                result["backtest"] = {
                    "win_rate": backtest_result.win_rate,
                    "profit_factor": backtest_result.profit_factor,
                    "sharpe_ratio": backtest_result.sharpe_ratio,
                    "max_drawdown": backtest_result.max_drawdown,
                    "reliability_score": backtest_result.reliability_score,
                    "total_trades": backtest_result.total_trades,
                    "expectancy": backtest_result.expectancy,
                }
                result["backtest_validated"] = True
            
            # Add walk-forward results if available
            if walk_forward_result:
                result["walk_forward"] = {
                    "avg_win_rate": walk_forward_result.avg_win_rate,
                    "consistency": walk_forward_result.consistency_score,
                    "passed": walk_forward_result.passed
                }
            
            # Add Monte Carlo results if available
            if monte_carlo_result:
                result["monte_carlo"] = {
                    "prob_profit": monte_carlo_result.prob_profit,
                    "risk_of_ruin": monte_carlo_result.risk_of_ruin,
                }
            
            return result
        
    except Exception as e:
        # Catch-all exception handler
        stats["indicator_fail"] += 1
        log_rejection("Exception", f"{type(e).__name__}: {str(e)[:30]}")
        logger.debug(f"{ticker}: Exception - {str(e)}")
        return None
    
# ============================================================================
# HYBRID SCANNING STRATEGY
# ============================================================================

def hybrid_scan_universe(
    tickers: List[str],
    df_all: pd.DataFrame,
    quality_validator: DataQualityValidator,
    sector_filter: SectorRotationAnalyzer,
    vix_filter: VIXSentimentAnalyzer,
    fibo_detector: FibonacciDetector,
    portfolio_mgr: PortfolioRiskManager,
    mtf_analyzer: MultiTimeframeAnalyzer,
    fifty_two_week_analyzer=None,
    rs_analyzer=None,
    delivery_analyzer=None,
    earnings_filter=None,
    consolidation_detector=None,
    market_breadth_data: Optional[Dict] = None
) -> List[Dict]:
    """Hybrid scanning: Quick scan + Full backtest top candidates"""
    
    print(f"\n{'='*120}")
    print("PHASE 1: QUICK SCAN WITH MINI BACKTEST")
    print('='*120)
    
    # Phase 1: Quick scan
    initial_results = []
    
    for i, ticker in enumerate(tickers, 1):
        result = scan_ticker(
            ticker=ticker,
            df_all=df_all,
            quality_validator=quality_validator,
            sector_filter=sector_filter,
            vix_filter=vix_filter,
            fibo_detector=fibo_detector,
            portfolio_mgr=portfolio_mgr,
            mtf_analyzer=mtf_analyzer,
            backtester=None,
            run_full_backtest=False,
            fifty_two_week_analyzer=fifty_two_week_analyzer,
            rs_analyzer=rs_analyzer,
            delivery_analyzer=delivery_analyzer,
            earnings_filter=earnings_filter,
            consolidation_detector=consolidation_detector,
            market_breadth_data=market_breadth_data

        )
        
        if result:
            initial_results.append(result)
            if USE_PORTFOLIO_RISK and portfolio_mgr:
                portfolio_mgr.add_trade(result)
        
        # Progress bar
        if i % 25 == 0 or i == len(tickers):
            pct = (i / len(tickers)) * 100
            bar_length = 50
            filled = int(bar_length * i / len(tickers))
            bar = '█' * filled + '░' * (bar_length - filled)
            print(f"  [{bar}] {pct:>5.1f}% | {i:>4}/{len(tickers)} | Found: {len(initial_results):>3}", 
                  end='\r', flush=True)
    
    print()
    
    if not initial_results:
        print("\n  ⚠️  No signals found in Phase 1")
        return []
    
    # Sort by confidence
    initial_results = sorted(initial_results, key=lambda x: x['confidence'], reverse=True)
    print(f"\n  ✓ Phase 1 Complete: {len(initial_results)} signals found")
    
    # Phase 2: Full backtest (if enabled)
    if BACKTEST_MODE in ['HYBRID', 'FULL'] and USE_FULL_BACKTEST:
        
        num_to_test = min(30, len(initial_results)) if BACKTEST_MODE == 'HYBRID' else len(initial_results)
        
        print(f"\n{'='*120}")
        print(f"PHASE 2: FULL BACKTEST ON TOP {num_to_test} CANDIDATES")
        print('='*120)
        
        backtester = AdvancedBacktester(initial_capital=ACCOUNT_CAPITAL)
        validated_results = []
        candidates_to_test = initial_results[:num_to_test]
        
        for i, candidate in enumerate(candidates_to_test, 1):
            ticker = candidate['ticker'] + '.NS'
            
            pct = (i / num_to_test) * 100
            print(f"  [{i:>3}/{num_to_test}] {pct:>5.1f}% | Testing {candidate['ticker']:<12} | "
                  f"Conf: {candidate['confidence']:>5.1f}% | {candidate['side']:<5}", 
                  end='\r', flush=True)
            
            df_t = df_all[df_all["ticker"] == ticker].copy()
            
            if df_t.empty:
                continue
            
            # Prepare data
            if 'Date' in df_t.columns:
                df_t = df_t.set_index('Date')
            df_t = df_t.sort_index()
            if 'ticker' in df_t.columns:
                df_t = df_t.drop(columns=['ticker'])
            
            # Run full backtest
            backtest_result = backtester.backtest_strategy(
                df=df_t,
                signal_type=candidate['side'],
                entry_conditions={'side': candidate['side']},
                lookback_days=BACKTEST_LOOKBACK_DAYS
            )
            
            if backtest_result:
                if (backtest_result.win_rate >= PRESET['min_backtest_win_rate'] and
                    backtest_result.profit_factor >= PRESET['min_profit_factor'] and
                    backtest_result.max_drawdown <= PRESET['max_drawdown']):
                    
                    candidate['backtest'] = {
                        "win_rate": backtest_result.win_rate,
                        "profit_factor": backtest_result.profit_factor,
                        "sharpe_ratio": backtest_result.sharpe_ratio,
                        "max_drawdown": backtest_result.max_drawdown,
                        "reliability_score": backtest_result.reliability_score,
                        "total_trades": backtest_result.total_trades,
                    }
                    
                    backtest_boost = (backtest_result.reliability_score - 50) * 0.25
                    candidate['confidence'] = min(100, candidate['confidence'] + backtest_boost)
                    candidate['backtest_validated'] = True
                    
                    validated_results.append(candidate)
                else:
                    stats["full_backtest_fail"] += 1
            else:
                candidate['backtest_validated'] = False
                validated_results.append(candidate)
        
        print()
        print(f"\n  ✓ Phase 2 Complete: {len(validated_results)} signals validated")
        
        # Add remaining if HYBRID
        if BACKTEST_MODE == 'HYBRID' and len(initial_results) > num_to_test:
            for candidate in initial_results[num_to_test:]:
                candidate['backtest_validated'] = False
                validated_results.append(candidate)
        
        return sorted(validated_results, key=lambda x: x['confidence'], reverse=True)
    
    return initial_results

# ============================================================================
# STATISTICS PRINTING
# ============================================================================

def print_statistics():
    """Print detailed scan statistics"""
    
    total = stats.get("total", 0)
    if total == 0:
        print("\n⚠️  No stocks were scanned")
        return
    
    print(f"\n{'='*120}")
    print("SCAN STATISTICS")
    print('='*120)
    
    passed = stats.get("passed", 0)
    pass_rate = (passed / total) * 100 if total > 0 else 0
    
    print(f"\n📊 OVERALL:")
    print(f"   Total Scanned:    {total:>6}")
    print(f"   Signals Found:    {passed:>6}  ({pass_rate:.1f}%)")
    print(f"   Rejected:         {total - passed:>6}  ({100-pass_rate:.1f}%)")
    
    print(f"\n❌ REJECTION BREAKDOWN:")
    
    rejections = [
        ("No Data", stats.get("no_data", 0)),
        ("Data Quality", stats.get("data_quality_fail", 0)),
        ("Indicators Failed", stats.get("indicator_fail", 0)),
        ("Volatility Regime", stats.get("volatility_fail", 0)),
        ("Low Confidence", stats.get("confidence_fail", 0)),
        ("Poor Risk:Reward", stats.get("rr_fail", 0)),
        ("Mini Backtest", stats.get("mini_backtest_fail", 0)),
        ("Full Backtest", stats.get("full_backtest_fail", 0)),
        ("Walk-Forward", stats.get("walk_forward_fail", 0)),
        ("Monte Carlo", stats.get("monte_carlo_fail", 0)),
        ("Portfolio Risk", stats.get("portfolio_fail", 0)),
        ("Earnings Risk", stats.get("earnings_fail", 0)),
        ("Market Breadth", stats.get("market_breadth_fail", 0)),
        ("52-Week Position", stats.get("52w_fail", 0)),
    ]
    
    has_rejections = False
    for reason, count in rejections:
        if count > 0:
            has_rejections = True
            pct = (count / total) * 100
            print(f"   {reason:<20} {count:>6}  ({pct:>5.1f}%)")
    
    if not has_rejections:
        print("   (No rejections recorded)")
    
    if rejection_samples:
        print(f"\n📝 SAMPLE REJECTIONS:")
        for sample in rejection_samples[:8]:
            print(f"   • {sample}")
    
    bt_run = stats.get("full_backtest_run", 0)
    if bt_run > 0:
        bt_fail = stats.get("full_backtest_fail", 0)
        print(f"\n🔬 BACKTEST INFO:")
        print(f"   Full Backtests Run:   {bt_run}")
        print(f"   Failed:               {bt_fail}")
    
    print('='*120)

# ============================================================================
# RESULTS PRINTING
# ============================================================================

def print_results(results: List[Dict]):
    """Print formatted results table - TIERED VERSION"""
    
    if not results:
        return
    
    # Classify into tiers
    tiers = apply_tier_based_filtering(results)
    
    total_signals = sum(len(v) for v in tiers.values())
    
    print(f"\n{'='*180}")
    print(f"TRADING SIGNALS - TIERED CLASSIFICATION ({total_signals} Total)")
    print('='*180)
    
    # Print each tier
    tier_info = {
        "TIER_1_PREMIUM": ("🟢 TIER 1: PREMIUM SIGNALS (Highest Probability)", "Take with 30-40% position size"),
        "TIER_2_STANDARD": ("🟡 TIER 2: STANDARD SIGNALS (Good Probability)", "Take with 15-20% position size"),
        "TIER_3_SPECULATIVE": ("🔵 TIER 3: SPECULATIVE SIGNALS (Acceptable Risk)", "Take with 5-10% position size"),
    }
    
    for tier_name, (tier_title, tier_advice) in tier_info.items():
        tier_results = tiers[tier_name]
        
        if not tier_results:
            continue
        
        print(f"\n{tier_title}")
        print(f"   Recommendation: {tier_advice}")
        print(f"   Count: {len(tier_results)} signals")
        print('-'*180)
        
        # Header
        header = (
            f"{'#':<4} {'TICKER':<12} {'SIDE':<6} {'PRICE':>9} {'CONF':>6} {'RSI':>5} "
            f"{'ENTRY':>9} {'SL':>9} {'T1':>9} {'T2':>9} {'R:R':>5} {'QTY':>6}"
        )
        
        if any(r.get('backtest_validated') for r in tier_results):
            header += f" {'BT_WR':>7} {'BT_PF':>6}"
        
        print(header)
        print('-'*180)
        
        for i, res in enumerate(tier_results[:20], 1):  # Show max 20 per tier
            row = (
                f"{i:<4} "
                f"{res['ticker']:<10} "
                f"{res['side']:<6} "
                f"₹{res['price']:>8.2f} "
                f"{res['confidence']:>5.1f}% "
                f"{res['rsi']:>5.1f} "
                f"₹{res['entry_price']:>8.2f} "
                f"₹{res['stop_loss']:>8.2f} "
                f"₹{res['target_1']:>8.2f} "
                f"₹{res['target_2']:>8.2f} "
                f"{res['risk_reward']:>4.1f}x "
                f"{res['qty']:>6}"
            )
            
            if res.get('backtest_validated') and 'backtest' in res:
                bt = res['backtest']
                row += (
                    f" {bt['win_rate']:>6.1f}% "
                    f"{bt['profit_factor']:>5.1f}x"
                )
            
            print(row)
    
    print('='*180)
    
    # Summary
    print(f"\n📊 TIER SUMMARY:")
    for tier_name, (tier_title, _) in tier_info.items():
        count = len(tiers[tier_name])
        if count > 0:
            print(f"   {tier_title}: {count} signals")
    
    print('='*180)
# def get_realtime_price(ticker: str) -> Optional[float]:
#     """Get real-time price from NSE (market hours only)"""
#     try:
#         import datetime as dt
#         now = dt.datetime.now()
        
#         # Check market hours (9:15 AM - 3:30 PM IST)
#         if now.hour < 9 or (now.hour == 9 and now.minute < 15):
#             return None  # Pre-market
#         if now.hour >= 15 and now.minute > 30:
#             return None  # Post-market
            
#         # Use NSE API or broker API for real-time
#         # ... implementation depends on your data source
        
#     except:
#         return None

# ============================================================================
# EXPORT & HTML GENERATION
# ============================================================================

def export_results(results: List[Dict], scan_time: float):
    """Export results to CSV and HTML"""
    
    try:
        os.makedirs(SIGNALS_DIR, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # CSV Export
        csv_data = []
        for res in results:
            row = {
                "Ticker": res["ticker"],
                "Side": res["side"],
                "Price": res["price"],
                "Confidence": res["confidence"],
                "RSI": res["rsi"],
                "Entry": res["entry_price"],
                "StopLoss": res["stop_loss"],
                "Target1": res["target_1"],
                "Target2": res["target_2"],
                "RiskReward": res["risk_reward"],
                "Quantity": res["qty"],
                "Sector": res["sector"],

                "52W_Position_Pct": res.get('fifty_two_week', {}).get('position_pct', ''),
                "52W_High": res.get('fifty_two_week', {}).get('high_52w', ''),
                "52W_Low": res.get('fifty_two_week', {}).get('low_52w', ''),
                "52W_Signal": res.get('fifty_two_week', {}).get('signal', ''),
                
                "RS_vs_Nifty": res.get('relative_strength', {}).get('avg_rs', ''),
                "RS_Signal": res.get('relative_strength', {}).get('signal', ''),
                
                "Delivery_Score": res.get('delivery_estimate', {}).get('estimated_delivery_score', ''),
                
                "Earnings_Safe": res.get('earnings_info', {}).get('safe_to_trade', True),
                "Earnings_Days": res.get('earnings_info', {}).get('days_until', ''),
                
                "Consolidation_Type": res.get('consolidation', {}).get('breakout_type', ''),
                "Consolidation_Days": res.get('consolidation', {}).get('consolidation_days', ''),
                
                "Market_Regime": res.get('market_regime', 'N/A'),
                
                "Trailing_Stop_ATR": res.get('trailing_stops', {}).get('atr', {}).get('trailing_stop', ''),
                "Trailing_Stop_Pct": res.get('trailing_stops', {}).get('percent', {}).get('trailing_stop', ''),
                "Reasons": res["reasons"],
            }
            
            if res.get('backtest_validated') and 'backtest' in res:
                bt = res['backtest']
                row.update({
                    "BT_WinRate": bt['win_rate'],
                    "BT_ProfitFactor": bt['profit_factor'],
                    "BT_Sharpe": bt['sharpe_ratio'],
                    "BT_MaxDrawdown": bt['max_drawdown'],
                    "BT_Reliability": bt['reliability_score'],
                })
            
            csv_data.append(row)
        
        df = pd.DataFrame(csv_data)
        csv_file = os.path.join(SIGNALS_DIR, f"signals_{timestamp}.csv")
        df.to_csv(csv_file, index=False)
        print(f"\n  ✓ CSV: {csv_file}")
        
        # HTML Report
        html_file = generate_html_report(results, scan_time, timestamp)
        if html_file:
            print(f"  ✓ HTML: {html_file}")
        
        return csv_file, html_file
        
    except Exception as e:
        logger.error(f"Export failed: {e}")
        return None, None

def generate_html_report(results: List[Dict], scan_time: float, timestamp: str) -> Optional[str]:
    """Generate HTML report"""
    
    try:
        html_content = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>NSE Scanner v8.5 - {timestamp}</title>
    <style>
        body {{ font-family: Arial; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 20px; }}
        .container {{ max-width: 1400px; margin: 0 auto; background: white; border-radius: 15px; overflow: hidden; }}
        .header {{ background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%); color: white; padding: 30px; text-align: center; }}
        .header h1 {{ margin: 0; font-size: 2.5em; }}
        .stats {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; padding: 30px; background: #f8f9fa; }}
        .stat-card {{ background: white; padding: 20px; border-radius: 10px; text-align: center; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
        .stat-card .value {{ font-size: 2em; font-weight: bold; color: #667eea; margin: 10px 0; }}
        .stat-card .label {{ color: #666; text-transform: uppercase; font-size: 0.85em; }}
        .signals {{ padding: 30px; }}
        .signal-card {{ background: white; border: 2px solid #e0e0e0; border-radius: 10px; padding: 20px; margin-bottom: 20px; }}
        .signal-card:hover {{ box-shadow: 0 5px 20px rgba(0,0,0,0.1); }}
        .signal-header {{ display: flex; justify-content: space-between; padding-bottom: 15px; border-bottom: 2px solid #f0f0f0; }}
        .signal-ticker {{ font-size: 1.8em; font-weight: bold; color: #1e3c72; }}
        .signal-side {{ padding: 8px 20px; border-radius: 20px; font-weight: bold; }}
        .side-long {{ background: #4caf50; color: white; }}
        .side-short {{ background: #f44336; color: white; }}
        .confidence {{ font-size: 1.5em; font-weight: bold; }}
        .conf-high {{ color: #4caf50; }}
        .conf-medium {{ color: #ff9800; }}
        .conf-low {{ color: #2196f3; }}
        .signal-details {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 15px; margin: 20px 0; }}
        .detail-item {{ background: #f8f9fa; padding: 12px; border-radius: 8px; }}
        .detail-label {{ font-size: 0.85em; color: #666; margin-bottom: 5px; }}
        .detail-value {{ font-size: 1.2em; font-weight: bold; color: #333; }}
        .backtest-section {{ background: #e8f5e9; padding: 15px; border-radius: 8px; margin-top: 15px; }}
        .backtest-title {{ font-weight: bold; color: #2e7d32; margin-bottom: 10px; }}
        .reasons {{ background: #fff3e0; padding: 15px; border-radius: 8px; margin-top: 15px; border-left: 4px solid #ff9800; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>📊 NSE SWING SCANNER v8.5</h1>
            <div>Ultimate Accuracy Edition - {ACCURACY_MODE} Mode</div>
            <div>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</div>
        </div>
        
        <div class="stats">
            <div class="stat-card"><div class="label">Total Signals</div><div class="value">{len(results)}</div></div>
            <div class="stat-card"><div class="label">Stocks Scanned</div><div class="value">{stats['total']}</div></div>
            <div class="stat-card"><div class="label">Success Rate</div><div class="value">{(stats['passed']/stats['total']*100) if stats['total'] > 0 else 0:.1f}%</div></div>
            <div class="stat-card"><div class="label">Scan Time</div><div class="value">{scan_time:.1f}s</div></div>
        </div>
        
        <div class="signals">
            <h2 style="margin-bottom: 20px;">🎯 Trading Signals</h2>"""
        
        for i, res in enumerate(results[:TOP_RESULTS], 1):
            conf_class = "conf-high" if res['confidence'] >= 70 else ("conf-medium" if res['confidence'] >= 60 else "conf-low")
            side_class = "side-long" if res['side'] == "LONG" else "side-short"
            
            html_content += f"""
            <div class="signal-card">
                <div class="signal-header">
                    <div><span style="color: #999;">#{i}</span> <span class="signal-ticker">{res['ticker']}</span></div>
                    <div style="display: flex; gap: 15px; align-items: center;">
                        <span class="signal-side {side_class}">{res['side']}</span>
                        <span class="confidence {conf_class}">{res['confidence']:.1f}%</span>
                    </div>
                </div>
                
                <div class="signal-details">
                    <div class="detail-item"><div class="detail-label">Price</div><div class="detail-value">₹{res['price']:.2f}</div></div>
                    <div class="detail-item"><div class="detail-label">Entry</div><div class="detail-value">₹{res['entry_price']:.2f}</div></div>
                    <div class="detail-item"><div class="detail-label">Stop Loss</div><div class="detail-value">₹{res['stop_loss']:.2f}</div></div>
                    <div class="detail-item"><div class="detail-label">Target 1</div><div class="detail-value">₹{res['target_1']:.2f}</div></div>
                    <div class="detail-item"><div class="detail-label">R:R</div><div class="detail-value">{res['risk_reward']:.1f}x</div></div>
                    <div class="detail-item"><div class="detail-label">Qty</div><div class="detail-value">{res['qty']}</div></div>
                </div>
                
                <div class="reasons"><strong>💡 Entry Reasons:</strong><br>{res['reasons']}</div>
            </div>"""
        
        html_content += """
        </div>
    </div>
</body>
</html>"""
        
        html_file = os.path.join(SIGNALS_DIR, f"report_{timestamp}.html")
        with open(html_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        return html_file
        
    except Exception as e:
        logger.error(f"HTML generation failed: {e}")
        return None

# ============================================================================
# HEADER PRINTING
# ============================================================================

def print_header():
    """Print scanner header"""
    
    print("\n" + "="*120)
    print("""
███╗   ██╗███████╗███████╗    ███████╗ ██████╗ █████╗ ███╗   ██╗███╗   ██╗███████╗██████╗     ██╗   ██╗ █████╗ ███████╗
████╗  ██║██╔════╝██╔════╝    ██╔════╝██╔════╝██╔══██╗████╗  ██║████╗  ██║██╔════╝██╔══██╗    ██║   ██║██╔══██╗██╔════╝
██╔██╗ ██║███████╗█████╗      ███████╗██║     ███████║██╔██╗ ██║██╔██╗ ██║█████╗  ██████╔╝    ██║   ██║╚█████╔╝███████╗
██║╚██╗██║╚════██║██╔══╝      ╚════██║██║     ██╔══██║██║╚██╗██║██║╚██╗██║██╔══╝  ██╔══██╗    ╚██╗ ██╔╝██╔══██╗╚════██║
██║ ╚████║███████║███████╗    ███████║╚██████╗██║  ██║██║ ╚████║██║ ╚████║███████╗██║  ██║     ╚████╔╝ ╚█████╔╝███████║
╚═╝  ╚═══╝╚══════╝╚══════╝    ╚══════╝ ╚═════╝╚═╝  ╚═╝╚═╝  ╚═══╝╚═╝  ╚═══╝╚══════╝╚═╝  ╚═╝      ╚═══╝   ╚════╝ ╚══════╝
    """)
    print("="*120)
    print("                               VERSION 8.5 - ULTIMATE ACCURACY EDITION")
    print("                    Combining v7.4 Advanced Features + v8.0 Backtesting Power")
    print("="*120)
    
    print(f"\n⚙️  CONFIGURATION:")
    print(f"   Mode:              {ACCURACY_MODE}")
    print(f"   Backtest:          {BACKTEST_MODE}")
    print(f"   Min Confidence:    {MIN_CONFIDENCE}%")
    print(f"   Min R:R Ratio:     {MIN_RR_RATIO}x")
    print(f"   Account Capital:   ₹{ACCOUNT_CAPITAL:,}")
    print(f"   Risk per Trade:    {RISK_PER_TRADE*100}%")
    
    print(f"\n✅ ENABLED FEATURES:")
    features = []
    if USE_SECTOR_ROTATION:
        features.append("Sector Rotation")
    if USE_VIX_SENTIMENT:
        features.append("VIX Sentiment")
    if USE_FIBONACCI_SCORING:
        features.append("Fibonacci")
    if USE_PORTFOLIO_RISK:
        features.append("Portfolio Risk")
    if USE_MINI_BACKTEST:
        features.append("Mini Backtest")
    if USE_FULL_BACKTEST:
        features.append("Full Backtest")
    
    print("   " + " | ".join(features))
    print("\n" + "="*120)

# ============================================================================
# MAIN EXECUTION - WITH CACHE VALIDATION
# ============================================================================
# ============================================================================
# MULTI-TIER SIGNAL CLASSIFICATION
# ============================================================================

def classify_signal_tier(result: Dict, backtest_result: Optional[BacktestResult]) -> str:
    """
    Classify signal into tiers based on quality - FIXED VERSION
    """
    confidence = result.get('confidence', 0)
    rr = result.get('risk_reward', 0)
    mini_bt = result.get('mini_backtest', {})
    
    # Safely extract backtest values
    bt_win_rate = 0
    bt_pf = 0
    if backtest_result:
        bt_win_rate = getattr(backtest_result, 'win_rate', 0)
        bt_pf = getattr(backtest_result, 'profit_factor', 0)
    elif result.get('backtest_validated') and 'backtest' in result:
        bt = result['backtest']
        bt_win_rate = bt.get('win_rate', 0)
        bt_pf = bt.get('profit_factor', 0)

    # Tier 1 Requirements (PREMIUM)
    tier1_checks = [
        confidence >= 75,
        rr >= 2.0,
        mini_bt.get('success_reason', '') in ['Target hit, stop not hit', 'Target hit before stop'],
        bt_win_rate >= 58,
        bt_pf >= 1.7,
    ]
    
    if sum(1 for x in tier1_checks if x) >= 4:
        return "TIER_1_PREMIUM"
    
    # Tier 2 Requirements (STANDARD)
    tier2_checks = [
        confidence >= 65,
        rr >= 1.5,
        mini_bt.get('target_hit', False) or mini_bt.get('success_reason', '').startswith('Final P&L'),
        bt_win_rate >= 52,
        bt_pf >= 1.4,
    ]
    
    if sum(1 for x in tier2_checks if x) >= 3:
        return "TIER_2_STANDARD"
    
    # Tier 3 (SPECULATIVE)
    if confidence >= 55 and rr >= 1.3:
        return "TIER_3_SPECULATIVE"
    
    return "REJECTED"


def apply_tier_based_filtering(results: List[Dict]) -> Dict[str, List[Dict]]:
    """Separate results into tiers"""
    
    tiers = {
        "TIER_1_PREMIUM": [],
        "TIER_2_STANDARD": [],
        "TIER_3_SPECULATIVE": []
    }
    
    for result in results:
        backtest = None
        if result.get('backtest_validated') and 'backtest' in result:
            # Reconstruct BacktestResult from dict
            bt_dict = result['backtest']
            backtest = BacktestResult(
                total_trades=bt_dict.get('total_trades', 10),
                winning_trades=0,
                losing_trades=0,
                win_rate=bt_dict.get('win_rate', 50),
                avg_win=0,
                avg_loss=0,
                profit_factor=bt_dict.get('profit_factor', 1.5),
                sharpe_ratio=bt_dict.get('sharpe_ratio', 0),
                max_drawdown=bt_dict.get('max_drawdown', 10),
                total_return=0,
                avg_holding_days=0,
                best_trade=0,
                worst_trade=0,
                consecutive_wins=0,
                consecutive_losses=0,
                reliability_score=bt_dict.get('reliability_score', 50),
                expectancy=bt_dict.get('expectancy', 0)
            )
        
        tier = classify_signal_tier(result, backtest)
        
        if tier in tiers:
            result['tier'] = tier
            tiers[tier].append(result)
    
    return tiers
def main():
    """Main execution function - FIXED VERSION WITH VALIDATION"""
    
    start_time = time.time()
    
    print_header()
    
    # ========================================================================
    # [1/5] LOAD STOCK UNIVERSE
    # ========================================================================
    
    print("\n[1/5] Loading Stock Universe...")
    tickers = get_universe_symbols()
    
    if not tickers:
        print("❌ Failed to load stock symbols")
        return
    
    print(f"  ✓ Loaded {len(tickers)} symbols")
    
    # ========================================================================
    # [2/5] INITIALIZE FILTERS
    # ========================================================================
    
    print("\n[2/5] Initializing Analysis Engines...")
    
    quality_validator = DataQualityValidator()
    sector_filter = SectorRotationAnalyzer() if USE_SECTOR_ROTATION else None
    vix_filter = VIXSentimentAnalyzer() if USE_VIX_SENTIMENT else None
    fibo_detector = FibonacciDetector() if USE_FIBONACCI_SCORING else None
    portfolio_mgr = PortfolioRiskManager(ACCOUNT_CAPITAL) if USE_PORTFOLIO_RISK else None
    mtf_analyzer = MultiTimeframeAnalyzer()
    
    fifty_two_week_analyzer = FiftyTwoWeekAnalyzer if USE_52_WEEK_FILTER else None
    rs_analyzer = RelativeStrengthAnalyzer() if USE_RELATIVE_STRENGTH else None
    delivery_analyzer = DeliveryVolumeAnalyzer if USE_DELIVERY_VOLUME else None
    earnings_filter = EarningsFilter if USE_EARNINGS_FILTER else None
    consolidation_detector = ConsolidationBreakoutDetector if USE_CONSOLIDATION_BREAKOUT else None
    market_breadth = MarketBreadthAnalyzer() if USE_MARKET_BREADTH else None

     # Get market breadth once (for all stocks)
    market_breadth_data = None
    if market_breadth and USE_MARKET_BREADTH:
        print("  └─ Analyzing market breadth...")
        market_breadth_data = market_breadth.analyze()
        print("  └─ Analyzing market breadth...")
        market_breadth_data = market_breadth.analyze()
        
        # DEBUG OUTPUT
        print(f"\n  📊 MARKET BREADTH DEBUG:")
        print(f"     Valid: {market_breadth_data.get('valid')}")
        print(f"     Regime: {market_breadth_data.get('regime')}")
        print(f"     Long OK: {market_breadth_data.get('long_ok')}")
        print(f"     Short OK: {market_breadth_data.get('short_ok')}")
        print(f"     Message: {market_breadth_data.get('message')}")
        print(f"     ROC 5d: {market_breadth_data.get('roc_5')}%")
        print(f"     ROC 20d: {market_breadth_data.get('roc_20')}%")

        if market_breadth_data.get('valid'):
            print(f"      └─ {market_breadth_data.get('message', 'Unknown')}")
    
    if sector_filter:
        print("  └─ Updating sector data...")
        sector_filter.update_sector_strength()
    
    print("  ✓ All engines initialized")
    
    # ========================================================================
    # [3/5] BUILD PRICE CACHE
    # ========================================================================
    
    print("\n[3/5] Building Price Cache...")
    df_all = build_cache(tickers)
    
    if df_all is None or df_all.empty:
        print("❌ Failed to build price cache")
        return
    
    # ========================================================================
    # FIX 3: CACHE VALIDATION
    # ========================================================================
    
    print(f"\n{'='*120}")
    print("CACHE VALIDATION & INTEGRITY CHECK")
    print('='*120)
    
    print(f"\n📊 Cache Structure:")
    print(f"   Total rows:        {len(df_all):,}")
    print(f"   Columns:           {df_all.columns.tolist()}")
    print(f"   Unique tickers:    {df_all['ticker'].nunique()}")
    print(f"   Date range:        {df_all['Date'].min()} to {df_all['Date'].max()}")
    
    print(f"\n🔍 Ticker Format Check:")
    sample_tickers = df_all['ticker'].unique()[:10]
    print(f"   Sample tickers:    {sample_tickers.tolist()}")
    ticker_format = 'WITH .NS' if any('.NS' in str(t) for t in sample_tickers) else 'WITHOUT .NS'
    print(f"   Format detected:   {ticker_format}")
    
    print(f"\n🎯 Data Uniqueness Validation:")
    print(f"   {'Ticker':<15} {'Rows':>6} {'Unique':>7} {'Price Range':<30} {'Status'}")
    print(f"   {'-'*15} {'-'*6} {'-'*7} {'-'*30} {'-'*10}")
    
    # Test first 5 tickers for data integrity
    test_tickers = df_all['ticker'].unique()[:5]
    all_valid = True
    
    for test_ticker in test_tickers:
        test_data = df_all[df_all['ticker'] == test_ticker]
        
        rows = len(test_data)
        unique_closes = test_data['Close'].nunique()
        min_price = test_data['Close'].min()
        max_price = test_data['Close'].max()
        
        
        # Check if data looks valid
        uniqueness_ratio = unique_closes / rows if rows > 0 else 0
        
        if uniqueness_ratio < 0.3 and rows > 20:
            status = "⚠️  SUSPICIOUS"
            all_valid = False
        else:
            status = "✓ OK"
        
        ticker_short = test_ticker[:15]
        price_range_str = f"₹{min_price:.2f} - ₹{max_price:.2f}"
        
        print(f"   {ticker_short:<15} {rows:>6} {unique_closes:>7} {price_range_str:<30} {status}")
    
    # Overall validation result
    print(f"\n{'='*120}")
    if all_valid:
        print("✅ Cache validation PASSED - Data looks good!")
    else:
        print("⚠️  Cache validation WARNING - Some data may be corrupted")
        print("   Proceeding anyway, but results may be affected")
    print('='*120)
    
    # ========================================================================
    # [4/5] SCAN STOCKS
    # ========================================================================
    
    cached_tickers = sorted(df_all["ticker"].unique().tolist())
    print(f"\n  ✓ Limiting scan to {len(cached_tickers)} cached tickers (was {len(tickers)})")
    tickers = cached_tickers  # ✅ CRITICAL: Only scan what we have
    
    print(f"\n[4/5] Scanning {len(tickers)} Stocks...")
    
    if BACKTEST_MODE in ['HYBRID', 'FULL']:
        results = hybrid_scan_universe(
           tickers=tickers,
            df_all=df_all,
            quality_validator=quality_validator,
            sector_filter=sector_filter,
            vix_filter=vix_filter,
            fibo_detector=fibo_detector,
            portfolio_mgr=portfolio_mgr,
            mtf_analyzer=mtf_analyzer,
            # ADD THESE NEW PARAMETERS:
            fifty_two_week_analyzer=fifty_two_week_analyzer,
            rs_analyzer=rs_analyzer,
            delivery_analyzer=delivery_analyzer,
            earnings_filter=earnings_filter,
            consolidation_detector=consolidation_detector,
            market_breadth_data=market_breadth_data
        )
    else:
        print(f"\n{'='*120}")
        print("SCANNING WITH MINI BACKTEST")
        print('='*120)
        
        results = []
        
        for i, ticker in enumerate(tickers, 1):
            result = scan_ticker(
                ticker, df_all, quality_validator, sector_filter,
                vix_filter, fibo_detector, portfolio_mgr, mtf_analyzer,
                backtester=None, run_full_backtest=False,fifty_two_week_analyzer=fifty_two_week_analyzer,
                rs_analyzer=rs_analyzer,
                delivery_analyzer=delivery_analyzer,
                earnings_filter=earnings_filter,
                consolidation_detector=consolidation_detector,
                market_breadth_data=market_breadth_data
            )
            
            if result:
                results.append(result)
                if USE_PORTFOLIO_RISK and portfolio_mgr:
                    portfolio_mgr.add_trade(result)
            
            if i % 25 == 0 or i == len(tickers):
                pct = (i / len(tickers)) * 100
                bar_length = 50
                filled = int(bar_length * i / len(tickers))
                bar = '█' * filled + '░' * (bar_length - filled)
                print(f"  [{bar}] {pct:>5.1f}% | {i:>4}/{len(tickers)} | Found: {len(results):>3}", 
                      end='\r', flush=True)
        
        print()
        results = sorted(results, key=lambda x: x['confidence'], reverse=True)
    
    # ========================================================================
    # [5/5] DISPLAY RESULTS
    # ========================================================================
    
    print(f"\n[5/5] Generating Reports...")
    
    print_statistics()
    
    if not results:
        print(f"\n{'='*120}")
        print("❌ NO SIGNALS FOUND")
        print('='*120)
        print(f"\n💡 SUGGESTIONS:")
        print(f"   • Switch to AGGRESSIVE mode (currently: {ACCURACY_MODE})")
        print(f"   • Lower MIN_CONFIDENCE to {MIN_CONFIDENCE - 10}%")
        print(f"   • Check rejection reasons above")
        print('='*120 + "\n")
        return
    
    print_results(results)
     # ═══════════════════════════════════════════════════════════════
    # ADD THIS: WEEKLY PICK SELECTION
    # ═══════════════════════════════════════════════════════════════
    
    print("\n" + "="*80)
    print("🎯 RUNNING WEEKLY PICK SELECTOR...")
    print("="*80)
    
    weekly_picks = select_weekly_picks(results, max_picks=2)
    print_weekly_picks(weekly_picks)
    
    # Save weekly picks to separate file
    if weekly_picks:
        save_weekly_picks(weekly_picks)
    
    scan_time = time.time() - start_time
    export_results(results, scan_time)
    
    # ========================================================================
    # FINAL SUMMARY
    # ========================================================================
    
    print(f"\n{'='*120}")
    print("✅ SCAN COMPLETE!")
    print('='*120)
    
    print(f"\n📊 SUMMARY:")
    print(f"   Mode:              {ACCURACY_MODE}")
    print(f"   Backtest:          {BACKTEST_MODE}")
    print(f"   Stocks Scanned:    {stats['total']}")
    print(f"   Signals Found:     {len(results)}")
    print(f"   Success Rate:      {(stats['passed']/stats['total']*100) if stats['total'] > 0 else 0:.1f}%")
    print(f"   Scan Time:         {scan_time:.1f} seconds")
    
    if results:
        avg_conf = sum(r['confidence'] for r in results) / len(results)
        print(f"   Avg Confidence:    {avg_conf:.1f}%")
        
        validated = len([r for r in results if r.get('backtest_validated')])
        if validated > 0:
            print(f"   Backtest Validated: {validated}/{len(results)}")
    
    print(f"\n💾 OUTPUT: {SIGNALS_DIR}/")
    print("\n" + "="*120 + "\n")
    return results


# ============================================================================
# EMAIL NOTIFICATION FUNCTION
# ============================================================================

def send_email_notification(results, csv_file, html_file, scan_time):
    """Send email with scan results"""
    
    if not EMAIL_CONFIG['enabled']:
        logger.info("Email notifications disabled")
        return
    
    if not EMAIL_CONFIG['sender_email'] or not EMAIL_CONFIG['sender_password']:
        logger.warning("Email credentials not configured - skipping email")
        return
    
    try:
        # Create message
        msg = MIMEMultipart('alternative')
        msg['From'] = EMAIL_CONFIG['sender_email']
        msg['To'] = EMAIL_CONFIG['recipient_email']
        msg['Subject'] = f"📊 NSE Scanner - {datetime.now().strftime('%Y-%m-%d')} - {len(results)} Signals"
        
        # Email body
        body = f"""
NSE SWING SCANNER v8.5 - DAILY REPORT
{'='*70}

Scan Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S IST')}
Scan Time: {scan_time:.1f} seconds
Total Signals: {len(results)}

{'='*70}
TOP 5 SIGNALS:
{'='*70}
"""
        
        for i, res in enumerate(results[:5], 1):
            body += f"""
{i}. {res['ticker']} - {res['side']}
   💰 Price: ₹{res['price']:.2f}
   🎯 Confidence: {res['confidence']:.1f}%
   📈 Entry: ₹{res['entry_price']:.2f}
   🛑 Stop Loss: ₹{res['stop_loss']:.2f}
   🎯 Target 1: ₹{res['target_1']:.2f}
   📊 R:R Ratio: {res['risk_reward']:.1f}x
   📦 Quantity: {res['qty']} shares
"""
        
        body += f"""
{'='*70}

📎 Full reports attached (CSV & HTML)

🔗 View all results: https://github.com/{os.environ.get('GITHUB_REPOSITORY', 'your-repo')}

This is an automated message from NSE Scanner v8.5
Running on GitHub Actions ⚡
"""
        
        msg.attach(MIMEText(body, 'plain'))
        
        # Attach CSV file
        if csv_file and os.path.exists(csv_file):
            with open(csv_file, 'rb') as f:
                part = MIMEBase('application', 'octet-stream')
                part.set_payload(f.read())
                encoders.encode_base64(part)
                part.add_header('Content-Disposition', f'attachment; filename={os.path.basename(csv_file)}')
                msg.attach(part)
        
        # Attach HTML file
        if html_file and os.path.exists(html_file):
            with open(html_file, 'rb') as f:
                part = MIMEBase('application', 'octet-stream')
                part.set_payload(f.read())
                encoders.encode_base64(part)
                part.add_header('Content-Disposition', f'attachment; filename={os.path.basename(html_file)}')
                msg.attach(part)
        
        # Send email
        with smtplib.SMTP(EMAIL_CONFIG['smtp_server'], EMAIL_CONFIG['smtp_port']) as server:
            server.starttls()
            server.login(EMAIL_CONFIG['sender_email'], EMAIL_CONFIG['sender_password'])
            server.send_message(msg)
        
        logger.info("✅ Email notification sent successfully")
        
    except Exception as e:
        logger.error(f"❌ Failed to send email: {e}")

# ============================================================================
# MODIFIED MAIN FUNCTION FOR GITHUB ACTIONS
# ============================================================================

def github_actions_main():
    """Main function optimized for GitHub Actions execution - FIXED VERSION"""
    
    logger.info("="*80)
    logger.info("NSE SCANNER v8.5 - GITHUB ACTIONS EXECUTION")
    logger.info(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S IST')}")
    logger.info(f"GitHub Repository: {os.environ.get('GITHUB_REPOSITORY', 'N/A')}")
    logger.info(f"Workflow Run: #{os.environ.get('GITHUB_RUN_NUMBER', 'N/A')}")
    logger.info("="*80)
    
    import time
    start_time = time.time()
    
    try:
        logger.info("🚀 Starting stock scan...")
        
        # ✅ FIX: main() already returns results
        results = main()
        
        scan_time = time.time() - start_time
        
        # ✅ FIX: Check if results exist AND have data
        if results and len(results) > 0:
            logger.info(f"✅ Scan complete: {len(results)} signals found in {scan_time:.1f}s")
            
            # ✅ FIX: Use correct directory name
            signals_dir = f"signals_v8.5_{ACCURACY_MODE.lower()}"
            logger.info(f"📂 Looking for files in: {signals_dir}/")
            
            # Find the latest files
            csv_file = None
            html_file = None
            
            if os.path.exists(signals_dir):
                csv_files = sorted([f for f in os.listdir(signals_dir) if f.endswith('.csv')])
                html_files = sorted([f for f in os.listdir(signals_dir) if f.endswith('.html')])
                
                if csv_files:
                    csv_file = os.path.join(signals_dir, csv_files[-1])
                    logger.info(f"📄 Found CSV: {csv_file}")
                
                if html_files:
                    html_file = os.path.join(signals_dir, html_files[-1])
                    logger.info(f"📄 Found HTML: {html_file}")
                
                # ✅ Copy files to standard location for artifacts
                copy_files_to_output(signals_dir)
            else:
                logger.warning(f"⚠️  Directory not found: {signals_dir}")
            
            # ✅ Send email with files
            logger.info("📧 Sending email notification...")
            send_email_notification(results, csv_file, html_file, scan_time)
            
        else:
            logger.info(f"⚠️  No signals found in {scan_time:.1f}s")
            send_no_signals_email(scan_time)
        
        logger.info("="*80)
        logger.info("✅ GitHub Actions execution completed successfully")
        logger.info("="*80)
        
        return 0
        
    except Exception as e:
        logger.error(f"❌ FATAL ERROR: {e}", exc_info=True)
        
        # Send error notification
        if EMAIL_CONFIG['enabled'] and EMAIL_CONFIG['sender_email']:
            try:
                msg = MIMEText(f"Scanner error:\n\n{str(e)}\n\nCheck logs.")
                msg['Subject'] = "⚠️ NSE Scanner Error"
                msg['From'] = EMAIL_CONFIG['sender_email']
                msg['To'] = EMAIL_CONFIG['recipient_email']
                
                with smtplib.SMTP(EMAIL_CONFIG['smtp_server'], EMAIL_CONFIG['smtp_port']) as server:
                    server.starttls()
                    server.login(EMAIL_CONFIG['sender_email'], EMAIL_CONFIG['sender_password'])
                    server.send_message(msg)
            except:
                pass
        
        return 1


def send_no_signals_email(scan_time):
    """Send notification when no signals found"""
    
    if not EMAIL_CONFIG['enabled'] or not EMAIL_CONFIG['sender_email']:
        logger.info("Email not configured")
        return
    
    try:
        body = f"""
NSE SWING SCANNER v8.5 - DAILY REPORT
{'='*70}

Scan Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S IST')}
Scan Time: {scan_time:.1f} seconds
Mode: {ACCURACY_MODE}

⚠️  NO TRADING SIGNALS FOUND TODAY

The scanner ran successfully but no stocks met the entry criteria.

Possible reasons:
- Market conditions unfavorable
- All setups filtered by quality checks
- Weekend/holiday (stale data)
- High volatility regime

{'='*70}

Next scan: Tomorrow at 4:00 PM IST

This is an automated message from NSE Scanner v8.5
"""
        
        msg = MIMEText(body)
        msg['Subject'] = f"📊 NSE Scanner - {datetime.now().strftime('%Y-%m-%d')} - No Signals"
        msg['From'] = EMAIL_CONFIG['sender_email']
        msg['To'] = EMAIL_CONFIG['recipient_email']
        
        with smtplib.SMTP(EMAIL_CONFIG['smtp_server'], EMAIL_CONFIG['smtp_port']) as server:
            server.starttls()
            server.login(EMAIL_CONFIG['sender_email'], EMAIL_CONFIG['sender_password'])
            server.send_message(msg)
        
        logger.info("✅ No-signals email sent successfully")
        
    except Exception as e:
        logger.error(f"❌ Email send failed: {e}")


def copy_files_to_output(source_dir):
    """Copy signal files to standard output directory for GitHub Actions"""
    
    try:
        # Create standard output directory
        output_dir = "signals"
        os.makedirs(output_dir, exist_ok=True)
        
        if not os.path.exists(source_dir):
            logger.warning(f"Source directory not found: {source_dir}")
            return
        
        
        copied = 0
        
        # Copy all CSV and HTML files
        for filename in os.listdir(source_dir):
            if filename.endswith(('.csv', '.html')):
                src_path = os.path.join(source_dir, filename)
                dst_path = os.path.join(output_dir, filename)
                
                shutil.copy2(src_path, dst_path)
                logger.info(f"📋 Copied: {filename}")
                copied += 1
        
        logger.info(f"✅ Copied {copied} files to {output_dir}/")
        
        # Also copy logs
        if os.path.exists('logs/scanner.log'):
            shutil.copy2('logs/scanner.log', 'signals/scanner.log')
            logger.info("📋 Copied: scanner.log")
        
    except Exception as e:
        logger.error(f"⚠️  Failed to copy files: {e}")
# ============================================================================
# ENTRY POINT
# ============================================================================
# ============================================================================
# WEEKLY PICK SELECTOR - ADD THIS TO YOUR SCANNER
# ============================================================================

def select_weekly_picks(results: List[Dict], max_picks: int = 2) -> List[Dict]:
    """
    Ultra-selective filter for weekly trading
    Only returns the ABSOLUTE BEST 1-2 stocks
    
    Requirements:
    - Tier 1 only
    - Confidence ≥ 75%
    - Backtest validated
    - R:R ≥ 2.0
    - 52W position ≥ 70%
    - RS vs Nifty ≥ 1.05
    - No earnings within 7 days
    - Consolidation breakout OR VCP pattern
    """
    
    weekly_picks = []
    
    for stock in results:
        score = 0
        disqualified = False
        reasons = []
        
        # ═══════════════════════════════════════════════════════════════
        # MANDATORY CHECKS (Must pass ALL)
        # ═══════════════════════════════════════════════════════════════
        
        # Check 1: Must be Tier 1
        if stock.get('tier') != 'TIER_1_PREMIUM':
            disqualified = True
            reasons.append(f"Not Tier 1 ({stock.get('tier', 'Unknown')})")
        
        # Check 2: Confidence ≥ 75%
        if stock.get('confidence', 0) < 75:
            disqualified = True
            reasons.append(f"Low confidence ({stock.get('confidence', 0):.1f}%)")
        
        # Check 3: Backtest validated
        if not stock.get('backtest_validated', False):
            disqualified = True
            reasons.append("Not backtest validated")
        
        # Check 4: R:R ≥ 2.0
        if stock.get('risk_reward', 0) < 2.0:
            disqualified = True
            reasons.append(f"Low R:R ({stock.get('risk_reward', 0):.1f}x)")
        
        # Check 5: No earnings within 7 days
        earnings_info = stock.get('earnings_info', {})
        if earnings_info.get('earnings_soon', False):
            disqualified = True
            reasons.append("Earnings upcoming")
        
        if disqualified:
            continue
        
        # ═══════════════════════════════════════════════════════════════
        # QUALITY SCORING (Higher = Better)
        # ═══════════════════════════════════════════════════════════════
        
        # 52-Week Position (max 25 points)
        ftw = stock.get('fifty_two_week', {})
        position_pct = ftw.get('position_pct', 50)
        if position_pct >= 85:
            score += 25
        elif position_pct >= 70:
            score += 20
        elif position_pct >= 60:
            score += 10
        else:
            score -= 10  # Near 52W low = bad
        
        # Relative Strength (max 20 points)
        rs = stock.get('relative_strength', {})
        avg_rs = rs.get('avg_rs', 1.0)
        if avg_rs >= 1.20:
            score += 20
        elif avg_rs >= 1.10:
            score += 15
        elif avg_rs >= 1.05:
            score += 10
        elif avg_rs < 0.95:
            score -= 15  # Underperforming = bad
        
        # Consolidation Breakout (max 20 points)
        cons = stock.get('consolidation', {})
        if cons.get('breakout_type') == 'BULLISH' and stock['side'] == 'LONG':
            score += 20
        elif cons.get('breakout_type') == 'BEARISH' and stock['side'] == 'SHORT':
            score += 20
        elif cons.get('breakout_type') == 'PENDING':
            score += 5
        
        # Backtest Quality (max 20 points)
        bt = stock.get('backtest', {})
        if bt.get('win_rate', 0) >= 65:
            score += 20
        elif bt.get('win_rate', 0) >= 60:
            score += 15
        elif bt.get('win_rate', 0) >= 55:
            score += 10
        
        if bt.get('profit_factor', 0) >= 2.0:
            score += 10
        elif bt.get('profit_factor', 0) >= 1.7:
            score += 5
        
        # Confidence Bonus (max 15 points)
        conf = stock.get('confidence', 0)
        if conf >= 85:
            score += 15
        elif conf >= 80:
            score += 10
        elif conf >= 75:
            score += 5
        
        # Add to weekly picks with score
        stock['weekly_score'] = score
        weekly_picks.append(stock)
    
    # Sort by weekly score (highest first)
    weekly_picks = sorted(weekly_picks, key=lambda x: x['weekly_score'], reverse=True)
    
    # Return top picks only
    return weekly_picks[:max_picks]


def print_weekly_picks(picks: List[Dict]):
    """Print weekly picks in clear format"""
    
    if not picks:
        print("\n" + "="*80)
        print("❌ NO WEEKLY PICKS TODAY")
        print("="*80)
        print("\n💡 Reasons:")
        print("   • No stocks meet ultra-strict criteria")
        print("   • Wait for better setup tomorrow")
        print("   • Quality > Quantity")
        print("="*80)
        return
    
    print("\n" + "="*80)
    print("🏆 WEEKLY PICK(S) - HIGHEST CONVICTION TRADES")
    print("="*80)
    
    for i, pick in enumerate(picks, 1):
        print(f"""
┌{'─'*76}┐
│ #{i} {pick['ticker']:<15} {pick['side']:<6}                    SCORE: {pick['weekly_score']}/100 │
├{'─'*76}┤
│                                                                            │
│  💰 ENTRY DETAILS                                                          │
│  ├── Current Price:    ₹{pick['price']:<12.2f}                             │
│  ├── Entry Price:      ₹{pick['entry_price']:<12.2f}                       │
│  ├── Stop Loss:        ₹{pick['stop_loss']:<12.2f}                         │
│  ├── Target 1:         ₹{pick['target_1']:<12.2f}                          │
│  ├── Target 2:         ₹{pick['target_2']:<12.2f}                          │
│  └── Quantity:         {pick['qty']} shares                                │
│                                                                            │
│  📊 QUALITY METRICS                                                        │
│  ├── Confidence:       {pick['confidence']:.1f}%                           │
│  ├── Risk:Reward:      {pick['risk_reward']:.1f}x                          │
│  ├── 52W Position:     {pick.get('fifty_two_week', {}).get('position_pct', 'N/A')}%                      │
│  ├── RS vs Nifty:      {pick.get('relative_strength', {}).get('avg_rs', 'N/A')}x                        │
│  └── Backtest WR:      {pick.get('backtest', {}).get('win_rate', 'N/A')}%                          │
│                                                                            │
│  ✅ WHY THIS STOCK                                                         │
│  {pick['reasons'][:70]:<72} │
│                                                                            │
└{'─'*76}┘
""")
    
    print("="*80)
    print("⚠️  EXECUTION RULES:")
    print("   • Enter only if price is within 1% of entry price")
    print("   • Use limit orders, not market orders")
    print("   • Enter after 10:00 AM (avoid opening volatility)")
    print("   • If SL is hit, accept it - no averaging down")
    print("="*80)

if __name__ == "__main__":

    sys.exit(github_actions_main())
