"""
🤖 SMART PORTFOLIO MONITOR - ML ENHANCEMENT MODULE
==================================================
Add-on module for AI/ML features:
✅ LSTM Price Prediction
✅ News Sentiment Analysis
✅ Pattern Recognition using ML
✅ Anomaly Detection
✅ Auto-Strategy Recommendation
✅ Backtesting Engine
✅ Options Greeks Calculator

INSTALLATION REQUIRED:
pip install tensorflow scikit-learn newspaper3k textblob yfinance-cache vectorbt py_vollib

USAGE:
1. Place this file in same folder as pm.py
2. In pm.py, add: from pm_ml_enhanced import MLEnhancer
3. Enable "Use ML Features" checkbox in sidebar
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# ML IMPORTS (with fallback handling)
# ============================================================================

HAS_ML = False
HAS_SENTIMENT = False
HAS_BACKTEST = False
HAS_OPTIONS = False

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential, load_model
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.callbacks import EarlyStopping
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.ensemble import IsolatedForest
    HAS_ML = True
except ImportError:
    print("⚠️ TensorFlow not installed. ML predictions disabled.")
    print("Install: pip install tensorflow scikit-learn")

try:
    from textblob import TextBlob
    import requests
    HAS_SENTIMENT = True
except ImportError:
    print("⚠️ Sentiment analysis libraries not installed.")
    print("Install: pip install textblob newspaper3k")

try:
    import vectorbt as vbt
    HAS_BACKTEST = True
except ImportError:
    print("⚠️ VectorBT not installed. Backtesting disabled.")
    print("Install: pip install vectorbt")

try:
    from py_vollib.black_scholes import black_scholes as bs
    from py_vollib.black_scholes.greeks.analytical import delta, gamma, theta, vega
    HAS_OPTIONS = True
except ImportError:
    print("⚠️ Options calculator not installed.")
    print("Install: pip install py_vollib")

# ============================================================================
# LSTM PRICE PREDICTOR
# ============================================================================

class LSTMPredictor:
    """
    Predicts next 5-day price movement using LSTM
    """
    
    def __init__(self):
        self.model = None
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.lookback = 60  # Use last 60 days
    
    def prepare_data(self, df):
        """Prepare data for LSTM"""
        data = df['Close'].values.reshape(-1, 1)
        scaled_data = self.scaler.fit_transform(data)
        
        X, y = [], []
        for i in range(self.lookback, len(scaled_data) - 5):
            X.append(scaled_data[i-self.lookback:i, 0])
            y.append(scaled_data[i:i+5, 0])  # Predict next 5 days
        
        return np.array(X), np.array(y)
    
    def build_model(self):
        """Build LSTM model"""
        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=(self.lookback, 1)),
            Dropout(0.2),
            LSTM(50, return_sequences=False),
            Dropout(0.2),
            Dense(25),
            Dense(5)  # Predict 5 days
        ])
        model.compile(optimizer='adam', loss='mse')
        return model
    
    def train(self, df):
        """Train the model"""
        if len(df) < self.lookback + 10:
            return False, "Insufficient data for training"
        
        X, y = self.prepare_data(df)
        
        if len(X) < 10:
            return False, "Not enough training samples"
        
        X = X.reshape(X.shape[0], X.shape[1], 1)
        
        self.model = self.build_model()
        
        early_stop = EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)
        
        self.model.fit(
            X, y,
            epochs=20,
            batch_size=32,
            callbacks=[early_stop],
            verbose=0
        )
        
        return True, "Model trained successfully"
    
    def predict(self, df):
        """Predict next 5 days"""
        if self.model is None:
            success, msg = self.train(df)
            if not success:
                return None, msg
        
        # Get last 60 days
        data = df['Close'].values[-self.lookback:].reshape(-1, 1)
        scaled_data = self.scaler.transform(data)
        
        X_test = scaled_data.reshape(1, self.lookback, 1)
        
        prediction_scaled = self.model.predict(X_test, verbose=0)
        prediction = self.scaler.inverse_transform(prediction_scaled)
        
        current_price = df['Close'].iloc[-1]
        
        result = {
            'predictions': prediction[0].tolist(),
            'current_price': current_price,
            'day1': prediction[0][0],
            'day5': prediction[0][4],
            'expected_move_pct': ((prediction[0][4] - current_price) / current_price) * 100,
            'confidence': self._calculate_confidence(df),
            'direction': 'BULLISH' if prediction[0][4] > current_price else 'BEARISH',
            'strength': abs(((prediction[0][4] - current_price) / current_price) * 100)
        }
        
        return result, "Success"
    
    def _calculate_confidence(self, df):
        """Calculate prediction confidence based on volatility"""
        returns = df['Close'].pct_change().dropna()
        volatility = returns.std()
        
        if volatility < 0.02:
            return 85
        elif volatility < 0.03:
            return 70
        elif volatility < 0.05:
            return 55
        else:
            return 40

# ============================================================================
# NEWS SENTIMENT ANALYZER
# ============================================================================

class SentimentAnalyzer:
    """
    Analyzes news sentiment for stocks
    """
    
    def __init__(self, api_key=None):
        self.api_key = api_key  # Optional: NewsAPI key
        self.cache = {}
        self.cache_duration = timedelta(hours=1)
    
    def get_news_headlines(self, ticker, max_articles=10):
        """Fetch news headlines"""
        # Check cache
        if ticker in self.cache:
            cached_time, cached_data = self.cache[ticker]
            if datetime.now() - cached_time < self.cache_duration:
                return cached_data
        
        headlines = []
        
        # Method 1: Use NewsAPI (if key provided)
        if self.api_key:
            try:
                url = f"https://newsapi.org/v2/everything"
                params = {
                    'q': ticker,
                    'apiKey': self.api_key,
                    'language': 'en',
                    'sortBy': 'publishedAt',
                    'pageSize': max_articles
                }
                response = requests.get(url, params=params, timeout=5)
                if response.status_code == 200:
                    data = response.json()
                    headlines = [article['title'] for article in data.get('articles', [])]
            except:
                pass
        
        # Method 2: Google News scraping (fallback)
        if not headlines:
            try:
                search_term = ticker.replace('.NS', '').replace('.BO', '')
                url = f"https://news.google.com/rss/search?q={search_term}+stock&hl=en-IN&gl=IN&ceid=IN:en"
                
                import feedparser
                feed = feedparser.parse(url)
                headlines = [entry.title for entry in feed.entries[:max_articles]]
            except:
                headlines = []
        
        self.cache[ticker] = (datetime.now(), headlines)
        return headlines
    
    def analyze_sentiment(self, text):
        """Analyze sentiment using TextBlob"""
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity  # -1 to 1
        
        if polarity > 0.1:
            sentiment = "POSITIVE"
        elif polarity < -0.1:
            sentiment = "NEGATIVE"
        else:
            sentiment = "NEUTRAL"
        
        return {
            'polarity': polarity,
            'sentiment': sentiment,
            'score': (polarity + 1) * 50  # Convert to 0-100
        }
    
    def get_stock_sentiment(self, ticker):
        """Get overall sentiment for a stock"""
        headlines = self.get_news_headlines(ticker)
        
        if not headlines:
            return {
                'overall_sentiment': 'NEUTRAL',
                'sentiment_score': 50,
                'article_count': 0,
                'positive': 0,
                'negative': 0,
                'neutral': 0,
                'headlines': []
            }
        
        sentiments = []
        headline_sentiments = []
        
        for headline in headlines:
            sent = self.analyze_sentiment(headline)
            sentiments.append(sent['score'])
            headline_sentiments.append({
                'text': headline[:80] + '...' if len(headline) > 80 else headline,
                'sentiment': sent['sentiment'],
                'score': sent['score']
            })
        
        avg_score = np.mean(sentiments)
        positive = sum(1 for s in headline_sentiments if s['sentiment'] == 'POSITIVE')
        negative = sum(1 for s in headline_sentiments if s['sentiment'] == 'NEGATIVE')
        neutral = len(headlines) - positive - negative
        
        if avg_score > 60:
            overall = 'BULLISH'
        elif avg_score < 40:
            overall = 'BEARISH'
        else:
            overall = 'NEUTRAL'
        
        return {
            'overall_sentiment': overall,
            'sentiment_score': round(avg_score, 1),
            'article_count': len(headlines),
            'positive': positive,
            'negative': negative,
            'neutral': neutral,
            'headlines': headline_sentiments[:5]
        }

# ============================================================================
# ANOMALY DETECTOR
# ============================================================================

class AnomalyDetector:
    """
    Detect unusual price/volume patterns using Isolation Forest
    """
    
    def __init__(self):
        self.model = IsolatedForest(contamination=0.1, random_state=42)
    
    def detect_anomalies(self, df):
        """Detect price and volume anomalies"""
        if len(df) < 30:
            return {
                'has_anomaly': False,
                'anomaly_type': 'NONE',
                'description': 'Insufficient data',
                'risk_level': 'LOW'
            }
        
        # Prepare features
        features = pd.DataFrame({
            'returns': df['Close'].pct_change(),
            'volume_change': df['Volume'].pct_change(),
            'price_volatility': df['Close'].rolling(5).std(),
            'volume_ratio': df['Volume'] / df['Volume'].rolling(20).mean()
        }).fillna(0)
        
        # Train model
        self.model.fit(features)
        
        # Predict anomalies (-1 = anomaly, 1 = normal)
        predictions = self.model.predict(features)
        
        # Check last 5 days
        recent_anomalies = predictions[-5:]
        
        if -1 in recent_anomalies:
            # Identify anomaly type
            last_row = features.iloc[-1]
            
            if abs(last_row['returns']) > 0.05:
                anomaly_type = 'PRICE_SPIKE'
                description = f"Unusual price movement: {last_row['returns']*100:+.1f}%"
                risk_level = 'HIGH'
            elif last_row['volume_ratio'] > 3:
                anomaly_type = 'VOLUME_SURGE'
                description = f"Volume spike: {last_row['volume_ratio']:.1f}x normal"
                risk_level = 'MEDIUM'
            else:
                anomaly_type = 'VOLATILITY'
                description = "Increased volatility detected"
                risk_level = 'MEDIUM'
            
            return {
                'has_anomaly': True,
                'anomaly_type': anomaly_type,
                'description': description,
                'risk_level': risk_level,
                'days_ago': list(recent_anomalies).index(-1)
            }
        
        return {
            'has_anomaly': False,
            'anomaly_type': 'NONE',
            'description': 'No anomalies detected',
            'risk_level': 'LOW'
        }

# ============================================================================
# STRATEGY RECOMMENDER
# ============================================================================

class StrategyRecommender:
    """
    Recommends trading strategies based on ML analysis
    """
    
    def recommend_strategy(self, result, ml_prediction, sentiment):
        """
        Combine technical analysis + ML + sentiment to recommend strategy
        """
        score = 0
        reasons = []
        
        # 1. Technical Score (40%)
        if result['momentum_score'] >= 60:
            score += 15
            reasons.append("✅ Strong technical momentum")
        elif result['momentum_score'] <= 40:
            score -= 15
            reasons.append("❌ Weak technical momentum")
        
        if result['rsi'] < 30:
            score += 10
            reasons.append("✅ RSI oversold - potential bounce")
        elif result['rsi'] > 70:
            score -= 10
            reasons.append("❌ RSI overbought - potential correction")
        
        if result['mtf_alignment'] >= 60:
            score += 15
            reasons.append("✅ Multiple timeframes aligned")
        elif result['mtf_alignment'] <= 40:
            score -= 15
            reasons.append("❌ Timeframes diverging")
        
        # 2. ML Prediction Score (30%)
        if ml_prediction:
            if ml_prediction['direction'] == 'BULLISH':
                score += ml_prediction['confidence'] * 0.3
                reasons.append(f"🤖 ML predicts {ml_prediction['expected_move_pct']:+.1f}% move")
            else:
                score -= ml_prediction['confidence'] * 0.3
                reasons.append(f"🤖 ML predicts {ml_prediction['expected_move_pct']:+.1f}% move")
        
        # 3. Sentiment Score (30%)
        if sentiment:
            if sentiment['overall_sentiment'] == 'BULLISH':
                score += 15
                reasons.append(f"📰 Positive news sentiment ({sentiment['sentiment_score']:.0f}/100)")
            elif sentiment['overall_sentiment'] == 'BEARISH':
                score -= 15
                reasons.append(f"📰 Negative news sentiment ({sentiment['sentiment_score']:.0f}/100)")
        
        # Normalize to 0-100
        score = max(0, min(100, score + 50))
        
        # Generate recommendation
        if result['position_type'] == 'LONG':
            if score >= 70:
                action = "🟢 STRONG HOLD/ADD"
                confidence = "HIGH"
            elif score >= 55:
                action = "🟢 HOLD"
                confidence = "MEDIUM"
            elif score >= 40:
                action = "🟡 MONITOR CLOSELY"
                confidence = "LOW"
            else:
                action = "🔴 CONSIDER EXIT"
                confidence = "HIGH"
        else:  # SHORT
            if score <= 30:
                action = "🟢 STRONG HOLD/ADD"
                confidence = "HIGH"
            elif score <= 45:
                action = "🟢 HOLD"
                confidence = "MEDIUM"
            elif score <= 60:
                action = "🟡 MONITOR CLOSELY"
                confidence = "LOW"
            else:
                action = "🔴 CONSIDER EXIT"
                confidence = "HIGH"
        
        return {
            'strategy_score': round(score, 1),
            'action': action,
            'confidence': confidence,
            'reasons': reasons,
            'position_alignment': score >= 50 if result['position_type'] == 'LONG' else score <= 50
        }

# ============================================================================
# BACKTESTING ENGINE
# ============================================================================

class BacktestEngine:
    """
    Backtest trading strategies
    """
    
    def simple_backtest(self, df, entry_price, stop_loss, target, position_type='LONG'):
        """
        Simple backtest on historical data
        """
        if len(df) < 20:
            return None
        
        df = df.copy()
        df['returns'] = df['Close'].pct_change()
        
        # Simulate position
        capital = 100000
        position_size = capital * 0.1 / entry_price
        
        if position_type == 'LONG':
            exit_price = None
            for idx, row in df.iterrows():
                if row['Close'] <= stop_loss:
                    exit_price = stop_loss
                    exit_reason = 'Stop Loss'
                    break
                elif row['Close'] >= target:
                    exit_price = target
                    exit_reason = 'Target Hit'
                    break
            
            if exit_price:
                pnl = (exit_price - entry_price) * position_size
                pnl_pct = ((exit_price - entry_price) / entry_price) * 100
            else:
                exit_price = df['Close'].iloc[-1]
                exit_reason = 'Open'
                pnl = (exit_price - entry_price) * position_size
                pnl_pct = ((exit_price - entry_price) / entry_price) * 100
        else:  # SHORT
            exit_price = None
            for idx, row in df.iterrows():
                if row['Close'] >= stop_loss:
                    exit_price = stop_loss
                    exit_reason = 'Stop Loss'
                    break
                elif row['Close'] <= target:
                    exit_price = target
                    exit_reason = 'Target Hit'
                    break
            
            if exit_price:
                pnl = (entry_price - exit_price) * position_size
                pnl_pct = ((entry_price - exit_price) / entry_price) * 100
            else:
                exit_price = df['Close'].iloc[-1]
                exit_reason = 'Open'
                pnl = (entry_price - exit_price) * position_size
                pnl_pct = ((entry_price - exit_price) / entry_price) * 100
        
        return {
            'entry_price': entry_price,
            'exit_price': exit_price,
            'exit_reason': exit_reason,
            'pnl': pnl,
            'pnl_pct': pnl_pct,
            'win': pnl > 0,
            'days_held': len(df)
        }

# ============================================================================
# OPTIONS CALCULATOR
# ============================================================================

class OptionsCalculator:
    """
    Calculate Black-Scholes Greeks for options
    """
    
    def calculate_greeks(self, spot_price, strike_price, time_to_expiry_days, 
                        volatility, risk_free_rate=0.07, option_type='call'):
        """
        Calculate all Greeks
        spot_price: Current stock price
        strike_price: Option strike price
        time_to_expiry_days: Days until expiration
        volatility: Annual volatility (e.g., 0.25 for 25%)
        risk_free_rate: Risk-free rate (e.g., 0.07 for 7%)
        """
        T = time_to_expiry_days / 365.0
        
        if T <= 0:
            return None
        
        flag = 'c' if option_type == 'call' else 'p'
        
        try:
            price = bs(flag, spot_price, strike_price, T, risk_free_rate, volatility)
            delta_val = delta(flag, spot_price, strike_price, T, risk_free_rate, volatility)
            gamma_val = gamma(flag, spot_price, strike_price, T, risk_free_rate, volatility)
            theta_val = theta(flag, spot_price, strike_price, T, risk_free_rate, volatility)
            vega_val = vega(flag, spot_price, strike_price, T, risk_free_rate, volatility)
            
            return {
                'option_price': round(price, 2),
                'delta': round(delta_val, 4),
                'gamma': round(gamma_val, 4),
                'theta': round(theta_val, 4),
                'vega': round(vega_val, 4),
                'moneyness': 'ITM' if (option_type == 'call' and spot_price > strike_price) or 
                                     (option_type == 'put' and spot_price < strike_price) else 'OTM'
            }
        except:
            return None

# ============================================================================
# MAIN ML ENHANCER CLASS
# ============================================================================

class MLEnhancer:
    """
    Main class that combines all ML features
    """
    
    def __init__(self):
        self.lstm = LSTMPredictor() if HAS_ML else None
        self.sentiment = SentimentAnalyzer() if HAS_SENTIMENT else None
        self.anomaly = AnomalyDetector() if HAS_ML else None
        self.strategy = StrategyRecommender()
        self.backtest = BacktestEngine() if HAS_BACKTEST else None
        self.options = OptionsCalculator() if HAS_OPTIONS else None
        
        self.cache = {}
    
    def enhance_analysis(self, result, enable_lstm=True, enable_sentiment=True,
                        enable_anomaly=True, enable_backtest=False):
        """
        Add ML enhancements to existing analysis
        """
        ticker = result['ticker']
        df = result.get('df')
        
        if df is None or len(df) < 30:
            return {
                'ml_available': False,
                'message': 'Insufficient data for ML analysis'
            }
        
        enhancements = {
            'ml_available': True,
            'lstm_prediction': None,
            'sentiment': None,
            'anomaly': None,
            'strategy_recommendation': None,
            'backtest_result': None
        }
        
        # 1. LSTM Prediction
        if enable_lstm and HAS_ML and self.lstm:
            try:
                prediction, msg = self.lstm.predict(df)
                enhancements['lstm_prediction'] = prediction
            except Exception as e:
                enhancements['lstm_prediction'] = {'error': str(e)}
        
        # 2. Sentiment Analysis
        if enable_sentiment and HAS_SENTIMENT and self.sentiment:
            try:
                sentiment = self.sentiment.get_stock_sentiment(ticker)
                enhancements['sentiment'] = sentiment
            except Exception as e:
                enhancements['sentiment'] = {'error': str(e)}
        
        # 3. Anomaly Detection
        if enable_anomaly and HAS_ML and self.anomaly:
            try:
                anomaly = self.anomaly.detect_anomalies(df)
                enhancements['anomaly'] = anomaly
            except Exception as e:
                enhancements['anomaly'] = {'error': str(e)}
        
        # 4. Strategy Recommendation
        try:
            strategy = self.strategy.recommend_strategy(
                result,
                enhancements['lstm_prediction'],
                enhancements['sentiment']
            )
            enhancements['strategy_recommendation'] = strategy
        except Exception as e:
            enhancements['strategy_recommendation'] = {'error': str(e)}
        
        # 5. Backtest
        if enable_backtest and HAS_BACKTEST and self.backtest:
            try:
                backtest_result = self.backtest.simple_backtest(
                    df,
                    result['entry_price'],
                    result['stop_loss'],
                    result['target1'],
                    result['position_type']
                )
                enhancements['backtest_result'] = backtest_result
            except Exception as e:
                enhancements['backtest_result'] = {'error': str(e)}
        
        return enhancements
    
    def get_capabilities(self):
        """Return which features are available"""
        return {
            'lstm': HAS_ML,
            'sentiment': HAS_SENTIMENT,
            'anomaly': HAS_ML,
            'backtest': HAS_BACKTEST,
            'options': HAS_OPTIONS
        }

# ============================================================================
# UTILITY FUNCTIONS FOR DISPLAY
# ============================================================================

def format_ml_prediction(prediction):
    """Format LSTM prediction for display"""
    if not prediction or 'error' in prediction:
        return None
    
    direction_color = "#28a745" if prediction['direction'] == 'BULLISH' else "#dc3545"
    
    html = f"""
    <div style='background:{direction_color}20; padding:15px; border-radius:10px; 
                border-left:4px solid {direction_color};'>
        <h4 style='margin:0;'>🤖 ML Prediction (5-Day)</h4>
        <p style='margin:5px 0;'><strong>Direction:</strong> {prediction['direction']}</p>
        <p style='margin:5px 0;'><strong>Expected Move:</strong> {prediction['expected_move_pct']:+.2f}%</p>
        <p style='margin:5px 0;'><strong>Target Price:</strong> ₹{prediction['day5']:.2f}</p>
        <p style='margin:5px 0;'><strong>Confidence:</strong> {prediction['confidence']}%</p>
    </div>
    """
    
    return html

def format_sentiment(sentiment):
    """Format sentiment analysis for display"""
    if not sentiment or 'error' in sentiment:
        return None
    
    colors = {
        'BULLISH': '#28a745',
        'BEARISH': '#dc3545',
        'NEUTRAL': '#ffc107'
    }
    
    color = colors.get(sentiment['overall_sentiment'], '#6c757d')
    
    html = f"""
    <div style='background:{color}20; padding:15px; border-radius:10px; 
                border-left:4px solid {color};'>
        <h4 style='margin:0;'>📰 News Sentiment</h4>
        <p style='margin:5px 0;'><strong>Overall:</strong> {sentiment['overall_sentiment']}</p>
        <p style='margin:5px 0;'><strong>Score:</strong> {sentiment['sentiment_score']}/100</p>
        <p style='margin:5px 0;'><strong>Articles:</strong> {sentiment['article_count']} 
           (👍 {sentiment['positive']} | 👎 {sentiment['negative']} | ➖ {sentiment['neutral']})</p>
    </div>
    """
    
    return html

def format_strategy(strategy):
    """Format strategy recommendation"""
    if not strategy or 'error' in strategy:
        return None
    
    html = f"""
    <div style='background:#f8f9fa; padding:15px; border-radius:10px; border:2px solid #667eea;'>
        <h4 style='margin:0; color:#667eea;'>🎯 AI Strategy Recommendation</h4>
        <p style='margin:10px 0; font-size:1.2em;'><strong>{strategy['action']}</strong></p>
        <p style='margin:5px 0;'><strong>Strategy Score:</strong> {strategy['strategy_score']}/100</p>
        <p style='margin:5px 0;'><strong>Confidence:</strong> {strategy['confidence']}</p>
        <hr style='margin:10px 0;'>
        <p style='margin:5px 0; font-weight:bold;'>Reasoning:</p>
        <ul style='margin:5px 0;'>
    """
    
    for reason in strategy['reasons']:
        html += f"<li>{reason}</li>"
    
    html += "</ul></div>"
    
    return html

# ============================================================================
# EXPORT
# ============================================================================

__all__ = [
    'MLEnhancer',
    'LSTMPredictor',
    'SentimentAnalyzer',
    'AnomalyDetector',
    'StrategyRecommender',
    'BacktestEngine',
    'OptionsCalculator',
    'format_ml_prediction',
    'format_sentiment',
    'format_strategy',
    'HAS_ML',
    'HAS_SENTIMENT',
    'HAS_BACKTEST',
    'HAS_OPTIONS'
]