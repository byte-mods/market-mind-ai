"""
MarketMind AI - Database Module
SQLite database for storing news and cache
"""

import sqlite3
import os
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import json


class Database:
    def __init__(self, db_path: str = None):
        if db_path is None:
            app_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            db_path = os.path.join(app_dir, "marketmind.db")

        self.db_path = db_path
        self.init_database()

    def init_database(self):
        """Initialize database tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # News table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS news (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                content TEXT,
                source TEXT,
                url TEXT UNIQUE,
                published_at TEXT,
                fetched_at TEXT,
                sentiment_score REAL,
                sentiment_label TEXT,
                sectors TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Stock prices cache table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS stock_prices (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                date TEXT NOT NULL,
                open REAL,
                high REAL,
                low REAL,
                close REAL,
                volume INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(symbol, date)
            )
        """)

        # Sector data table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS sector_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                sector_name TEXT NOT NULL,
                date TEXT NOT NULL,
                close REAL,
                change_pct REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(sector_name, date)
            )
        """)

        # RL signals table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS rl_signals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                symbol TEXT,
                action TEXT,
                confidence REAL,
                entry_price REAL,
                exit_price REAL,
                stop_loss REAL,
                rationale TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Portfolio simulation results
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS portfolio_sims (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                initial_capital REAL,
                final_value REAL,
                sharpe_ratio REAL,
                max_drawdown REAL,
                win_rate REAL,
                scenario TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        conn.commit()
        conn.close()

    def store_news(self, news_item: Dict) -> bool:
        """Store a news item"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            sectors_json = json.dumps(news_item.get('sectors', []))

            cursor.execute("""
                INSERT OR REPLACE INTO news
                (title, content, source, url, published_at, fetched_at,
                 sentiment_score, sentiment_label, sectors)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                news_item.get('title'),
                news_item.get('content', ''),
                news_item.get('source', ''),
                news_item.get('url'),
                news_item.get('published_at'),
                news_item.get('fetched_at', datetime.now().isoformat()),
                news_item.get('sentiment_score'),
                news_item.get('sentiment_label'),
                sectors_json
            ))

            conn.commit()
            conn.close()
            return True
        except Exception as e:
            print(f"Error storing news: {e}")
            return False

    def get_news(self, days: int = 7, sector: str = None, limit: int = 100) -> List[Dict]:
        """Get news items"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        since_date = (datetime.now() - timedelta(days=days)).isoformat()

        query = "SELECT * FROM news WHERE fetched_at >= ?"
        params = [since_date]

        if sector:
            query += " AND sectors LIKE ?"
            params.append(f"%{sector}%")

        query += " ORDER BY fetched_at DESC LIMIT ?"
        params.append(limit)

        cursor.execute(query, params)
        rows = cursor.fetchall()

        news_list = []
        for row in rows:
            news_list.append({
                'id': row[0],
                'title': row[1],
                'content': row[2],
                'source': row[3],
                'url': row[4],
                'published_at': row[5],
                'fetched_at': row[6],
                'sentiment_score': row[7],
                'sentiment_label': row[8],
                'sectors': json.loads(row[9]) if row[9] else []
            })

        conn.close()
        return news_list

    def store_stock_price(self, symbol: str, date: str, data: Dict):
        """Store stock price data"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute("""
                INSERT OR REPLACE INTO stock_prices
                (symbol, date, open, high, low, close, volume)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (symbol, date, data.get('open'), data.get('high'),
                  data.get('low'), data.get('close'), data.get('volume')))

            conn.commit()
            conn.close()
        except Exception as e:
            print(f"Error storing stock price: {e}")

    def get_stock_prices(self, symbol: str, days: int = 365) -> List[Dict]:
        """Get historical stock prices"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        since_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')

        cursor.execute("""
            SELECT date, open, high, low, close, volume
            FROM stock_prices
            WHERE symbol = ? AND date >= ?
            ORDER BY date ASC
        """, (symbol, since_date))

        rows = cursor.fetchall()

        prices = []
        for row in rows:
            prices.append({
                'date': row[0],
                'open': row[1],
                'high': row[2],
                'low': row[3],
                'close': row[4],
                'volume': row[5]
            })

        conn.close()
        return prices

    def store_rl_signal(self, signal: Dict):
        """Store RL trading signal"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO rl_signals
            (timestamp, symbol, action, confidence, entry_price, exit_price,
             stop_loss, rationale)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            signal.get('timestamp', datetime.now().isoformat()),
            signal.get('symbol'),
            signal.get('action'),
            signal.get('confidence'),
            signal.get('entry_price'),
            signal.get('exit_price'),
            signal.get('stop_loss'),
            signal.get('rationale', '')
        ))

        conn.commit()
        conn.close()

    def get_recent_signals(self, limit: int = 50) -> List[Dict]:
        """Get recent RL signals"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT timestamp, symbol, action, confidence, entry_price,
                   exit_price, stop_loss, rationale
            FROM rl_signals
            ORDER BY timestamp DESC
            LIMIT ?
        """, (limit,))

        rows = cursor.fetchall()

        signals = []
        for row in rows:
            signals.append({
                'timestamp': row[0],
                'symbol': row[1],
                'action': row[2],
                'confidence': row[3],
                'entry_price': row[4],
                'exit_price': row[5],
                'stop_loss': row[6],
                'rationale': row[7]
            })

        conn.close()
        return signals

    def store_portfolio_simulation(self, sim_result: Dict):
        """Store portfolio simulation result"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO portfolio_sims
            (timestamp, initial_capital, final_value, sharpe_ratio,
             max_drawdown, win_rate, scenario)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            datetime.now().isoformat(),
            sim_result.get('initial_capital'),
            sim_result.get('final_value'),
            sim_result.get('sharpe_ratio'),
            sim_result.get('max_drawdown'),
            sim_result.get('win_rate'),
            sim_result.get('scenario', 'base')
        ))

        conn.commit()
        conn.close()
