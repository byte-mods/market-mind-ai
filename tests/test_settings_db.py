"""Tests for the SQLite settings KV store and watchlist persistence."""
import os
import json
import tempfile

import pytest

from marketmind.core.database import Database


@pytest.fixture
def db():
    """Database backed by a fresh temp file — isolates the settings table
    from the dev-time marketmind.db so tests can't pollute real preferences."""
    fd, path = tempfile.mkstemp(suffix='.db')
    os.close(fd)
    try:
        yield Database(db_path=path)
    finally:
        os.unlink(path)


def test_set_and_get_setting_roundtrips_a_list(db):
    """Watchlist is the canonical use case — list of strings."""
    db.set_setting('watchlist', ['RELIANCE', 'TCS', 'INFY'])
    assert db.get_setting('watchlist') == ['RELIANCE', 'TCS', 'INFY']


def test_set_setting_upserts(db):
    db.set_setting('watchlist', ['A'])
    db.set_setting('watchlist', ['B', 'C'])
    assert db.get_setting('watchlist') == ['B', 'C']


def test_get_setting_returns_default_when_missing(db):
    assert db.get_setting('nope') is None
    assert db.get_setting('nope', default=[]) == []


def test_set_setting_handles_dict_value(db):
    db.set_setting('app', {'max_order_value': 50000, 'sl': 1.5})
    out = db.get_setting('app')
    assert out == {'max_order_value': 50000, 'sl': 1.5}


def test_get_setting_returns_default_on_corrupt_json(db):
    """If a row is hand-edited and the JSON is invalid, fall back to default
    rather than raising — the UI must still render."""
    import sqlite3
    conn = sqlite3.connect(db.db_path)
    conn.execute("INSERT INTO settings (key, value) VALUES (?, ?)",
                 ('busted', '{not valid json'))
    conn.commit()
    conn.close()
    assert db.get_setting('busted', default='fallback') == 'fallback'


def test_settings_table_persists_across_database_instances(db):
    """A second Database opening the same file should see the saved values —
    proves we're not relying on in-memory state."""
    db.set_setting('watchlist', ['HDFC', 'ICICI'])
    db2 = Database(db_path=db.db_path)
    assert db2.get_setting('watchlist') == ['HDFC', 'ICICI']
