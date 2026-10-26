"""Persistent reuse of exact deterministic generation inputs; never used for latency."""
from __future__ import annotations
import hashlib
import json
import sqlite3
from pathlib import Path


def canonical(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False)


class PredictionCache:
    def __init__(self,path,identity):
        path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
        self.identity=identity
        self.fingerprint=hashlib.sha256(canonical(identity).encode()).hexdigest()
        self.db=sqlite3.connect(path)
        self.db.execute('PRAGMA journal_mode=WAL')
        self.db.execute('CREATE TABLE IF NOT EXISTS predictions (identity TEXT, input_key TEXT, value TEXT, PRIMARY KEY(identity,input_key))')
        self.hits=self.misses=0
        self._pending=0

    @staticmethod
    def key(encoded):
        # Includes IDs and masks; no labels, source IDs, caps or narrative aliases.
        return hashlib.sha256(canonical({k:encoded[k] for k in sorted(encoded)}).encode()).hexdigest()

    def get(self,key):
        row=self.db.execute('SELECT value FROM predictions WHERE identity=? AND input_key=?',(self.fingerprint,key)).fetchone()
        if row is None:
            self.misses+=1;return None
        self.hits+=1;return json.loads(row[0])

    def put(self,key,result):
        value=canonical(result)
        previous=self.db.execute('SELECT value FROM predictions WHERE identity=? AND input_key=?',(self.fingerprint,key)).fetchone()
        if previous is not None and previous[0]!=value:
            raise RuntimeError('Conflicting outputs for the same frozen deterministic input')
        self.db.execute('INSERT OR IGNORE INTO predictions VALUES (?,?,?)',(self.fingerprint,key,value))
        self._pending+=1
        if self._pending>=32:self.flush()

    def flush(self):
        self.db.commit();self._pending=0

    def close(self):
        self.flush();self.db.close()
