"""Bounded, leased in-memory browser games for a single web worker."""
from dataclasses import dataclass, field
from threading import RLock
from time import monotonic
from uuid import uuid4

from GameSession import GameSession


@dataclass
class BrowserState:
    game_session: GameSession = field(default_factory=GameSession)
    sgf_replay: object = None
    recommendation_ai: dict = field(default_factory=dict)
    touched: float = 0.0
    active: int = 0

    @property
    def game(self):
        return self.game_session.game

    @game.setter
    def game(self, value):
        self.game_session.game = value


class SessionCapacityError(RuntimeError):
    pass


class BrowserSessionRegistry:
    def __init__(self, max_sessions=8, ttl=3600, clock=monotonic):
        if max_sessions < 1 or ttl <= 0:
            raise ValueError('positive session capacity and lifetime required')
        self.max_sessions, self.ttl, self.clock = max_sessions, ttl, clock
        self.states = {}
        self.lock = RLock()

    def acquire(self, key):
        with self.lock:
            now = self.clock()
            for old, state in list(self.states.items()):
                if not state.active and now - state.touched > self.ttl:
                    del self.states[old]
            state = self.states.get(key) if isinstance(key, str) else None
            created = state is None
            if created:
                if len(self.states) >= self.max_sessions:
                    idle = [(s.touched, k) for k, s in self.states.items() if not s.active]
                    if not idle:
                        raise SessionCapacityError('All game slots are currently busy. Please try again shortly.')
                    del self.states[min(idle)[1]]
                key, state = uuid4().hex, BrowserState()
                self.states[key] = state
            state.touched = now
            state.active += 1
            return key, state, created

    def release(self, key):
        with self.lock:
            state = self.states.get(key)
            if state is not None:
                state.active = max(0, state.active - 1)
                state.touched = self.clock()


class LegacyState:
    """Keep the existing desktop globals/test hooks outside isolated mode."""
    def __init__(self, namespace):
        object.__setattr__(self, '_namespace', namespace)

    def __getattr__(self, name):
        return self._namespace[name]

    def __setattr__(self, name, value):
        self._namespace[name] = value
