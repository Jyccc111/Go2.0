"""Manage a Go game and the players assigned to each color."""

from enum import IntEnum

from Game import Game, Player


class Type_player(IntEnum):
    AI = 0
    HUMAN = 1


class GameSession:
    """Coordinate player turns while leaving Go rules in Game."""

    def __init__(self, ai_factory=None, game_factory=Game):
        self._ai_factory = ai_factory
        self._game_factory = game_factory
        self.game = game_factory()
        self.player_black = Type_player.HUMAN
        self.player_white = Type_player.HUMAN
        self.AI_black = None
        self.AI_white = None
        self.komi = 0.0
        self.next_player = Player.BLACK
        self.history = []
        self.pass_count = 0
        self.game_over = False
        self.result = None

    def initialize(self, player_black=Type_player.HUMAN,
                   player_white=Type_player.HUMAN, komi=0):
        try:
            black_type = Type_player(player_black)
            white_type = Type_player(player_white)
        except (TypeError, ValueError) as exc:
            raise ValueError("Player types must be Type_player.AI or Type_player.HUMAN") from exc
        try:
            komi_value = float(komi)
        except (TypeError, ValueError) as exc:
            raise ValueError("komi must be numeric") from exc

        self.game = self._game_factory()
        self.player_black = black_type
        self.player_white = white_type
        self.komi = komi_value
        self.AI_black = self._new_ai() if black_type == Type_player.AI else None
        self.AI_white = self._new_ai() if white_type == Type_player.AI else None
        self.next_player = Player.BLACK
        self.history = []
        self.pass_count = 0
        self.game_over = False
        self.result = None
        return self.get_state()

    def _new_ai(self):
        if self._ai_factory is not None:
            return self._ai_factory()
        from AI import AI
        return AI()

    @staticmethod
    def _normalize_color(color):
        try:
            value = int(color.value) if isinstance(color, Player) else int(color)
        except (TypeError, ValueError, AttributeError) as exc:
            raise ValueError("color must be 1 (black) or -1 (white)") from exc
        if value not in (1, -1):
            raise ValueError("color must be 1 (black) or -1 (white)")
        return Player(value)

    @staticmethod
    def _color_name(color):
        return "black" if color == Player.BLACK else "white"

    def player_type(self, color):
        color = self._normalize_color(color)
        return self.player_black if color == Player.BLACK else self.player_white

    def play_human(self, row, col, color):
        color = self._normalize_color(color)
        if self.player_type(color) != Type_player.HUMAN:
            return {"success": False, "error": "The current side is AI-controlled."}
        return self.play(row, col, color)

    def play(self, row, col, color):
        color = self._normalize_color(color)
        if self.game_over:
            return {"success": False, "error": "The game is over."}
        if color != self.next_player:
            return {"success": False, "error": f"It is {self._color_name(self.next_player)}'s turn."}
        if isinstance(row, bool) or isinstance(col, bool):
            return {"success": False, "error": "row and col must be integers from 0 to 18."}
        try:
            parsed_row, parsed_col = int(row), int(col)
            if isinstance(row, float) and not row.is_integer():
                raise ValueError
            if isinstance(col, float) and not col.is_integer():
                raise ValueError
            if isinstance(row, str) and str(parsed_row) != row.strip():
                raise ValueError
            if isinstance(col, str) and str(parsed_col) != col.strip():
                raise ValueError
            row, col = parsed_row, parsed_col
        except (TypeError, ValueError):
            return {"success": False, "error": "row and col must be integers from 0 to 18."}
        if not (0 <= row < 19 and 0 <= col < 19):
            return {"success": False, "error": "row and col must be integers from 0 to 18."}
        if self.game.board.get(row, col) != 0:
            return {"success": False, "error": "The intersection is occupied."}

        previous_player = self.game.current_player
        success = self.game.play(row, col, color.value)
        if not success:
            # Game.play assigns current_player before checking legality.
            self.game.current_player = previous_player
            return {"success": False, "error": "Illegal move."}

        move = {"color": color.value, "row": row, "col": col, "pass": False}
        self.history.append(move)
        self.next_player = Player(-color.value)
        self.pass_count = 0
        # Game.play() leaves current_player set to the player who just moved
        # and computes islegal for that color. The next AI must receive a
        # legality map for the opponent whose turn it is now.
        self.game.current_player = self.next_player
        self.game.update_legal_position()
        return {"success": True, "move": move}

    def pass_human(self, color):
        color = self._normalize_color(color)
        if self.player_type(color) != Type_player.HUMAN:
            return {"success": False, "error": "The current side is AI-controlled."}
        return self.pass_turn(color)

    def pass_turn(self, color=None):
        color = self.next_player if color is None else self._normalize_color(color)
        if self.game_over:
            return {"success": False, "error": "The game is over."}
        if color != self.next_player:
            return {"success": False, "error": f"It is {self._color_name(self.next_player)}'s turn."}

        # GameSession keeps current_player aligned with the side to move.
        self.game.current_player = color
        self.game.is_pass()
        move = {"color": color.value, "row": None, "col": None, "pass": True}
        self.history.append(move)
        self.next_player = Player(-color.value)
        self.pass_count += 1
        if self.pass_count >= 2:
            self.game_over = True
            self.result = "two_passes"
        return {"success": True, "pass": True, "game_over": self.game_over, "move": move}

    def resign(self, color=None):
        color = self.next_player if color is None else self._normalize_color(color)
        if self.game_over:
            return {"success": False, "error": "The game is over."}
        if color != self.next_player:
            return {"success": False, "error": f"It is {self._color_name(self.next_player)}'s turn."}
        self.game_over = True
        self.result = "W+R" if color == Player.BLACK else "B+R"
        move = {"color": color.value, "row": None, "col": None, "resign": True}
        self.history.append(move)
        return {"success": True, "resigned": True, "result": self.result, "move": move}

    def play_ai(self):
        color = self.next_player
        if self.game_over:
            return {"success": False, "error": "The game is over."}
        if self.player_type(color) != Type_player.AI:
            return {"success": False, "error": "The current side is human-controlled."}
        ai = self.AI_black if color == Player.BLACK else self.AI_white
        try:
            if hasattr(ai, "predict_action"):
                action = ai.predict_action(self.game, color.value)
                kind = action.get("kind")
                if kind == "pass":
                    return self.pass_turn(color)
                if kind == "resign":
                    return self.resign(color)
                if kind != "move":
                    return {"success": False, "error": "AI returned an invalid action."}
                row, col = action.get("row"), action.get("col")
            else:
                # Compatibility with older AI implementations that return
                # coordinates and reserve (0, 0) / (-1, -1) for pass/resign.
                move = ai.predict_move(self.game)
                if move is None:
                    return self.resign(color)
                row, col = move
                if (row, col) == (-1, -1):
                    return self.resign(color)
                if (row, col) == (0, 0):
                    return self.pass_turn(color)
        except Exception as exc:
            return {"success": False, "error": f"AI prediction failed: {exc}"}
        return self.play(row, col, color)

    def auto_play(self, max_moves=1000):
        if self.player_black != Type_player.AI or self.player_white != Type_player.AI:
            raise ValueError("auto_play requires AI on both sides")
        if isinstance(max_moves, bool) or not isinstance(max_moves, int) or max_moves < 1:
            raise ValueError("max_moves must be a positive integer")
        moves = []
        while not self.game_over and len(moves) < max_moves:
            outcome = self.play_ai()
            if not outcome.get("success"):
                return {"success": False, "moves": moves, "error": outcome.get("error")}
            moves.append(outcome.get("move"))
        return {
            "success": True,
            "moves": moves,
            "game_over": self.game_over,
            "result": self.result,
            "move_limit_reached": not self.game_over,
        }

    def get_state(self):
        return {
            "board": self.game.board.board.tolist(),
            "air": self.game.air.board.tolist(),
            "cluster_air": self.game.cluster_air.tolist(),
            "islegal": self.game.islegal.tolist(),
            "num_eat": self.game.num_eat.tolist(),
            "current_player": int(self.game.current_player.value),
            "next_player": int(self.next_player.value),
            "player_black": int(self.player_black),
            "player_white": int(self.player_white),
            "komi": self.komi,
            "game_over": self.game_over,
            "result": self.result,
            "history": list(self.history),
        }


Game_session = GameSession

