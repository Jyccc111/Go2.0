from copy import deepcopy
from functools import wraps
from math import isfinite
import os
import secrets
from flask import Flask, g, has_request_context, jsonify, render_template, request, session
from threading import Lock, RLock
from Game import Game, Player
from sgfmill import sgf, sgf_properties
from GameSession import GameSession, Type_player
from policy_models import list_policy_models, base_policy_choices
from web_i18n import TRANSLATIONS, normalize_language, translate
from web_sessions import BrowserSessionRegistry, LegacyState, SessionCapacityError
app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY') or secrets.token_hex(32)
app.config.update(
    ISOLATE_BROWSER_SESSIONS=os.environ.get('GO_ISOLATE_SESSIONS', '1' if os.environ.get('RENDER') else '0') == '1',
    SESSION_COOKIE_HTTPONLY=True, SESSION_COOKIE_SAMESITE='Lax',
    SESSION_COOKIE_SECURE=bool(os.environ.get('RENDER')),
    MAX_CONTENT_LENGTH=2 * 1024 * 1024,
)
game_session = GameSession()
game = game_session.game
sgf_replay = None
recommendation_ai = {}
value_predictor = None
value_predictor_lock = Lock()
session_lock = RLock()
legacy_state = LegacyState(globals())
browser_sessions = BrowserSessionRegistry(
    max_sessions=int(os.environ.get('GO_MAX_SESSIONS', '8')),
    ttl=float(os.environ.get('GO_SESSION_TTL', '3600')),
)


def runtime_state():
    return getattr(g, 'go_state', legacy_state) if has_request_context() else legacy_state


@app.before_request
def select_browser_game():
    if (not app.config['ISOLATE_BROWSER_SESSIONS'] or request.endpoint in (None, 'static', 'healthz')):
        return None
    old_key = session.get('go_session_id')
    try:
        key, state, created = browser_sessions.acquire(old_key)
    except SessionCapacityError as exc:
        return jsonify(success=False, ok=False, error=str(exc)), 503
    g.go_state, g.go_state_key = state, key
    session['go_session_id'] = key
    if old_key and created and request.path in ('/play', '/pass', '/ai_move'):
        return jsonify(success=False, error='Your game session expired. Refresh the page to start a new game.'), 409
    return None


@app.teardown_request
def release_browser_game(error=None):
    key = g.pop('go_state_key', None)
    if key:
        browser_sessions.release(key)


@app.errorhandler(413)
def upload_too_large(error):
    return jsonify(success=False, ok=False, error='SGF file is too large. Maximum upload size is 2 MB.'), 413


@app.route('/healthz', methods=['GET'])
def healthz():
    return jsonify(status='ok')


def lock_session(view):
    @wraps(view)
    def locked_view(*args, **kwargs):
        with session_lock:
            return view(*args, **kwargs)
    return locked_view


def get_recommendation_ai(color):
    """Use the requested side's chosen source, never the other side's different model."""

    selected = runtime_state().game_session.policy_model_for(color)
    session_ai = runtime_state().game_session.AI_black if color == Player.BLACK else runtime_state().game_session.AI_white
    if session_ai is not None:
        return session_ai
    other_ai = runtime_state().game_session.AI_white if color == Player.BLACK else runtime_state().game_session.AI_black
    if other_ai is not None and getattr(other_ai, 'policy_model', None) == selected:
        return other_ai
    if not isinstance(runtime_state().recommendation_ai, dict):
        runtime_state().recommendation_ai = {}
    if selected not in runtime_state().recommendation_ai:
        from AI import AI
        runtime_state().recommendation_ai[selected] = AI(policy_model=selected)
    return runtime_state().recommendation_ai[selected]


def get_value_predictor():
    global value_predictor
    if value_predictor is None:
        with value_predictor_lock:
            if value_predictor is None:
                from AI import ValuePredictor
                app.logger.info("Loading Go-Value checkpoint")
                value_predictor = ValuePredictor()
                app.logger.info("Go-Value checkpoint loaded")
    return value_predictor


def build_sgf_winrate_curve(moves):
    """Replay the main sequence without changing the visible SGF board."""
    from AI import ValuePredictor

    replay = Game()
    states = []

    for move_number, move in enumerate(moves, start=1):
        color = Player(move["color"])
        replay.current_player = color
        replay.update_legal_position()
        states.append(ValuePredictor.encode_board(replay, color.value))

        if move["pass"]:
            replay.is_pass()
            continue

        if not replay.play(move["row"], move["col"], color.value):
            raise ValueError(f"第 {move_number} 手在当前 Game 规则下为非法落子")

    final_color = -moves[-1]["color"] if moves else Player.BLACK.value
    replay.current_player = Player(final_color)
    replay.update_legal_position()
    states.append(ValuePredictor.encode_board(replay, final_color))

    app.logger.info("Predicting win rate for %s SGF positions", len(states))
    probabilities = get_value_predictor().predict_black_win_probabilities(states)
    app.logger.info("Finished SGF win-rate inference")
    return [
        {"move": move_number, "black_win_probability": probability}
        for move_number, probability in enumerate(probabilities)
    ]


def replay_payload(message=None):
    replay_game = runtime_state().sgf_replay["game"]
    return {
        "ok": True,
        "message": message,
        "filename": runtime_state().sgf_replay["filename"],
        "index": runtime_state().sgf_replay["index"],
        "total": len(runtime_state().sgf_replay["moves"]),
        "failed": runtime_state().sgf_replay["failed"],
        "result": runtime_state().sgf_replay["result"],
        "done": runtime_state().sgf_replay["index"] >= len(runtime_state().sgf_replay["moves"]),
        "board": replay_game.board.board.tolist(),
        "air": replay_game.air.board.tolist(),
        "cluster_air": replay_game.cluster_air.tolist(),
        "islegal": replay_game.islegal.tolist(),
        "num_eat": replay_game.num_eat.tolist(),
        "current_player": int(replay_game.current_player.value),
        "move": runtime_state().sgf_replay.get("last_move"),
    }

@app.route("/")
def index():
    language = normalize_language(request.args.get('lang') or request.cookies.get('go_language'))
    return render_template("show.html", language=language, translations=TRANSLATIONS,
                           tr=lambda message: translate(message, language))


@app.route('/sgf/state', methods=['GET'])
@lock_session
def sgf_state():
    """Read the existing replay without advancing it when changing UI language."""
    if runtime_state().sgf_replay is None:
        return jsonify(ok=False, error='请先载入 SGF 文件。'), 400
    return jsonify(replay_payload())

@app.route("/policy/models", methods=["GET"])
def policy_models_catalog():
    try:
        models = list_policy_models(refresh=request.args.get('refresh') == '1')
        return jsonify(success=True, models=models)
    except Exception:
        app.logger.exception('Could not read policy checkpoint catalog')
        return jsonify(success=False, models=base_policy_choices(),
                       error='暂时无法获取 epoch 列表，请检查网络后刷新。基础模型仍可选择。'), 503


@app.route("/state", methods=["GET"])
@lock_session
def get_state():
    return jsonify(runtime_state().game_session.get_state())


@app.route("/score", methods=["GET"])
@app.route("/sgf/score", methods=["GET"], defaults={"replay": True})
def score(replay=False):
    try:
        with session_lock:
            if replay:
                if runtime_state().sgf_replay is None:
                    return jsonify(success=False, error="请先载入 SGF 文件。"), 400
                scoring_game = deepcopy(runtime_state().sgf_replay["game"])
                if scoring_game.komi is None:
                    return jsonify(success=False, error="棋谱的贴目格式有误，无法点目。"), 400
                played = runtime_state().sgf_replay["index"] - int(runtime_state().sgf_replay["failed"])
                if played:
                    scoring_game.current_player = Player(-runtime_state().sgf_replay["moves"][played - 1]["color"])
            else:
                scoring_game = deepcopy(runtime_state().game_session.game)
                scoring_game.komi = runtime_state().game_session.komi
        black_score, white_area, _ = scoring_game.cal_result()
        white_score = white_area + scoring_game.komi
        lead = black_score - white_score
        winner = "black" if lead > 0 else "white" if lead < 0 else "draw"
        return jsonify(
            success=True,
            black_score=black_score,
            white_area=white_area,
            territory=scoring_game.eye_board.tolist(),
            scored_board=scoring_game.board.board.tolist(),
            white_score=white_score,
            komi=scoring_game.komi,
            winner=winner,
            margin=abs(lead),
        )
    except Exception:
        app.logger.exception("Could not score the current game")
        return jsonify(success=False, error="暂时无法计算当前盘面的目数。"), 500


@app.route("/session/init", methods=["POST"])
@lock_session
def initialize_session():

    data = request.get_json(silent=True) or {}
    try:
        state = runtime_state().game_session.initialize(
            data.get("player_black", Type_player.HUMAN),
            data.get("player_white", Type_player.HUMAN),
            data.get("komi", 0),
            policy_model=data.get("policy_model"),
            policy_model_black=data.get("policy_model_black"),
            policy_model_white=data.get("policy_model_white"),
        )
    except ValueError as exc:
        return jsonify({"success": False, "error": str(exc)}), 400
    except Exception as exc:
        return jsonify({"success": False, "error": f"Could not initialize players: {exc}"}), 503
    runtime_state().game = runtime_state().game_session.game
    runtime_state().recommendation_ai = {}
    response = {"success": True, **state}
    if runtime_state().game_session.player_type(runtime_state().game_session.next_player) == Type_player.AI:
        ai_outcome = runtime_state().game_session.play_ai()
        response.update(runtime_state().game_session.get_state())
        response["ai_move"] = ai_outcome.get("move")
        if not ai_outcome.get("success"):
            response["ai_error"] = ai_outcome.get("error")
    return jsonify(response)
@app.route("/play", methods=["POST"])
@lock_session
def play():
    data = request.get_json(silent=True) or {}
    try:
        outcome = runtime_state().game_session.play_human(
            data.get("row"), data.get("col"),
            data.get("color", runtime_state().game_session.next_player.value),
        )
    except ValueError as exc:
        return jsonify({"success": False, "error": str(exc)}), 400

    if not outcome["success"]:
        return jsonify({**outcome, **runtime_state().game_session.get_state()}), 400

    ai_outcome = None
    if not runtime_state().game_session.game_over and runtime_state().game_session.player_type(runtime_state().game_session.next_player) == Type_player.AI:
        ai_outcome = runtime_state().game_session.play_ai()
    response = {
        "success": True,
        **runtime_state().game_session.get_state(),
        "human_move": outcome.get("move"),
    }
    if ai_outcome is not None:
        response["ai_move"] = ai_outcome.get("move")
        if not ai_outcome.get("success"):
            response["ai_error"] = ai_outcome.get("error")
    return jsonify(response)


@app.route("/pass", methods=["POST"])
@lock_session
def pass_turn():
    data = request.get_json(silent=True) or {}
    try:
        outcome = runtime_state().game_session.pass_human(
            data.get("color", runtime_state().game_session.next_player.value)
        )
    except ValueError as exc:
        return jsonify({"success": False, "error": str(exc)}), 400
    if not outcome["success"]:
        return jsonify({**outcome, **runtime_state().game_session.get_state()}), 400
    if (outcome["success"] and not runtime_state().game_session.game_over
            and runtime_state().game_session.player_type(runtime_state().game_session.next_player) == Type_player.AI):
        ai_outcome = runtime_state().game_session.play_ai()
        outcome["ai_move"] = ai_outcome.get("move")
        if not ai_outcome.get("success"):
            outcome["ai_error"] = ai_outcome.get("error")
    return jsonify({**outcome, **runtime_state().game_session.get_state()})


@app.route("/ai_move", methods=["POST"])
@lock_session
def ai_move():
    outcome = runtime_state().game_session.play_ai()
    status = 200 if outcome.get("success") else 400
    return jsonify({**outcome, **runtime_state().game_session.get_state()}), status


@app.route("/ai/recommendations", methods=["GET"])
@lock_session
def ai_recommendations():
    if runtime_state().game_session.game_over:
        return jsonify({"success": False, "error": "对局已结束，无法生成推荐。"}), 400
    color = runtime_state().game_session.next_player
    try:
        ai = get_recommendation_ai(color)
        recommendations = ai.recommend_moves(runtime_state().game_session.game, color.value, top_k=5)
    except Exception as exc:
        return jsonify({"success": False, "error": f"生成 AI 推荐失败：{exc}"}), 503
    return jsonify({
        "success": True,
        "color": color.value,
        "recommendations": recommendations,
        **runtime_state().game_session.get_state(),
    })


@app.route("/value/prediction", methods=["GET"])
@lock_session
def predict_game_result():
    try:
        predictor = get_value_predictor()
        black_probability = predictor.predict_black_win_probability(
            runtime_state().game_session.game,
            runtime_state().game_session.next_player.value,
        )
    except Exception as exc:
        return jsonify({
            "success": False,
            "error": f"预测胜负失败：{exc}",
        }), 503

    white_probability = 1.0 - black_probability
    predicted_winner = "black" if black_probability >= 0.5 else "white"
    return jsonify({
        "success": True,
        "predicted_winner": predicted_winner,
        "black_win_probability": black_probability,
        "white_win_probability": white_probability,
        "next_player": runtime_state().game_session.next_player.value,
    })


@app.route("/sgf/prediction", methods=["GET"])
@lock_session
def predict_sgf_result():
    if runtime_state().sgf_replay is None:
        return jsonify({"success": False, "error": "请先载入 SGF 文件。"}), 400

    replay_game = runtime_state().sgf_replay["game"]
    next_index = runtime_state().sgf_replay["index"]
    moves = runtime_state().sgf_replay["moves"]
    if next_index < len(moves):
        color = moves[next_index]["color"]
    elif moves:
        color = -moves[-1]["color"]
    else:
        color = Player.BLACK.value

    try:
        replay_game.current_player = Player(color)
        replay_game.update_legal_position()
        predictor = get_value_predictor()
        black_probability = predictor.predict_black_win_probability(replay_game, color)
    except Exception as exc:
        return jsonify({
            "success": False,
            "error": f"预测棋谱局面失败：{exc}",
        }), 503

    return jsonify({
        "success": True,
        "predicted_winner": "black" if black_probability >= 0.5 else "white",
        "black_win_probability": black_probability,
        "white_win_probability": 1.0 - black_probability,
        "next_player": color,
    })


@app.route("/sgf/winrate-curve", methods=["GET"])
@lock_session
def sgf_winrate_curve():
    if runtime_state().sgf_replay is None:
        return jsonify({"success": False, "error": "请先载入 SGF 文件。"}), 400

    try:
        curve = runtime_state().sgf_replay.get("winrate_curve")
        if curve is None:
            curve = build_sgf_winrate_curve(runtime_state().sgf_replay["moves"])
            runtime_state().sgf_replay["winrate_curve"] = curve
    except Exception as exc:
        return jsonify({
            "success": False,
            "error": f"计算胜率曲线失败：{exc}",
        }), 503

    return jsonify({"success": True, "curve": curve})


@app.route("/sgf/load", methods=["POST"])
@lock_session
def load_sgf():


    uploaded = request.files.get("sgf")
    if uploaded is None or not uploaded.filename:
        return jsonify({"ok": False, "error": "请选择一个 SGF 文件。"}), 400

    try:
        parsed = sgf.Sgf_game.from_bytes(uploaded.read())
        root = parsed.get_root()
        raw_result = root.get_raw("RE") if "RE" in root.properties() else None
        result = raw_result.decode("utf-8", "replace") if raw_result else None
        size_raw = root.get_raw("SZ") if "SZ" in root.properties() else b"19"
        size = int(size_raw.decode("ascii"))
        if size != 19:
            return jsonify({"ok": False, "error": f"棋盘大小为 {size} 路；当前 Game 只支持 19 路。"}), 400

        moves = []
        for node_index, node in enumerate(parsed.get_main_sequence(), start=1):
            setup = [key for key in ("AB", "AW", "AE") if key in node.properties()]
            if setup:
                return jsonify({
                    "ok": False,
                    "error": f"SGF 第 {node_index} 个节点包含棋盘设置 {', '.join(setup)}；当前 Game 没有设置棋子的同步逻辑。",
                }), 400

            if "B" in node.properties():
                color, raw = 1, node.get_raw("B")
            elif "W" in node.properties():
                color, raw = -1, node.get_raw("W")
            else:
                continue

            if raw == b"":
                point = None
            else:
                point = sgf_properties.interpret_go_point(raw, size)
                if point is None:
                    return jsonify({"ok": False, "error": f"无法解析第 {node_index} 个节点的坐标 {raw!r}。"}), 400

            moves.append({
                "node": node_index,
                "color": color,
                "raw": raw.decode("ascii", "replace"),
                "row": point[0] if point is not None else None,
                "col": point[1] if point is not None else None,
                "pass": point is None,
            })
    except Exception as exc:
        return jsonify({"ok": False, "error": f"读取 SGF 失败: {exc}"}), 400

    replay_game = Game()
    try:
        komi = parsed.get_komi()
        replay_game.komi = komi if isfinite(komi) else None
    except ValueError:
        replay_game.komi = None

    runtime_state().sgf_replay = {
        "filename": uploaded.filename,
        "result": result,
        "game": replay_game,
        "moves": moves,
        "index": 0,
        "failed": False,
        "last_move": None,
        "winrate_curve": None,
    }
    return jsonify(replay_payload(f"已载入 {uploaded.filename}，共 {len(moves)} 手。"))


@app.route("/sgf/step", methods=["POST"])
@lock_session
def step_sgf():
    if runtime_state().sgf_replay is None:
        return jsonify({"ok": False, "error": "请先载入 SGF 文件。"}), 400
    if runtime_state().sgf_replay["failed"]:
        return jsonify(replay_payload("上一手非法，回放已暂停。"))
    if runtime_state().sgf_replay["index"] >= len(runtime_state().sgf_replay["moves"]):
        return jsonify(replay_payload("棋谱已播放完。"))

    move = runtime_state().sgf_replay["moves"][runtime_state().sgf_replay["index"]]
    replay_game = runtime_state().sgf_replay["game"]
    color = move["color"]
    color_name = "黑" if color == 1 else "白"
    replay_game.current_player = Player(color)
    replay_game.update_legal_position()

    if move["pass"]:
        replay_game.lasteated.fill(0)
        replay_game.is_pass()
        success = True
        reason = "pass"
    else:
        row, col = move["row"], move["col"]
        pre_move_legal = int(replay_game.islegal[row, col])
        pre_move_stone = int(replay_game.board.get(row, col))
        count = int(replay_game.count)
        black_eat = int(replay_game.num_eat[0,row,col])
        white_eat = int(replay_game.num_eat[1,row,col])
        success = replay_game.play(row, col, color)
        reason = None
        if not success:
            reason = {
                "kind": "illegal_move",
                "sgf_node": move["node"],
                "move_number": runtime_state().sgf_replay["index"] + 1,
                "color": color_name,
                "raw": move["raw"],
                "row": row,
                "col": col,
                "board_value_before": pre_move_stone,
                "islegal_before": pre_move_legal,
                "count": count,
                "black_eat": black_eat,
                "white_eat": white_eat,
                "neighbors": {
                    "up": int(replay_game.board.get(row - 1, col)) if row > 0 else None,
                    "down": int(replay_game.board.get(row + 1, col)) if row < 18 else None,
                    "left": int(replay_game.board.get(row, col - 1)) if col > 0 else None,
                    "right": int(replay_game.board.get(row, col + 1)) if col < 18 else None,
                },
                "neighbor_cluster_air": {
                    "up": int(replay_game.cluster_air[row - 1, col]) if row > 0 else None,
                    "down": int(replay_game.cluster_air[row + 1, col]) if row < 18 else None,
                    "left": int(replay_game.cluster_air[row, col - 1]) if col > 0 else None,
                    "right": int(replay_game.cluster_air[row, col + 1]) if col < 18 else None,
                },
            }
            runtime_state().sgf_replay["failed"] = True

    runtime_state().sgf_replay["index"] += 1
    runtime_state().sgf_replay["last_move"] = {
        **move,
        "color_name": color_name,
        "success": bool(success),
        "diagnostic": reason,
    }
    if move["pass"]:
        move_label = "pass"
    else:
        move_label = f"({move['row']}, {move['col']})"
    message = f"第 {move['node']} 个 SGF 节点：{color_name} {move_label}"
    if not success:
        message += " 落子失败，已暂停。"
    return jsonify(replay_payload(message))

if __name__ == "__main__":
    app.run(debug=True)
