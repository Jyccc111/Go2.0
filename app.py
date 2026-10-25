from flask import Flask, jsonify, render_template, request
from Game import Game, Player
from sgfmill import sgf, sgf_properties
from GameSession import GameSession, Type_player
app = Flask(__name__)
game_session = GameSession()
game = game_session.game
sgf_replay = None
recommendation_ai = None


def get_recommendation_ai(color):
    """Reuse a session AI or lazily load one for human-vs-human games."""
    global recommendation_ai
    session_ai = (
        game_session.AI_black if color == Player.BLACK
        else game_session.AI_white
    )
    if session_ai is not None:
        return session_ai
    other_ai = game_session.AI_white if color == Player.BLACK else game_session.AI_black
    if other_ai is not None:
        return other_ai
    if recommendation_ai is None:
        from AI import AI
        recommendation_ai = AI()
    return recommendation_ai


def replay_payload(message=None):
    replay_game = sgf_replay["game"]
    return {
        "ok": True,
        "message": message,
        "filename": sgf_replay["filename"],
        "index": sgf_replay["index"],
        "total": len(sgf_replay["moves"]),
        "failed": sgf_replay["failed"],
        "done": sgf_replay["index"] >= len(sgf_replay["moves"]),
        "board": replay_game.board.board.tolist(),
        "air": replay_game.air.board.tolist(),
        "cluster_air": replay_game.cluster_air.tolist(),
        "islegal": replay_game.islegal.tolist(),
        "num_eat": replay_game.num_eat.tolist(),
        "current_player": int(replay_game.current_player.value),
        "move": sgf_replay.get("last_move"),
    }

@app.route("/")
def index():
    return render_template("show.html")

@app.route("/state", methods=["GET"])
def get_state():
    return jsonify(game_session.get_state())


@app.route("/session/init", methods=["POST"])
def initialize_session():
    global game
    data = request.get_json(silent=True) or {}
    try:
        state = game_session.initialize(
            data.get("player_black", Type_player.HUMAN),
            data.get("player_white", Type_player.HUMAN),
            data.get("komi", 0),
        )
    except ValueError as exc:
        return jsonify({"success": False, "error": str(exc)}), 400
    except Exception as exc:
        return jsonify({"success": False, "error": f"Could not initialize players: {exc}"}), 503
    game = game_session.game
    response = {"success": True, **state}
    if game_session.player_type(game_session.next_player) == Type_player.AI:
        ai_outcome = game_session.play_ai()
        response.update(game_session.get_state())
        response["ai_move"] = ai_outcome.get("move")
        if not ai_outcome.get("success"):
            response["ai_error"] = ai_outcome.get("error")
    return jsonify(response)
@app.route("/play", methods=["POST"])
def play():
    data = request.get_json(silent=True) or {}
    try:
        outcome = game_session.play_human(
            data.get("row"), data.get("col"),
            data.get("color", game_session.next_player.value),
        )
    except ValueError as exc:
        return jsonify({"success": False, "error": str(exc)}), 400

    if not outcome["success"]:
        return jsonify({**outcome, **game_session.get_state()}), 400

    ai_outcome = None
    if not game_session.game_over and game_session.player_type(game_session.next_player) == Type_player.AI:
        ai_outcome = game_session.play_ai()
    response = {
        "success": True,
        **game_session.get_state(),
        "human_move": outcome.get("move"),
    }
    if ai_outcome is not None:
        response["ai_move"] = ai_outcome.get("move")
        if not ai_outcome.get("success"):
            response["ai_error"] = ai_outcome.get("error")
    return jsonify(response)


@app.route("/pass", methods=["POST"])
def pass_turn():
    data = request.get_json(silent=True) or {}
    try:
        outcome = game_session.pass_human(
            data.get("color", game_session.next_player.value)
        )
    except ValueError as exc:
        return jsonify({"success": False, "error": str(exc)}), 400
    if not outcome["success"]:
        return jsonify({**outcome, **game_session.get_state()}), 400
    if (outcome["success"] and not game_session.game_over
            and game_session.player_type(game_session.next_player) == Type_player.AI):
        ai_outcome = game_session.play_ai()
        outcome["ai_move"] = ai_outcome.get("move")
        if not ai_outcome.get("success"):
            outcome["ai_error"] = ai_outcome.get("error")
    return jsonify({**outcome, **game_session.get_state()})


@app.route("/ai_move", methods=["POST"])
def ai_move():
    outcome = game_session.play_ai()
    status = 200 if outcome.get("success") else 400
    return jsonify({**outcome, **game_session.get_state()}), status


@app.route("/ai/recommendations", methods=["GET"])
def ai_recommendations():
    if game_session.game_over:
        return jsonify({"success": False, "error": "对局已结束，无法生成推荐。"}), 400
    color = game_session.next_player
    try:
        ai = get_recommendation_ai(color)
        recommendations = ai.recommend_moves(game_session.game, color.value, top_k=5)
    except Exception as exc:
        return jsonify({"success": False, "error": f"生成 AI 推荐失败：{exc}"}), 503
    return jsonify({
        "success": True,
        "color": color.value,
        "recommendations": recommendations,
        **game_session.get_state(),
    })


@app.route("/sgf/load", methods=["POST"])
def load_sgf():
    global sgf_replay

    uploaded = request.files.get("sgf")
    if uploaded is None or not uploaded.filename:
        return jsonify({"ok": False, "error": "请选择一个 SGF 文件。"}), 400

    try:
        parsed = sgf.Sgf_game.from_bytes(uploaded.read())
        root = parsed.get_root()
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

    sgf_replay = {
        "filename": uploaded.filename,
        "game": Game(),
        "moves": moves,
        "index": 0,
        "failed": False,
        "last_move": None,
    }
    return jsonify(replay_payload(f"已载入 {uploaded.filename}，共 {len(moves)} 手。"))


@app.route("/sgf/step", methods=["POST"])
def step_sgf():
    if sgf_replay is None:
        return jsonify({"ok": False, "error": "请先载入 SGF 文件。"}), 400
    if sgf_replay["failed"]:
        return jsonify(replay_payload("上一手非法，回放已暂停。"))
    if sgf_replay["index"] >= len(sgf_replay["moves"]):
        return jsonify(replay_payload("棋谱已播放完。"))

    move = sgf_replay["moves"][sgf_replay["index"]]
    replay_game = sgf_replay["game"]
    color = move["color"]
    color_name = "黑" if color == 1 else "白"

    if move["pass"]:
        # current_player 表示刚刚行动的一方；更新合法点表时取其对手。
        replay_game.current_player = Player(color)
        replay_game.lasteated.fill(0)
        replay_game.update_legal_position()
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
                "move_number": sgf_replay["index"] + 1,
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
            sgf_replay["failed"] = True

    sgf_replay["index"] += 1
    sgf_replay["last_move"] = {
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
