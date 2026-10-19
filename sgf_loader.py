from sgfmill import sgf
from sgfmill import sgf_properties
from Game import Game, Player
import numpy as np

class sgf_loader():
    def __init__(self, filename):
        self.filename = filename
        self.game = None
        self.size = 19
        self.result = None
        self.result = None
        self.moves = []
        self.komi = None
        self.handicap = None
        self.lasteated1 = []
        self.lasteated2 = []
        self.states = []
        self.actions = []
        self.results = []

    def print_board(self):
        board = self.game.board.board
        symbols = {0: ".", 1: "B", -1: "W"}

        print("    " + " ".join(f"{c:2}" for c in range(19)))
        for r in range(19):
            line = " ".join(symbols[int(board[r, c])] for c in range(19))
            print(f"{r:2}  {line}")

    def encode_result(self):
        if self.result is None:
            return None

        if self.result.startswith("B+"):
            return 1

        if self.result.startswith("W+"):
                return 0

        return None
    def encode_board(self, color):

        size = self.size

        x = np.zeros(
            (8, size, size),
            dtype=np.int16
        )

        x[0,:,:] = self.game.get_black()
        x[1,:,:] = self.game.get_white()
        x[2,:,:] = self.game.turn_since
        x[3,:,:] = self.game.air.board
        x[4,:,:] = self.game.cluster_air
        x[5,:,:], x[6,:,:] = self.game.get_capture(color)
        x[7,:,:] = self.game.islegal
        return x
    
    def load(self):
        with open(self.filename, "rb") as f:
            sgf_game = sgf.Sgf_game.from_bytes(f.read())
        root = sgf_game.get_root()
        if "SZ" in root.properties():

            size = root.get_raw("SZ")

            self.size = int(size)

        else:

            self.size = 19

        if self.size is not None:
            self.size = int(self.size)
        else:
            self.size = 19
        if self.size != 19:
            print(
                "跳过非19路:",
                self.filename,
                self.size
            )
            return self
        
        self.game = Game()

        if "RE" in root.properties():
            self.result = root.get_raw("RE").decode()
        else:
            self.result = None
            print("跳过没有胜负结果的棋谱:", self.filename)
            return self

        result = self.encode_result()

# 只保留能够确定黑棋或白棋获胜的棋谱
        if result is None:
            print("跳过无有效胜负结果的棋谱:", self.filename, self.result)
            return self
        self.komi = (
            root.get_raw("KM")
            if "KM" in root.properties()
            else None
        )

        self.handicap = (
            root.get_raw("HA")
            if "HA" in root.properties()
            else None
        )
        if self.handicap != None:
            print(
                "跳过让子棋:",
                self.filename,
                self.handicap
                )
            return self

        for node_index, node in enumerate(sgf_game.get_main_sequence()):
            if "AB" in node.properties():

                for x, y in node.get("AB"):
                    self.game.add_stone(
                        x,
                        y,
                        1
                    )

                # 2. 正常棋步
            color = None
            move = None

            if "B" in node.properties():

                color = "b"

                raw = node.get_raw("B")


            elif "W" in node.properties():

                color = "w"

                raw = node.get_raw("W")


            else:

                continue
            stone = 1 if color == "b" else -1
            # pass
            if raw == b"":
                state = self.encode_board(stone)
                self.states.append(state)
                self.actions.append(self.size*self.size
                self.results.append(result)
                self.game.is_pass()
                continue
            try:
                move = sgf_properties.interpret_go_point(
                    raw,
                    self.size
                )

            except ValueError as e:
                raise ValueError(
                    f"无法解析 SGF 棋步: {raw!r}; "
                    f"文件: {self.filename}; "
                    f"节点: {node_index + 1}"
                ) from e

            if move is None:
                continue
            row, col = move

            # 只尝试落子一次
            success = self.game.play(row, col, stone)

            if not success:
                board = self.game.board.board
                current_stone = int(board[row, col])
                symbols = {0: ".", 1: "B", -1: "W"}

                print("\n" + "=" * 70)
                print("SGF 落子失败")
                print("文件:", self.filename)
                print("SGF 节点序号:", node_index + 1)
                print("Game 成功落子数:", self.game.num_turns)
                print("棋手:", "黑" if stone == 1 else "白")
                print("SGF 原始坐标:", raw)
                print(f"转换坐标: row={row}, col={col}")
                print("目标点当前棋子:", symbols[current_stone])
                print("目标点 islegal:", int(self.game.islegal[row, col]))
                print("Game.current_player:", int(self.game.current_player))
                print("落子前棋盘:")
                self.print_board()
                print("=" * 70)

                raise RuntimeError("遇到非法棋步，已停止读取棋谱")
            state = self.encode_board(stone)
            self.states.append(state)
            action = row * self.size + col
            self.actions.append(action)
            self.results.append(result)
        # 只给以认输结束的棋谱添加投降样本
        if self.result and self.result.upper() in ("B+R", "W+R"):
            loser = -1 if self.result.upper() == "B+R" else 1
            if loser == 1:
                self.game.current_player = Player.BLACK
            else:
                self.game.current_player = Player.WHITE
            self.game.update_legal_position()
            self.states.append(self.encode_board(loser))
            self.actions.append(self.size * self.size + 1)  # 362：resign
            self.results.append(result)

if __name__ == "__main__":

    sgf_file = "/Users/jiangyuncong/Downloads/games/Kisei/23/F4a.sgf"


    loader = sgf_loader(sgf_file)

    loader.load()

    print(
        np.array(loader.states).shape
    )

    print(
        np.array(loader.actions).shape
    )

    print(
        np.array(loader.results).shape
    )
    

    print("棋盘大小:", loader.size)
    print("结果:", loader.result)
    print("贴目:", loader.komi)
    print("让子:", loader.handicap)
