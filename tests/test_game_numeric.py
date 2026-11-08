"""Large-board numeric regressions without model downloads or training."""
import unittest
import numpy as np

from Game import Game, Player
from rl.game import encode_board


def neighbours(row, col):
    for r, c in ((row - 1, col), (row + 1, col), (row, col - 1), (row, col + 1)):
        if 0 <= r < 19 and 0 <= c < 19:
            yield r, c


def install_position(game, board):
    """Build coherent cached liberties for a test setup, not a new game rule."""
    game.board.board[:] = board
    game.cluster_air.fill(0)
    game.air.board.fill(0)
    for row, col in np.argwhere(board == 0):
        game.air.board[row, col] = sum(board[r, c] == 0 for r, c in neighbours(row, col))
    seen = set()
    for row, col in np.argwhere(board != 0):
        row, col = int(row), int(col)
        if (row, col) in seen:
            continue
        color = int(board[row, col])
        pending, group, liberties = [(row, col)], [], set()
        seen.add((row, col))
        while pending:
            r, c = pending.pop()
            group.append((r, c))
            for u, v in neighbours(r, c):
                if board[u, v] == 0:
                    liberties.add((u, v))
                elif board[u, v] == color and (u, v) not in seen:
                    seen.add((u, v))
                    pending.append((u, v))
        for r, c in group:
            game.cluster_air[r, c] = len(liberties)
    game.current_player = Player.BLACK
    game.update_legal_position()


class GameNumericTests(unittest.TestCase):
    def test_real_move_records_139_stone_capture_and_encoder_preserves_it(self):
        # A 140-point rectangle with two holes contains a connected 138-stone
        # black group with two liberties. Playing one hole makes 139 stones
        # with exactly one liberty: the original crash at Game.py line 464.
        board = np.zeros((19, 19), dtype=np.int8)
        board[1:11, 1:15] = 1
        board[0, :16] = board[11, :16] = -1
        board[1:11, 0] = board[1:11, 15] = -1
        board[5, 7] = board[5, 8] = 0
        game = Game()
        install_position(game, board)
        self.assertEqual(np.count_nonzero(game.board.board == 1), 138)
        self.assertEqual(game.cluster_air[1, 1], 2)
        self.assertEqual(game.islegal[5, 8], 1)
        self.assertTrue(game.play(5, 8, 1))
        self.assertEqual(np.count_nonzero(game.board.board == 1), 139)
        self.assertEqual(game.cluster_air[1, 1], 1)
        self.assertEqual(game.num_eat[1, 5, 7], 139)
        white_captures, black_captures = game.get_capture(-1)
        self.assertEqual(white_captures[5, 7], 139)
        self.assertEqual(black_captures[5, 7], 0)
        black_captures, white_captures = game.get_capture(1)
        self.assertEqual(white_captures[5, 7], 139)
        state = encode_board(game, -1)
        self.assertEqual(state.dtype, np.int16)
        self.assertEqual(state[5, 5, 7], 139)
        self.assertEqual(state[6, 5, 7], 0)

    def test_connected_comb_can_have_more_than_127_liberties(self):
        game = Game()
        # All odd rows are empty corridors; column 0 joins the even-row teeth.
        game.board.board[:, 0] = 1
        game.board.board[::2, :18] = 1
        liberties = game.calculate_cluster_air(0, 0, 0, 1)
        self.assertEqual(liberties, 163)
        game.is_checked.fill(0)
        game.update_cluster_air(0, 0, liberties, 1)
        self.assertTrue(np.all(game.cluster_air[game.board.board == 1] == 163))
        self.assertEqual(encode_board(game, 1)[4, 0, 0], 163)

    def test_relation_indices_support_more_than_127_stone_and_empty_groups(self):
        game = Game()
        rows, cols = np.indices((19, 19))
        game.board.board[:] = ((rows + cols) % 2 == 0).astype(np.int8)
        game.create_clusters()
        self.assertEqual(len(game.black_clusters), 181)
        self.assertEqual(len(game.air_clusters), 180)
        game.construct_relation_mat()
        self.assertEqual(game.relation_mat.shape, (181, 180))
        stone_indices = game.cluster_indexing[game.board.board != 0]
        empty_indices = game.cluster_indexing[game.board.board == 0]
        self.assertEqual(int(stone_indices.max()), 180)
        self.assertEqual(int(empty_indices.max()), 179)
        self.assertEqual(len(set(stone_indices.tolist())), 181)
        self.assertTrue(np.all(stone_indices >= 0))
        self.assertTrue(np.all(empty_indices >= 0))

    def test_alive_relation_dot_products_do_not_wrap_at_128(self):
        game = Game()
        # Isolate the original relation-propagation arithmetic: 130 seeded
        # groups sharing a region must stay alive; int8 dot wraps 130 to -126.
        # Allocate matrices through the production method to exercise its dtype.
        game.black_clusters = [[] for _ in range(130)]
        game.air_clusters = [[]]
        game.construct_relation_mat()
        game.relation_mat.fill(1)
        game.black_num_eyes.fill(2)
        self.assertTrue(game.cal_alive())
        np.testing.assert_array_equal(game.black_alive_indexes, np.ones(130))


if __name__ == '__main__':
    unittest.main()
