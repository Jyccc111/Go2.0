import os
import io
import numpy as np

from tqdm import tqdm
from huggingface_hub import HfApi
from sgf_loader import sgf_loader


class ProgressBytesIO(io.BytesIO):
    """可被 huggingface_hub 上传，同时更新 tqdm 进度条的 BytesIO。"""

    def __init__(self):
        super().__init__()
        self.progress_bar = None

    def set_progress_bar(self, progress_bar):
        self.progress_bar = progress_bar
        self.progress_bar.n = self.tell()
        self.progress_bar.refresh()

    def read(self, size=-1):
        data = super().read(size)
        if self.progress_bar is not None:
            self.progress_bar.n = self.tell()
            self.progress_bar.refresh()
        return data

    def readinto(self, buffer):
        count = super().readinto(buffer)
        if self.progress_bar is not None:
            self.progress_bar.n = self.tell()
            self.progress_bar.refresh()
        return count

    def seek(self, offset, whence=io.SEEK_SET):
        position = super().seek(offset, whence)
        if self.progress_bar is not None:
            self.progress_bar.n = min(position, self.progress_bar.total)
            self.progress_bar.refresh()
        return position


def find_sgf_files(root):
    files = []
    for path, dirs, filenames in os.walk(root):
        for filename in filenames:
            if filename.endswith(".sgf"):
                files.append(os.path.join(path, filename))
    return files


ROOT = "/Users/jiangyuncong/Downloads/games"
REPO_ID = "Jycccc111/Go2.0_train_data"
CHUNK_SIZE = 100_000

api = HfApi()
api.create_repo(
    repo_id=REPO_ID,
    repo_type="dataset",
    exist_ok=True,
)

files = find_sgf_files(ROOT)
print("SGF数量:", len(files))

all_states = []
all_moves = []
all_results = []

chunk_id = 0
total_positions = 0

def upload_chunk():
    global chunk_id, total_positions

    # 先检查原始列表，避免转换后才发现问题
    n_states = len(all_states)
    n_moves = len(all_moves)
    n_results = len(all_results)

    if not (n_states == n_moves == n_results):
        print("\n数据长度不一致！")
        print("chunk_id:", chunk_id)
        print("states:", n_states)
        print("moves:", n_moves)
        print("results:", n_results)

        # 查看结果标签是否有 None
        none_count = sum(x is None for x in all_results)
        print("results 中 None 的数量:", none_count)

        raise ValueError("数据长度不一致，停止上传。")

    states = np.asarray(all_states, dtype=np.uint8)
    moves = np.asarray(all_moves, dtype=np.int16)
    results = np.asarray(all_results, dtype=np.int8)

    assert len(states) == len(moves) == len(results)

    # 后面的上传代码保持原样

    buffer = ProgressBytesIO()
    np.savez_compressed(
        buffer,
        states=states,
        moves=moves,
        results=results,
    )

    # 获取文件大小，然后从头开始上传。
    buffer_size = buffer.getbuffer().nbytes
    buffer.seek(0)

    filename = f"data/chunk_{chunk_id:05d}.npz"

    with tqdm(
        total=buffer_size,
        desc=f"上传 chunk {chunk_id}",
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
    ) as progress:
        buffer.set_progress_bar(progress)
        api.upload_file(
            path_or_fileobj=buffer,
            path_in_repo=filename,
            repo_id=REPO_ID,
            repo_type="dataset",
            commit_message=f"Upload {filename}",
        )

    total_positions += len(states)
    tqdm.write(
        f"已上传 chunk {chunk_id}: {len(states)} positions；"
        f"累计 {total_positions}"
    )

    chunk_id += 1
    all_states.clear()
    all_moves.clear()
    all_results.clear()


for filename in tqdm(files, desc="处理 SGF", unit="file"):
    loader = sgf_loader(filename)

    try:
        loader.load()
    except Exception as e:
        tqdm.write(f"跳过文件: {filename}；错误: {e}")
        continue

    all_states.extend(loader.states)
    all_moves.extend(loader.actions)
    all_results.extend(loader.results)

    if len(all_states) >= CHUNK_SIZE:
        upload_chunk()

if all_states:
    upload_chunk()

print("完成")
print("总 positions:", total_positions)
print(f"数据集地址: https://huggingface.co/datasets/{REPO_ID}")