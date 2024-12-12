import os
from pathlib import Path

from vit_prisma.vjepa_hf.src.utils.logging import get_logger


class LocalFileDataset:
    """An iterator over local video files, input as a list of file paths."""

    def __init__(
        self,
        data_path: str,
        offset: int = 0,
        limit: int = -1,
        ext: str = ".mp4",
        id_key: str = "video_id",
        data_key: str = "video",
    ):
        self.logger = get_logger(self.__class__.__name__)
        self.data_path = data_path
        if offset < 0:
            raise ValueError("offset cannot be negative")
        self.offset = offset
        self.limit = limit
        self.flist = sorted(Path(data_path).rglob(f"*{ext}"))
        self.id_key = id_key
        self.data_key = data_key

        if self.limit > 0:
            self.flist = self.flist[self.offset : self.offset + self.limit]
        else:
            self.flist = self.flist[self.offset :]

    def __len__(self):
        return len(self.flist)

    def __getitem__(self, idx):
        local_file_path = str(self.flist[idx].resolve())
        video_id = os.path.splitext(os.path.basename(local_file_path))[0]
        with open(local_file_path, "rb") as f:
            data = f.read()
        # Iterator format should be the same as for AIRStore
        return {self.id_key: video_id, self.data_key: data}
