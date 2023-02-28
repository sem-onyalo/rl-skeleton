import io
import os

import polars as pl

from .registry import Registry

class LocalRegistry(Registry):
    def __init__(self, eval_root:str, data_root:str) -> None:
        eval_root_local = f"{self.get_root_prefix(eval_root)}{eval_root}"
        data_root_local = f"{self.get_root_prefix(data_root)}{data_root}"
        super().__init__(eval_root_local, data_root_local)

    def read_training_data(self, path:str, filename:str, separator=",") -> pl.DataFrame:
        model_data_file_path = os.path.join(self.data_root, path, filename)
        assert os.path.exists(model_data_file_path), f"Could not find file {model_data_file_path}"
        return pl.read_csv(model_data_file_path, sep=separator, infer_schema_length=10000)

    def write_bytes(self, path:str, buffer:io.BytesIO, filename:str, root:str=None) -> None:
        root_path = root if root != None else self.eval_root
        root_path = root_path if root_path[0] == "." else f".{root_path}"
        dir_path = os.path.join(root_path, path)
        os.makedirs(dir_path, exist_ok=True)

        file_path = os.path.join(dir_path, filename)
        with open(file_path, mode="wb") as fd:
            fd.write(buffer.getvalue())

    def read_bytes(self, path:str, filename:str, root:str=None) -> io.BytesIO:
        root_path = root if root != None else self.eval_root
        file_path = os.path.join(root_path, path, filename)
        assert os.path.exists(file_path), f"Error: the file path {file_path} could not be found"
        buffer = open(file_path, mode="rb")
        return buffer

    def get_root_prefix(self, root:str) -> str:
        return "./" if root[0] != "." and root[0] != "/" else ""
