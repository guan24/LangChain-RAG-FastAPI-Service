from dataclasses import dataclass, asdict
from typing import Optional, Any
import json

EVENT_RESPONSE = "response"
EVENT_ERROR = "error"
EVENT_DONE = "done"


@dataclass
class SSEEvent:
    """事件格式"""

    event_type: str  # 事件类型：start/error/done/slicing_completed/writing/completed
    message: str  # 人类可读的消息
    total_files: int = 0  # 总文件数
    file_index: Optional[int] = None  # 当前处理的文件索引
    filename: Optional[str] = None  # 文件名
    step: Optional[str] = None  # 处理步骤：validation/slicing/writing/completed
    progress: int = 0  # 进度百分比 (0-100)
    success_count: int = 0  # 成功处理的文件数
    failed_count: int = 0  # 失败的文件数
    slice_success_count: int = 0  # 成功切片的文件数
    error_message: Optional[str] = None  # 错误详情
    chunk_count: Optional[int] = None  # 切片数量

    def to_sse(self) -> str:
        payload = {k: v for k, v in asdict(self).items() if v is not None}
        return f"event: progress\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"


class SliceResult:
    """切片结果格式"""

    def __init__(self):
        self.file_index: int = 0  # 文件索引
        self.filename: str = ""  # 文件名
        self.documents: list = []  # 切分后的文档列表
        self.md5: str = ""  # 文件的MD5值
        self.success: bool = False  # 是否成功
        self.error: Optional[str] = None  # 错误信息
        self.chunk_count: int = 0  # 切片数量

    @classmethod
    def success_result(
        cls, file_index: int, filename: str, documents: list, md5: str
    ) -> "SliceResult":
        result = cls()
        result.file_index = file_index
        result.filename = filename
        result.documents = documents
        result.md5 = md5
        result.success = True
        result.chunk_count = len(documents)
        return result

    @classmethod
    def error_result(cls, file_index: int, filename: str, error: str) -> "SliceResult":
        result = cls()
        result.file_index = file_index
        result.filename = filename
        result.success = False
        result.error = error
        return result

    def to_dict(self) -> dict:
        return {
            "file_index": self.file_index,
            "filename": self.filename,
            "documents": self.documents,
            "md5": self.md5,
            "success": self.success,
            "error": self.error,
            "chunk_count": self.chunk_count,
        }
