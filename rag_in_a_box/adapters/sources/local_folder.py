from __future__ import annotations

import mimetypes
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from rag_in_a_box.core.models import SourceItem


@dataclass
class LocalFolderSource:
    root: str
    recursive: bool = True

    def iter_items(self) -> Iterable[SourceItem]:
        base = Path(self.root)
        if not base.exists():
            raise FileNotFoundError(f"LocalFolderSource root does not exist: {self.root}")

        pattern = "**/*" if self.recursive else "*"
        for path in base.glob(pattern):
            if not path.is_file():
                continue

            mime, _ = mimetypes.guess_type(str(path))
            mime_type = mime or "application/octet-stream"

            data = path.read_bytes()

            yield SourceItem(
                uri=str(path.resolve()),
                data=data,
                mime_type=mime_type,
                metadata={
                    "source": "local_folder",
                    "filename": path.name,
                },
            )
