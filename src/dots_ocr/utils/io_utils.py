

from fastapi import UploadFile
import os
import tempfile


def save_upload_to_temp(upload: UploadFile) -> str:
    _, ext = os.path.splitext(upload.filename or "")
    if not ext:
        ext = ".jpg"
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        tmp.write(upload.file.read())
        return tmp.name
