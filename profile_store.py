import hashlib
import hmac
import os
import secrets
import sqlite3
import time
from contextlib import closing


_TYPE_SQL = {"int": "INTEGER", "float": "REAL", "str": "TEXT"}


class ProfileAccessDenied(PermissionError):
    """表示档案访问令牌缺失或不匹配。"""


def _token_hash(token: str) -> str:
    return hashlib.sha256((token or "").encode("utf-8")).hexdigest()


def _verify_token(provided: str, stored_hash: str) -> bool:
    if not provided or not stored_hash:
        return False
    return hmac.compare_digest(_token_hash(provided), stored_hash)


def _allow_legacy_profiles() -> bool:
    return os.environ.get("RAG_ALLOW_LEGACY_PROFILES", "").strip().lower() in {
        "1",
        "true",
        "yes",
    }


def _connect(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def _ensure_schema(conn: sqlite3.Connection, profile_fields) -> None:
    columns = ",\n                ".join(
        "%s %s" % (key, _TYPE_SQL.get(field_type, "TEXT"))
        for key, _label, field_type in profile_fields
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS profiles (
            user_id TEXT PRIMARY KEY,
            %s,
            updated_at TEXT,
            access_token_hash TEXT
        )
        """
        % columns
    )
    existing = {row[1] for row in conn.execute("PRAGMA table_info(profiles)")}
    for key, _label, field_type in profile_fields:
        if key not in existing:
            conn.execute(
                "ALTER TABLE profiles ADD COLUMN %s %s"
                % (key, _TYPE_SQL.get(field_type, "TEXT"))
            )
    if "access_token_hash" not in existing:
        conn.execute("ALTER TABLE profiles ADD COLUMN access_token_hash TEXT")


def init_profile_db(db_path: str, profile_fields) -> None:
    """创建或升级健康档案表。"""
    with closing(_connect(db_path)) as conn:
        with conn:
            _ensure_schema(conn, profile_fields)


def save_profile_record(
    db_path: str,
    profile_fields,
    user_id: str,
    data: dict,
    access_token: str = "",
) -> tuple[str, bool, bool]:
    """保存档案并返回访问令牌、创建状态和令牌生成状态。"""
    allowed = {key for key, _label, _field_type in profile_fields}
    clean_data = {
        key: value
        for key, value in data.items()
        if key in allowed and value is not None
    }

    with closing(_connect(db_path)) as conn:
        with conn:
            _ensure_schema(conn, profile_fields)
            row = conn.execute(
                "SELECT access_token_hash FROM profiles WHERE user_id = ?",
                (user_id,),
            ).fetchone()
            created = row is None
            generated = False

            if row and row["access_token_hash"]:
                if not _verify_token(access_token, row["access_token_hash"]):
                    raise ProfileAccessDenied("档案访问令牌无效")
                raw_token = access_token
                token_hash = row["access_token_hash"]
            else:
                if row and not _allow_legacy_profiles():
                    raise ProfileAccessDenied(
                        "现有档案尚未绑定访问令牌，请先由管理员迁移"
                    )
                raw_token = access_token
                if not raw_token:
                    raw_token = secrets.token_urlsafe(32)
                    generated = True
                token_hash = _token_hash(raw_token)

            columns = ["user_id"] + list(clean_data.keys()) + [
                "updated_at",
                "access_token_hash",
            ]
            values = [user_id] + list(clean_data.values()) + [
                time.strftime("%Y-%m-%d %H:%M:%S"),
                token_hash,
            ]
            updates = ", ".join(
                "%s=excluded.%s" % (column, column)
                for column in list(clean_data.keys())
                + ["updated_at", "access_token_hash"]
            )
            sql = (
                "INSERT INTO profiles (%s) VALUES (%s) "
                "ON CONFLICT(user_id) DO UPDATE SET %s"
                % (
                    ", ".join(columns),
                    ", ".join("?" * len(columns)),
                    updates,
                )
            )
            conn.execute(sql, values)
    return raw_token, created, generated


def load_profile_record(
    db_path: str,
    profile_fields,
    user_id: str,
    access_token: str = "",
) -> dict:
    """读取通过访问令牌校验的健康档案。"""
    if not user_id:
        return {}
    with closing(_connect(db_path)) as conn:
        _ensure_schema(conn, profile_fields)
        row = conn.execute(
            "SELECT * FROM profiles WHERE user_id = ?",
            (str(user_id),),
        ).fetchone()
    if not row:
        return {}

    stored_hash = row["access_token_hash"]
    if stored_hash:
        if not _verify_token(access_token, stored_hash):
            raise ProfileAccessDenied("档案访问令牌无效")
    elif not _allow_legacy_profiles():
        raise ProfileAccessDenied("现有档案尚未绑定访问令牌，请先由管理员迁移")

    excluded = {"user_id", "updated_at", "access_token_hash"}
    return {key: row[key] for key in row.keys() if key not in excluded}


def delete_profile_record(
    db_path: str,
    profile_fields,
    user_id: str,
    access_token: str = "",
) -> bool:
    """删除通过访问令牌校验的健康档案。"""
    with closing(_connect(db_path)) as conn:
        with conn:
            _ensure_schema(conn, profile_fields)
            row = conn.execute(
                "SELECT access_token_hash FROM profiles WHERE user_id = ?",
                (str(user_id),),
            ).fetchone()
            if not row:
                return False
            stored_hash = row["access_token_hash"]
            if stored_hash:
                if not _verify_token(access_token, stored_hash):
                    raise ProfileAccessDenied("档案访问令牌无效")
            elif not _allow_legacy_profiles():
                raise ProfileAccessDenied(
                    "现有档案尚未绑定访问令牌，请先由管理员迁移"
                )
            conn.execute(
                "DELETE FROM profiles WHERE user_id = ?",
                (str(user_id),),
            )
    return True


def list_profile_ids(db_path: str, profile_fields) -> list[dict]:
    """列出档案编号和更新时间，不返回健康内容。"""
    with closing(_connect(db_path)) as conn:
        _ensure_schema(conn, profile_fields)
        rows = conn.execute(
            "SELECT user_id, updated_at FROM profiles "
            "ORDER BY updated_at DESC"
        ).fetchall()
    return [
        {"user_id": row["user_id"], "updated_at": row["updated_at"]}
        for row in rows
    ]
