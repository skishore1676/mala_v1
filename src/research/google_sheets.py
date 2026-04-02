"""Minimal Google Sheets integration for bias routing."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any


_SHEET_ID_RE = re.compile(r"/d/(?P<sheet_id>[a-zA-Z0-9-_]+)")


def spreadsheet_id_from_url(url_or_id: str) -> str:
    match = _SHEET_ID_RE.search(url_or_id)
    if match is not None:
        return match.group("sheet_id")
    return url_or_id.strip()


@dataclass(slots=True)
class GoogleSheetTableClient:
    spreadsheet_id: str
    sheet_name: str
    credentials_path: Path
    service: Any | None = None

    def __post_init__(self) -> None:
        self.spreadsheet_id = spreadsheet_id_from_url(self.spreadsheet_id)
        self.credentials_path = Path(self.credentials_path).expanduser().resolve()
        if self.service is None:
            self.service = self._build_service()

    def read_rows(self, *, range_suffix: str = "A1:Z1000") -> list[dict[str, Any]]:
        result = (
            self.service.spreadsheets()
            .values()
            .get(
                spreadsheetId=self.spreadsheet_id,
                range=f"{self.sheet_name}!{range_suffix}",
            )
            .execute()
        )
        values = result.get("values", [])
        if not values or len(values) < 2:
            return []
        headers = [str(header).strip() for header in values[0]]
        rows: list[dict[str, Any]] = []
        for index, row in enumerate(values[1:], start=2):
            padded = list(row) + [""] * (len(headers) - len(row))
            payload = dict(zip(headers, padded, strict=False))
            payload["row_index"] = index
            rows.append(payload)
        return rows

    def batch_update_rows(
        self,
        *,
        rows: list[dict[str, Any]],
        columns: list[str],
    ) -> dict[str, Any]:
        if not rows:
            return {}
        header_map = self._header_column_map()
        data: list[dict[str, Any]] = []
        for row in rows:
            row_index = int(row["row_index"])
            for column in columns:
                column_letter = header_map.get(column)
                if column_letter is None:
                    continue
                data.append(
                    {
                        "range": f"{self.sheet_name}!{column_letter}{row_index}",
                        "values": [[row.get(column, "")]],
                    }
                )
        if not data:
            return {}
        return (
            self.service.spreadsheets()
            .values()
            .batchUpdate(
                spreadsheetId=self.spreadsheet_id,
                body={
                    "valueInputOption": "USER_ENTERED",
                    "data": data,
                },
            )
            .execute()
        )

    def overwrite_table(
        self,
        *,
        headers: list[str],
        rows: list[dict[str, Any]],
        clear_range_suffix: str = "A1:ZZ5000",
    ) -> dict[str, Any]:
        self.ensure_sheet_exists()
        self.service.spreadsheets().values().clear(
            spreadsheetId=self.spreadsheet_id,
            range=f"{self.sheet_name}!{clear_range_suffix}",
            body={},
        ).execute()
        values = [headers]
        for row in rows:
            values.append([row.get(header, "") for header in headers])
        return (
            self.service.spreadsheets()
            .values()
            .update(
                spreadsheetId=self.spreadsheet_id,
                range=f"{self.sheet_name}!A1",
                valueInputOption="USER_ENTERED",
                body={"values": values},
            )
            .execute()
        )

    def ensure_sheet_exists(self) -> None:
        metadata = (
            self.service.spreadsheets()
            .get(spreadsheetId=self.spreadsheet_id)
            .execute()
        )
        sheets = metadata.get("sheets", [])
        for sheet in sheets:
            props = sheet.get("properties", {})
            if str(props.get("title", "")).strip() == self.sheet_name:
                return
        (
            self.service.spreadsheets()
            .batchUpdate(
                spreadsheetId=self.spreadsheet_id,
                body={
                    "requests": [
                        {
                            "addSheet": {
                                "properties": {
                                    "title": self.sheet_name,
                                }
                            }
                        }
                    ]
                },
            )
            .execute()
        )

    def _header_column_map(self) -> dict[str, str]:
        result = (
            self.service.spreadsheets()
            .values()
            .get(
                spreadsheetId=self.spreadsheet_id,
                range=f"{self.sheet_name}!1:1",
            )
            .execute()
        )
        values = result.get("values", [])
        headers = values[0] if values else []
        return {
            str(header).strip(): _column_letter(index + 1)
            for index, header in enumerate(headers)
            if str(header).strip()
        }

    def _build_service(self) -> Any:
        try:
            from google.oauth2.service_account import Credentials
            from googleapiclient.discovery import build
        except ModuleNotFoundError as exc:  # pragma: no cover - exercised when deps missing
            raise RuntimeError(
                "Google Sheets dependencies are not installed. "
                "Run `uv sync` or install `google-api-python-client`, `google-auth`, and `google-auth-httplib2`."
            ) from exc

        credentials = Credentials.from_service_account_file(
            str(self.credentials_path),
            scopes=["https://www.googleapis.com/auth/spreadsheets"],
        )
        return build("sheets", "v4", credentials=credentials)


def _column_letter(index: int) -> str:
    letters = ""
    while index > 0:
        index, remainder = divmod(index - 1, 26)
        letters = chr(65 + remainder) + letters
    return letters


__all__ = [
    "GoogleSheetTableClient",
    "spreadsheet_id_from_url",
]
