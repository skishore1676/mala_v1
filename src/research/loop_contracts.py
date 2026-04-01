"""Shared contract metadata for Mala <-> Bhiksha loop artifacts."""

from __future__ import annotations

from typing import Any


DEPLOYMENT_CANDIDATES_CONTRACT_NAME = "deployment_candidates"
PLAYBOOK_CATALOG_CONTRACT_NAME = "playbook_catalog"
LOOP_ARTIFACT_SCHEMA_VERSION = 2


def build_contract_metadata(contract_name: str) -> dict[str, Any]:
    return {
        "contract_name": contract_name,
        "schema_version": LOOP_ARTIFACT_SCHEMA_VERSION,
    }


def validate_contract_metadata(
    payload: dict[str, Any],
    *,
    expected_contract_name: str | None = None,
) -> None:
    contract_name = payload.get("contract_name")
    if not contract_name:
        raise ValueError("Missing required contract_name")
    if expected_contract_name is not None and contract_name != expected_contract_name:
        raise ValueError(
            f"Unexpected contract_name {contract_name!r}; expected {expected_contract_name!r}"
        )
    schema_version = payload.get("schema_version")
    if schema_version != LOOP_ARTIFACT_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported schema_version {schema_version!r}; "
            f"expected {LOOP_ARTIFACT_SCHEMA_VERSION}"
        )


__all__ = [
    "DEPLOYMENT_CANDIDATES_CONTRACT_NAME",
    "PLAYBOOK_CATALOG_CONTRACT_NAME",
    "LOOP_ARTIFACT_SCHEMA_VERSION",
    "build_contract_metadata",
    "validate_contract_metadata",
]
