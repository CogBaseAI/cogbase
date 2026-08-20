"""Unit tests for the deployment-mode / jwt-secret config wiring and the
managed-mode startup guard.

These cover the pure helpers around auth configuration — the signing-secret
resolution, the ``require_configured_jwt_secret`` fail-fast, the deployment-mode
setter/getter, and how ``SystemConfig`` parses the two new fields. They touch
module-level state (``auth._jwt_secret``, ``auth._warned_dev_secret``,
``dependencies._deployment_mode``); the autouse fixture snapshots and restores it
so nothing leaks into the ASGI-based tests in the rest of the suite.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from api import auth
from api import dependencies
from api.auth import (
    InvalidToken,
    _DEV_SECRET,
    create_access_token,
    decode_token,
    get_jwt_secret,
    jwt_secret_is_configured,
    require_configured_jwt_secret,
    set_jwt_secret,
)
from api.dependencies import (
    get_deployment_mode,
    get_system_config_writable,
    get_tenant_skill_upload,
    set_deployment_mode,
    set_system_config_writable,
    set_tenant_skill_upload,
)
from api.system_config import SystemConfig


@pytest.fixture(autouse=True)
def _restore_module_state():
    """Snapshot and restore the auth/deployment module globals around each test."""
    saved = (
        auth._jwt_secret,
        auth._warned_dev_secret,
        dependencies._deployment_mode,
        dependencies._tenant_skill_upload,
        dependencies._system_config_writable,
    )
    yield
    (
        auth._jwt_secret,
        auth._warned_dev_secret,
        dependencies._deployment_mode,
        dependencies._tenant_skill_upload,
        dependencies._system_config_writable,
    ) = saved


# ---------------------------------------------------------------------------
# Signing-secret resolution
# ---------------------------------------------------------------------------


class TestJwtSecretResolution:
    def test_unset_falls_back_to_dev_secret(self):
        set_jwt_secret(None)
        assert jwt_secret_is_configured() is False
        assert get_jwt_secret() == _DEV_SECRET

    def test_configured_secret_is_used(self):
        set_jwt_secret("shared-across-every-node")
        assert jwt_secret_is_configured() is True
        assert get_jwt_secret() == "shared-across-every-node"

    def test_empty_string_treated_as_unset(self):
        # A blank ``jwt_secret:`` in YAML must not satisfy the guard.
        set_jwt_secret("")
        assert jwt_secret_is_configured() is False
        assert get_jwt_secret() == _DEV_SECRET


# ---------------------------------------------------------------------------
# Managed-mode startup guard
# ---------------------------------------------------------------------------


class TestRequireConfiguredJwtSecret:
    def test_saas_without_secret_refuses_to_boot(self):
        set_jwt_secret(None)
        with pytest.raises(RuntimeError, match="jwt_secret must be set in saas mode"):
            require_configured_jwt_secret("saas")

    def test_saas_with_secret_passes(self):
        set_jwt_secret("a-real-secret")
        require_configured_jwt_secret("saas")  # no raise

    @pytest.mark.parametrize("mode", ["dev", "single_tenant"])
    def test_trust_on_declaration_modes_never_require_a_secret(self, mode):
        set_jwt_secret(None)
        require_configured_jwt_secret(mode)  # no raise


# ---------------------------------------------------------------------------
# The secret actually signs tokens (multi-node correctness)
# ---------------------------------------------------------------------------


class TestTokenSigningUsesConfiguredSecret:
    def test_token_round_trips_under_same_secret(self):
        set_jwt_secret("node-shared-secret")
        token = create_access_token(
            user_id="u1", account_id="acct_1", email="u@x.co", role="owner"
        )
        claims = decode_token(token)
        assert claims["sub"] == "u1"
        assert claims["account_id"] == "acct_1"

    def test_token_from_a_different_secret_is_rejected(self):
        # A token minted on a node using secret A must not verify on a node using
        # secret B — this is exactly why every node must share one secret.
        set_jwt_secret("secret-A")
        token = create_access_token(
            user_id="u1", account_id="acct_1", email="u@x.co", role="owner"
        )
        set_jwt_secret("secret-B")
        with pytest.raises(InvalidToken, match="bad signature"):
            decode_token(token)


# ---------------------------------------------------------------------------
# Deployment-mode setter / getter
# ---------------------------------------------------------------------------


class TestDeploymentModeState:
    def test_pre_startup_fallback_is_dev(self):
        # The module default before lifespan runs (bare-ASGI unit tests).
        assert dependencies._deployment_mode == "dev"
        assert get_deployment_mode() == "dev"

    def test_set_overrides_for_process_lifetime(self):
        set_deployment_mode("saas")
        assert get_deployment_mode() == "saas"


# ---------------------------------------------------------------------------
# SystemConfig parsing of the two fields
# ---------------------------------------------------------------------------


class TestSystemConfigFields:
    def test_defaults(self):
        cfg = SystemConfig()
        assert cfg.deployment_mode == "dev"
        assert cfg.jwt_secret is None
        assert cfg.tenant_skill_upload is False
        assert cfg.system_config_writable is False

    def test_omitted_keys_keep_defaults(self):
        cfg = SystemConfig.from_yaml("system_db: {type: memory}")
        assert cfg.deployment_mode == "dev"
        assert cfg.jwt_secret is None
        assert cfg.tenant_skill_upload is False
        assert cfg.system_config_writable is False

    def test_values_parsed_from_yaml(self):
        cfg = SystemConfig.from_yaml("deployment_mode: saas\njwt_secret: hunter2\n")
        assert cfg.deployment_mode == "saas"
        assert cfg.jwt_secret == "hunter2"

    def test_invalid_deployment_mode_rejected(self):
        with pytest.raises(ValidationError):
            SystemConfig.from_yaml("deployment_mode: bogus")

    def test_tenant_skill_upload_parsed_from_yaml(self):
        cfg = SystemConfig.from_yaml("tenant_skill_upload: true")
        assert cfg.tenant_skill_upload is True

    def test_system_config_writable_parsed_from_yaml(self):
        cfg = SystemConfig.from_yaml("system_config_writable: true")
        assert cfg.system_config_writable is True


# ---------------------------------------------------------------------------
# tenant_skill_upload / system_config_writable setter / getter
# ---------------------------------------------------------------------------


class TestTenantSkillUploadState:
    def test_default_is_closed(self):
        # No permissive pre-startup fallback, unlike deployment_mode: the safe
        # default must be what a bare-ASGI unit test gets too.
        assert dependencies._tenant_skill_upload is False
        assert get_tenant_skill_upload() is False

    def test_set_overrides_for_process_lifetime(self):
        set_tenant_skill_upload(True)
        assert get_tenant_skill_upload() is True


class TestSystemConfigWritableState:
    def test_default_is_closed(self):
        assert dependencies._system_config_writable is False
        assert get_system_config_writable() is False

    def test_set_overrides_for_process_lifetime(self):
        set_system_config_writable(True)
        assert get_system_config_writable() is True
