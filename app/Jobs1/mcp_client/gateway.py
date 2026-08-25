"""ProfileGateway MCP client.

Connects the agent to the AgentCore ProfileGateway (MCP) and returns the
allowed profile tools. The gateway URL + AWS_IAM auth type are injected into the
runtime environment by the AgentCore MCP construct on deploy
(`AGENTCORE_GATEWAY_PROFILEGATEWAY_URL`). Requests are SigV4-signed with the runtime's
execution-role credentials. Returns [] when the gateway is not wired (e.g. local
dev) or unreachable, so the agent degrades gracefully.
"""

import json
import os
import uuid

import botocore.session
import httpx
from botocore.auth import SigV4Auth
from botocore.awsrequest import AWSRequest
from mcp.client.streamable_http import streamablehttp_client
from strands.tools.mcp.mcp_client import MCPClient

_GATEWAY_URL_ENV = "AGENTCORE_GATEWAY_PROFILEGATEWAY_URL"
_SERVICE = "bedrock-agentcore"


class _SigV4Auth(httpx.Auth):
    """Signs each outgoing request with SigV4 for the gateway's AWS_IAM authorizer."""

    requires_request_body = True

    def __init__(self, service: str, region: str):
        self._service = service
        self._region = region
        self._session = botocore.session.Session()

    def auth_flow(self, request):
        creds = self._session.get_credentials()
        if creds is None:
            yield request
            return
        frozen = creds.get_frozen_credentials()
        aws_req = AWSRequest(
            method=request.method,
            url=str(request.url),
            data=request.content,
            headers=dict(request.headers),
        )
        SigV4Auth(frozen, self._service, self._region).add_auth(aws_req)
        request.headers.update(dict(aws_req.headers))
        yield request


def get_profile_gateway_tools(allowed: set[str]) -> list:
    """Return the allowed ProfileGateway tools, or [] if the gateway is unavailable.

    `allowed` is the set of tool names this agent may use (e.g. {"get_user_profile"}
    for read-only agents, or {"get_user_profile", "update_profile"} for editors).
    The client is started once and kept open for the process lifetime.
    """
    url = os.getenv(_GATEWAY_URL_ENV)
    if not url:
        return []
    region = os.getenv("AWS_REGION") or "eu-west-1"
    client = MCPClient(lambda: streamablehttp_client(url, auth=_SigV4Auth(_SERVICE, region)))
    try:
        client.start()
        tools = client.list_tools_sync()
    except Exception as exc:  # noqa: BLE001
        print(f"[profile-gateway] connect/list failed: {type(exc).__name__}: {exc}", flush=True)
        return []
    # Gateway tool names are target-prefixed ("ProfileTools___get_user_profile").
    return [t for t in tools if str(getattr(t, "tool_name", "")).split("___")[-1] in allowed]


_GET_PROFILE_TOOL = "ProfileTools___get_user_profile"


def _profile_from_result(result):
    """Extract the profile dict from an MCP tool result (text/json content blocks)."""
    content = result.get("content") if isinstance(result, dict) else getattr(result, "content", None)
    if not content:
        return None
    for block in content:
        b = block if isinstance(block, dict) else {}
        data = None
        if isinstance(b.get("json"), (dict, list)):
            data = b["json"]
        elif isinstance(b.get("text"), str):
            try:
                data = json.loads(b["text"])
            except Exception:  # noqa: BLE001
                continue
        if isinstance(data, dict):
            if isinstance(data.get("profile"), dict):
                return data["profile"] or None
            if "userId" not in data and data:
                return data
    return None


def fetch_user_profile(user_id):
    """Read the user's profile from the gateway using a FRESH short-lived MCP session.

    Deterministic per-turn read: it does not depend on the model calling the tool, and a new
    session per call avoids a stale long-lived connection dropping the profile after turn 1.
    Returns the profile dict, or None when the gateway is unavailable or the user has none.
    """
    url = os.getenv(_GATEWAY_URL_ENV)
    if not url or not user_id:
        return None
    region = os.getenv("AWS_REGION") or "eu-west-1"
    client = MCPClient(lambda: streamablehttp_client(url, auth=_SigV4Auth(_SERVICE, region)))
    try:
        with client:
            result = client.call_tool_sync(
                tool_use_id=f"profile-read-{uuid.uuid4().hex}",
                name=_GET_PROFILE_TOOL,
                arguments={"userId": user_id},
            )
    except Exception as exc:  # noqa: BLE001
        print(f"[profile-gateway] fetch_user_profile failed: {type(exc).__name__}: {exc}", flush=True)
        return None
    return _profile_from_result(result)


# --- Deterministic userId injection: the model must not be relied on to pass it ---
import contextvars  # noqa: E402

from strands.hooks import BeforeToolCallEvent, HookProvider, HookRegistry  # noqa: E402

_CURRENT_USER_ID = contextvars.ContextVar("profile_user_id", default=None)
_PROFILE_TOOLS = {"get_user_profile", "update_profile"}


def set_current_user_id(uid) -> None:
    """Bind the signed-in user's id for the current invocation (read by the injector)."""
    _CURRENT_USER_ID.set(uid or None)


class ProfileUserIdInjector(HookProvider):
    """Stamp userId onto every profile tool call so persistence never depends on the model."""

    def register_hooks(self, registry: HookRegistry, **_kwargs) -> None:
        registry.add_callback(BeforeToolCallEvent, self._inject)

    def _inject(self, event) -> None:
        name = str(event.tool_use.get("name", "")).split("___")[-1]
        if name in _PROFILE_TOOLS:
            uid = _CURRENT_USER_ID.get()
            if uid:
                event.tool_use.setdefault("input", {})["userId"] = uid
