"""Private remote MCP exceptions are exact, operator-owned and origin-bound."""

import socket

import httpx
import pytest

from xagent import config
from xagent.core.tools.core.mcp.http import MCPNetworkPolicy, create_mcp_http_client
from xagent.core.utils.public_http_transport import UnsafePublicHttpTarget
from xagent.core.utils.security import PrivateNetworkHostError

ENDPOINT = "http://mcp-tools:8000"


@pytest.fixture
def network(monkeypatch):
    answers = ["172.20.0.5"]
    monkeypatch.setattr(
        socket,
        "getaddrinfo",
        lambda *args: [
            (
                socket.AF_INET6 if ":" in value else socket.AF_INET,
                socket.SOCK_STREAM,
                6,
                "",
                (value, args[1]),
            )
            for value in answers
        ],
    )
    seen = []
    monkeypatch.setattr(
        httpx,
        "AsyncHTTPTransport",
        lambda **kwargs: httpx.MockTransport(
            lambda request: seen.append(request) or httpx.Response(200, json={})
        ),
    )
    return answers, seen


def approve(monkeypatch, endpoints=ENDPOINT, cidrs="172.20.0.0/16"):
    monkeypatch.setenv("XAGENT_MCP_PRIVATE_ENDPOINTS", endpoints)
    monkeypatch.setenv("XAGENT_MCP_PRIVATE_CIDRS", cidrs)


def test_config_default_deny_and_operator_lists(monkeypatch):
    assert config.get_mcp_private_endpoints() == []
    assert config.get_mcp_private_cidrs() == []
    approve(monkeypatch, f" {ENDPOINT}, http://10.0.3.7:9000 ", " 172.20.0.0/16 ")
    assert config.get_mcp_private_endpoints() == [ENDPOINT, "http://10.0.3.7:9000"]
    assert config.get_mcp_private_cidrs() == ["172.20.0.0/16"]


@pytest.mark.asyncio
async def test_approved_private_origin_is_pinned_to_checked_ip(monkeypatch, network):
    _, seen = network
    approve(monkeypatch)
    async with create_mcp_http_client(
        headers={"X-Key": "key"}, endpoint_url=f"{ENDPOINT}/mcp"
    ) as client:
        await client.post(f"{ENDPOINT}/mcp", json={})
    assert seen[0].url.host == "172.20.0.5"
    assert seen[0].headers["Host"] == "mcp-tools:8000"
    assert seen[0].headers["X-Key"] == "key"


@pytest.mark.asyncio
async def test_approved_literal_private_ip_origin(monkeypatch, network):
    answers, seen = network
    answers[:] = ["10.0.3.7"]
    approve(monkeypatch, "http://10.0.3.7:9000", "10.0.3.7/32")
    async with create_mcp_http_client(
        endpoint_url="http://10.0.3.7:9000/mcp"
    ) as client:
        await client.get("http://10.0.3.7:9000/mcp")
    assert seen[0].url.host == "10.0.3.7"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("endpoints", "cidrs", "answer"),
    [
        (ENDPOINT, "", "172.20.0.5"),
        ("", "172.20.0.0/16", "172.20.0.5"),
        ("http://mcp-tools:9000", "172.20.0.0/16", "172.20.0.5"),
        ("https://mcp-tools:8000", "172.20.0.0/16", "172.20.0.5"),
        (ENDPOINT, "172.20.0.0/16", "10.0.0.5"),
    ],
)
async def test_origin_and_every_dns_answer_must_both_be_approved(
    monkeypatch, network, endpoints, cidrs, answer
):
    answers, seen = network
    answers[:] = [answer]
    approve(monkeypatch, endpoints, cidrs)
    async with create_mcp_http_client(endpoint_url=f"{ENDPOINT}/mcp") as client:
        with pytest.raises(UnsafePublicHttpTarget):
            await client.post(f"{ENDPOINT}/mcp", content=b"credentials")
    assert not seen


@pytest.mark.asyncio
async def test_one_unapproved_dns_answer_blocks_the_origin(monkeypatch, network):
    answers, seen = network
    answers[:] = ["172.20.0.5", "192.168.1.9"]
    approve(monkeypatch)
    async with create_mcp_http_client(endpoint_url=f"{ENDPOINT}/mcp") as client:
        with pytest.raises(UnsafePublicHttpTarget):
            await client.get(f"{ENDPOINT}/mcp")
    assert not seen


@pytest.mark.asyncio
async def test_model_allowlist_does_not_approve_mcp(monkeypatch, network):
    _, seen = network
    monkeypatch.setenv("XAGENT_MODEL_PRIVATE_ENDPOINTS", ENDPOINT)
    monkeypatch.setenv("XAGENT_MODEL_PRIVATE_CIDRS", "172.20.0.0/16")
    async with create_mcp_http_client(endpoint_url=f"{ENDPOINT}/mcp") as client:
        with pytest.raises(UnsafePublicHttpTarget):
            await client.get(f"{ENDPOINT}/mcp")
    assert not seen


@pytest.mark.parametrize(
    ("endpoints", "cidrs", "variable"),
    [
        ("http://mcp-tools:8000/mcp", "172.20.0.0/16", "XAGENT_MCP_PRIVATE_ENDPOINTS"),
        ("http://*.internal:8000", "172.20.0.0/16", "XAGENT_MCP_PRIVATE_ENDPOINTS"),
        ("http://169.254.169.254", "172.20.0.0/16", "XAGENT_MCP_PRIVATE_ENDPOINTS"),
        (ENDPOINT, "169.254.0.0/16", "XAGENT_MCP_PRIVATE_CIDRS"),
        (ENDPOINT, "8.8.8.0/24", "XAGENT_MCP_PRIVATE_CIDRS"),
        (ENDPOINT, "not-a-cidr", "XAGENT_MCP_PRIVATE_CIDRS"),
    ],
)
def test_invalid_operator_configuration_fails_closed(
    monkeypatch, endpoints, cidrs, variable
):
    approve(monkeypatch, endpoints, cidrs)
    with pytest.raises(PrivateNetworkHostError, match=variable):
        MCPNetworkPolicy()
    with pytest.raises(PrivateNetworkHostError, match=variable):
        create_mcp_http_client(endpoint_url=f"{ENDPOINT}/mcp")


@pytest.mark.asyncio
async def test_unconfigured_policy_keeps_the_shared_public_resolver(network):
    answers, seen = network
    answers[:] = ["64:ff9b::808:808"]
    async with create_mcp_http_client(endpoint_url=f"{ENDPOINT}/mcp") as client:
        with pytest.raises(UnsafePublicHttpTarget, match="private network"):
            await client.get(f"{ENDPOINT}/mcp")
    assert not seen
