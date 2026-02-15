import os
from collections.abc import Sequence
from typing import Any, Optional

import requests

from .base import BaseRerank

OLD_FORMAT_MODELS = {"gte-rerank-v2"}
"""A set of model names that use the old WebAPI JSON format.

see: https://help.aliyun.com/model-studio/text-rerank-api
"""


class DashscopeRerank(BaseRerank):
    """Dashscope rerank model implementation."""

    def __init__(
        self,
        model: str = "qwen3-rerank",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        top_n: Optional[int] = None,
        instruct: Optional[str] = None,
    ):
        """
        Initialize Dashscope rerank model.

        Args:
            model: Model name (default: qwen3-rerank)
            api_key: API key (defaults to DASHSCOPE_API_KEY env var)
            base_url: API base URL (defaults to DashScope rerank endpoint)
            top_n: Number of top results to return
            instruct: Custom instruction for reranking
        """
        self.model = model
        self.api_key = api_key or os.getenv("DASHSCOPE_API_KEY")
        self.top_n = top_n
        self.instruct = instruct
        self.url = (
            base_url
            or "https://dashscope.aliyuncs.com/api/v1/services/rerank/text-rerank/text-rerank"
        )

        if not self.api_key:
            raise ValueError("API key required")

    def compress(
        self,
        documents: Sequence[str],
        query: str,
    ) -> Sequence[str]:
        """
        Rerank documents based on query relevance.

        Supports two API formats:
        - New format (qwen3-rerank): query and documents at top level
        - Old format (gte-rerank-v2): query and documents in input field

        Args:
            documents: List of document strings to rerank
            query: Query string

        Returns:
            Reranked list of documents

        Raises:
            requests.HTTPError: If API returns non-2xx status
            KeyError: If API response has unexpected format
            ValueError: If index in response is invalid
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        documents = list(documents)

        is_new_format = self.model.lower() not in OLD_FORMAT_MODELS

        optional_params: dict[str, Any] = {}
        if self.top_n is not None:
            optional_params["top_n"] = self.top_n
        if self.instruct is not None:
            optional_params["instruct"] = self.instruct

        payload: dict[str, Any]

        if is_new_format:
            # New format (qwen3-rerank)
            payload = {
                "model": self.model,
                "query": query,
                "documents": documents,
            } | optional_params
        else:
            # Old format (gte-rerank-v2)
            payload = {
                "model": self.model,
                "input": {
                    "query": query,
                    "documents": documents,
                },
                "parameters": {"return_documents": True} | optional_params,
            }

        response = requests.post(self.url, headers=headers, json=payload)
        response.raise_for_status()

        data = response.json()

        if is_new_format:
            # New qwen3-rerank format, no “documents.text” field!
            # eg:
            # {"object":"list","results":[{"index":0,"relevance_score":0.923461278969369},{"index":2,"relevance_score":0.7611337117952084}],"model":"qwen3-rerank","id":"f43f3fd9-db15-99da-8a0d-e21420198696","usage":{"total_tokens":105}}
            results = data["results"]
            return [documents[int(result["index"])] for result in results]
        else:
            # Old qwen3-gte-rerank-v2 format
            # eg:
            # {"output":{"results":[{"document":{"text":"..."},"index":0,"relevance_score":0.9272574008348661},{"document":{"text":"..."},"index":2,"relevance_score":0.7576691095659295}]},"usage":{"total_tokens":105},"request_id":"56a156f6-9b3a-47a8-b21c-0bf3415b8177"}
            results = data["output"]["results"]
            return [result["document"]["text"] for result in results]
