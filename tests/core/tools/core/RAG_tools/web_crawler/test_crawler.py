"""Unit tests for web crawler."""

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from xagent.core.tools.core.RAG_tools.core.schemas import WebCrawlConfig
from xagent.core.tools.core.RAG_tools.web_crawler.crawler import WebCrawler


class TestWebCrawler:
    """Test web crawler functionality."""

    @pytest.fixture
    def crawl_config(self):
        """Create a test crawl configuration."""
        return WebCrawlConfig(
            start_url="https://example.com",
            max_pages=5,
            max_depth=2,
            concurrent_requests=2,
            request_delay=0,
        )

    @pytest.fixture
    def sample_html(self):
        """Sample HTML content for testing."""
        return """
        <html>
            <head><title>Test Page</title></head>
            <body>
                <h1>Main Heading</h1>
                <p>This is a test page with some content.</p>
                <a href="/page1">Page 1</a>
                <a href="/page2">Page 2</a>
                <a href="https://other.com/external">External</a>
            </body>
        </html>
        """

    @pytest.mark.asyncio
    async def test_crawler_initialization(self, crawl_config):
        """Test crawler initialization."""
        crawler = WebCrawler(crawl_config)

        assert crawler.config == crawl_config
        assert len(crawler.visited_urls) == 0
        assert len(crawler.pending_urls) == 0
        assert len(crawler.crawl_results) == 0

    @pytest.mark.asyncio
    async def test_crawl_single_page(self, crawl_config, sample_html):
        """Test crawling a single page."""
        # Mock HTTP response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = sample_html
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.get.return_value = mock_response
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock()

        with patch("httpx.AsyncClient", return_value=mock_client):
            crawler = WebCrawler(crawl_config)
            results = await crawler.crawl()

        assert len(results) >= 1
        assert any(r.url == "https://example.com" for r in results)

    @pytest.mark.asyncio
    async def test_crawl_with_links(self, crawl_config, sample_html):
        """Test crawling and link discovery."""
        # Mock HTTP responses
        responses = {
            "https://example.com": sample_html,
            "https://example.com/page1": "<html><body><h1>Page 1</h1></body></html>",
            "https://example.com/page2": "<html><body><h1>Page 2</h1></body></html>",
        }

        def create_mock_response(url):
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.text = responses.get(url, "")
            mock_resp.raise_for_status = MagicMock()
            return mock_resp

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(
            side_effect=lambda url, **kw: create_mock_response(url)
        )
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock()

        with patch("httpx.AsyncClient", return_value=mock_client):
            crawler = WebCrawler(crawl_config)
            results = await crawler.crawl()

        # Should have crawled start page and discovered links
        assert len(results) >= 1
        # Check that links were extracted
        stats = crawler.get_statistics()
        assert stats["total_urls_found"] > 0

    @pytest.mark.asyncio
    async def test_max_pages_limit(self, crawl_config, sample_html):
        """Test that max_pages limit is respected."""
        config = WebCrawlConfig(
            start_url="https://example.com",
            max_pages=2,  # Limit to 2 pages
            max_depth=3,
            concurrent_requests=1,
            request_delay=0,
        )

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = sample_html
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.get.return_value = mock_response
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock()

        with patch("httpx.AsyncClient", return_value=mock_client):
            crawler = WebCrawler(config)
            await crawler.crawl()

        # Should not exceed max_pages
        assert len(crawler.visited_urls) <= 2

    @pytest.mark.asyncio
    async def test_max_depth_limit(self, sample_html):
        """Test that max_depth limit is respected."""
        config = WebCrawlConfig(
            start_url="https://example.com",
            max_pages=100,
            max_depth=1,  # Limit depth to 1
            concurrent_requests=1,
            request_delay=0,
        )

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = sample_html
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.get.return_value = mock_response
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock()

        with patch("httpx.AsyncClient", return_value=mock_client):
            crawler = WebCrawler(config)
            results = await crawler.crawl()

        # All crawled pages should be at depth 0 or 1
        for result in results:
            assert result.depth <= 1

    @pytest.mark.asyncio
    async def test_http_error_handling(self, crawl_config):
        """Test handling of HTTP errors."""
        # Mock HTTP error response
        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "Not Found", request=MagicMock(), response=mock_response
        )

        mock_client = AsyncMock()
        mock_client.get.return_value = mock_response
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock()

        with patch("httpx.AsyncClient", return_value=mock_client):
            crawler = WebCrawler(crawl_config)
            await crawler.crawl()

        # Should handle error gracefully
        assert len(crawler.failed_urls) > 0
        assert "https://example.com" in crawler.failed_urls

    @pytest.mark.asyncio
    async def test_network_error_handling(self, crawl_config):
        """Test handling of network errors."""
        mock_client = AsyncMock()
        mock_client.get = AsyncMock(side_effect=httpx.RequestError("Connection error"))
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock()

        with patch("httpx.AsyncClient", return_value=mock_client):
            crawler = WebCrawler(crawl_config)
            await crawler.crawl()

        # Should handle error gracefully
        assert len(crawler.failed_urls) > 0

    @pytest.mark.asyncio
    async def test_insufficient_content_handling(self, crawl_config):
        """Test handling of pages with insufficient content."""
        # Mock response with very short content
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = "<html><body>Hi</body></html>"
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.get.return_value = mock_response
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock()

        with patch("httpx.AsyncClient", return_value=mock_client):
            crawler = WebCrawler(crawl_config)
            results = await crawler.crawl()

        # Should skip pages with insufficient content
        assert len([r for r in results if r.status == "success"]) == 0

    @pytest.mark.asyncio
    async def test_same_domain_filtering(self, sample_html):
        """Test same domain filtering."""
        config = WebCrawlConfig(
            start_url="https://example.com",
            max_pages=10,
            max_depth=2,
            same_domain_only=True,
            concurrent_requests=1,
            request_delay=0,
        )

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = sample_html
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.get.return_value = mock_response
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock()

        with patch("httpx.AsyncClient", return_value=mock_client):
            crawler = WebCrawler(config)
            results = await crawler.crawl()

        # External links should not be crawled
        assert not any(r.url == "https://other.com/external" for r in results)

    @pytest.mark.asyncio
    async def test_get_statistics(self, crawl_config, sample_html):
        """Test statistics collection."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = sample_html
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.get.return_value = mock_response
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock()

        with patch("httpx.AsyncClient", return_value=mock_client):
            crawler = WebCrawler(crawl_config)
            await crawler.crawl()

        stats = crawler.get_statistics()
        assert "total_urls_found" in stats
        assert "visited_urls" in stats
        assert "successful_pages" in stats
        assert "failed_pages" in stats
        assert "pending_urls" in stats

    @pytest.mark.asyncio
    async def test_progress_callback(self, crawl_config, sample_html):
        """Test progress callback functionality."""
        progress_updates = []

        def progress_callback(message, completed, total):
            progress_updates.append((message, completed, total))

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = sample_html
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.get.return_value = mock_response
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock()

        with patch("httpx.AsyncClient", return_value=mock_client):
            crawler = WebCrawler(crawl_config, progress_callback)
            await crawler.crawl()

        # Progress callback should have been called
        assert len(progress_updates) > 0
