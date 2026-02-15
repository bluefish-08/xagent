"""
Tests for ImageWebSearch tool
"""

import os
from pathlib import Path
from unittest.mock import AsyncMock, Mock, mock_open, patch

import httpx
import pytest

from xagent.core.tools.adapters.vibe.image_web_search import (
    ImageWebSearchArgs,
    ImageWebSearchResult,
    ImageWebSearchTool,
)


@pytest.fixture(autouse=True)
def cleanup_test_files():
    """Clean up test files after each test"""
    # Simple cleanup - remove any temp directories that might exist
    temp_dirs = [
        Path("uploads/temp"),
        Path("workspaces"),
        Path("temp"),
    ]

    yield

    # Clean up after test
    for temp_dir in temp_dirs:
        try:
            if temp_dir.exists():
                import shutil

                shutil.rmtree(temp_dir, ignore_errors=True)
        except (OSError, PermissionError):
            pass


@pytest.fixture
def image_search_tool():
    """Create ImageWebSearchTool instance for testing"""
    return ImageWebSearchTool()


@pytest.fixture
def mock_google_image_response():
    """Mock Google Custom Search API response for image search"""
    return {
        "items": [
            {
                "title": "Test Image 1",
                "link": "https://example.com/image1.jpg",
                "snippet": "A beautiful test image",
                "image": {
                    "thumbnailLink": "https://example.com/thumb1.jpg",
                    "contextLink": "https://example.com/context1",
                    "height": 800,
                    "width": 600,
                    "fileFormat": "jpeg",
                },
            },
            {
                "title": "Test Image 2",
                "link": "https://example.com/image2.png",
                "snippet": "Another test image",
                "image": {
                    "thumbnailLink": "https://example.com/thumb2.png",
                    "contextLink": "https://example.com/context2",
                    "height": 1024,
                    "width": 768,
                    "fileFormat": "png",
                },
            },
        ]
    }


@pytest.fixture
def mock_image_content():
    """Mock image content"""
    return b"fake_image_data_for_testing"


def has_real_google_credentials():
    """Check if real Google API credentials are available"""
    api_key = os.getenv("GOOGLE_API_KEY")
    cse_id = os.getenv("GOOGLE_CSE_ID")
    return bool(api_key and cse_id and api_key != "test_key" and cse_id != "test_cse")


class TestImageWebSearchTool:
    """Test cases for ImageWebSearchTool"""

    def test_tool_properties(self, image_search_tool):
        """Test basic tool properties"""
        assert image_search_tool.name == "image_web_search"
        assert "search" in image_search_tool.tags
        assert "image" in image_search_tool.tags
        assert image_search_tool.args_type() == ImageWebSearchArgs
        assert image_search_tool.return_type() == ImageWebSearchResult

    def test_sync_not_implemented(self, image_search_tool):
        """Test that sync execution raises NotImplementedError"""
        with pytest.raises(NotImplementedError):
            image_search_tool.run_json_sync({"query": "test"})

    @pytest.mark.asyncio
    async def test_missing_api_credentials(self, image_search_tool):
        """Test behavior when API credentials are missing"""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(
                ValueError, match="Missing required environment variables"
            ):
                await image_search_tool.run_json_async({"query": "test search"})

    @pytest.mark.asyncio
    async def test_successful_search_without_download(
        self, image_search_tool, mock_google_image_response
    ):
        """Test successful search without downloading images"""
        with patch.dict(
            os.environ, {"GOOGLE_API_KEY": "test_key", "GOOGLE_CSE_ID": "test_cse"}
        ):
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = mock_google_image_response
            mock_response.raise_for_status.return_value = None

            with patch("httpx.AsyncClient.get", return_value=mock_response):
                result = await image_search_tool.run_json_async(
                    {
                        "query": "test search",
                        "num_results": 2,
                        "save_to_workspace": False,
                    }
                )

                assert result["results"]
                assert len(result["results"]) == 2
                assert result["results"][0]["title"] == "Test Image 1"
                assert result["results"][0]["link"] == "https://example.com/image1.jpg"
                # local_path should be None when save_to_workspace is False
                assert result["results"][0]["local_path"] is None

    @pytest.mark.asyncio
    async def test_successful_search_with_download(
        self, image_search_tool, mock_google_image_response, mock_image_content
    ):
        """Test successful search with image download"""
        with patch.dict(
            os.environ, {"GOOGLE_API_KEY": "test_key", "GOOGLE_CSE_ID": "test_cse"}
        ):
            # Mock Google API response
            mock_api_response = Mock()
            mock_api_response.status_code = 200
            mock_api_response.json.return_value = mock_google_image_response
            mock_api_response.raise_for_status.return_value = None

            # Mock image download response
            mock_image_response = Mock()
            mock_image_response.status_code = 200
            mock_image_response.content = mock_image_content
            mock_image_response.raise_for_status.return_value = None

            async def mock_get(url, **kwargs):
                if "googleapis.com" in url:
                    return mock_api_response
                else:
                    return mock_image_response

            # Mock file operations to avoid actual file creation
            mock_bytes_io = Mock()
            mock_bytes_io.getvalue.return_value = b"fake_image_data"

            with patch("httpx.AsyncClient") as mock_client_class:
                mock_client = AsyncMock()
                mock_client.get.side_effect = mock_get
                mock_client_class.return_value.__aenter__.return_value = mock_client

                with (
                    patch("pathlib.Path.exists", return_value=True),
                    patch("pathlib.Path.write_bytes"),
                    patch("builtins.open", mock_open()),
                    patch("pathlib.Path.mkdir"),
                    patch("io.BytesIO", return_value=mock_bytes_io),
                ):
                    result = await image_search_tool.run_json_async(
                        {
                            "query": "test search",
                            "num_results": 2,
                            "save_to_workspace": True,
                        }
                    )

                    assert result["results"]
                    assert len(result["results"]) == 2
                    # Check that local_path is set (not None)
                    assert result["results"][0]["local_path"] is not None
                    assert result["results"][1]["local_path"] is not None
                    # Verify the path contains expected components
                    assert "uploads/temp/" in result["results"][0]["local_path"]

    @pytest.mark.asyncio
    async def test_api_403_error(self, image_search_tool):
        """Test handling of Google API 403 error"""
        with patch.dict(
            os.environ, {"GOOGLE_API_KEY": "invalid_key", "GOOGLE_CSE_ID": "test_cse"}
        ):
            mock_response = Mock()
            mock_response.status_code = 403
            mock_response.json.return_value = {
                "error": {
                    "message": "API quota exceeded",
                    "errors": [{"reason": "quotaExceeded"}],
                }
            }

            with patch("httpx.AsyncClient.get", return_value=mock_response):
                with pytest.raises(ValueError, match="API quota exceeded"):
                    await image_search_tool.run_json_async({"query": "test"})

    @pytest.mark.asyncio
    async def test_api_403_error_without_json_parse(self, image_search_tool):
        """Test handling of Google API 403 error when JSON parsing fails"""
        with patch.dict(
            os.environ, {"GOOGLE_API_KEY": "invalid_key", "GOOGLE_CSE_ID": "test_cse"}
        ):
            mock_response = Mock()
            mock_response.status_code = 403
            mock_response.json.side_effect = ValueError("Invalid JSON")

            with patch("httpx.AsyncClient.get", return_value=mock_response):
                with pytest.raises(ValueError, match="Google API 403 Forbidden error"):
                    await image_search_tool.run_json_async({"query": "test"})

    @pytest.mark.asyncio
    async def test_network_error(self, image_search_tool):
        """Test handling of network errors"""
        with patch.dict(
            os.environ, {"GOOGLE_API_KEY": "test_key", "GOOGLE_CSE_ID": "test_cse"}
        ):
            with patch(
                "httpx.AsyncClient.get",
                side_effect=httpx.ConnectError("Connection failed"),
            ):
                with pytest.raises(
                    ValueError, match="Network error during image search"
                ):
                    await image_search_tool.run_json_async({"query": "test"})

    @pytest.mark.asyncio
    async def test_empty_search_results(self, image_search_tool):
        """Test handling when no search results are found"""
        with patch.dict(
            os.environ, {"GOOGLE_API_KEY": "test_key", "GOOGLE_CSE_ID": "test_cse"}
        ):
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {}  # No items
            mock_response.raise_for_status.return_value = None

            with patch("httpx.AsyncClient.get", return_value=mock_response):
                result = await image_search_tool.run_json_async(
                    {"query": "nonexistent query"}
                )

                assert result["results"] == []

    @pytest.mark.asyncio
    async def test_num_results_limits(
        self, image_search_tool, mock_google_image_response
    ):
        """Test that num_results is properly limited between 1 and 10"""
        with patch.dict(
            os.environ, {"GOOGLE_API_KEY": "test_key", "GOOGLE_CSE_ID": "test_cse"}
        ):
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = mock_google_image_response
            mock_response.raise_for_status.return_value = None

            with patch("httpx.AsyncClient.get", return_value=mock_response) as mock_get:
                # Test with num_results > 10 (should be limited to 10)
                await image_search_tool.run_json_async(
                    {"query": "test", "num_results": 15, "save_to_workspace": False}
                )

                # Check that the API was called with num=10
                call_args = mock_get.call_args
                params = call_args[1]["params"]
                assert params["num"] == 10

                # Test with num_results < 1 (should be set to 1)
                await image_search_tool.run_json_async(
                    {"query": "test", "num_results": 0, "save_to_workspace": False}
                )

                call_args = mock_get.call_args
                params = call_args[1]["params"]
                assert params["num"] == 1

    @pytest.mark.asyncio
    async def test_proxy_configuration(
        self, image_search_tool, mock_google_image_response
    ):
        """Test proxy configuration from environment variables"""
        with patch.dict(os.environ, {}, clear=True):  # Start with clean environment
            os.environ["GOOGLE_API_KEY"] = "test_key"
            os.environ["GOOGLE_CSE_ID"] = "test_cse"
            os.environ["HTTP_PROXY"] = "http://proxy:8080"

            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = mock_google_image_response
            mock_response.raise_for_status.return_value = None

            with patch("httpx.AsyncClient") as mock_client_class:
                mock_client = AsyncMock()
                mock_client.get.return_value = mock_response
                mock_client_class.return_value.__aenter__.return_value = mock_client

                await image_search_tool.run_json_async(
                    {"query": "test", "save_to_workspace": False}
                )

                # Check that proxy was passed to AsyncClient
                mock_client_class.assert_called_once()
                call_kwargs = mock_client_class.call_args[1]
                assert call_kwargs.get("proxy") == "http://proxy:8080"
                # timeout might be passed separately, check the call args structure

    @pytest.mark.asyncio
    async def test_image_download_failure(
        self, image_search_tool, mock_google_image_response
    ):
        """Test handling of image download failures"""
        with patch.dict(
            os.environ, {"GOOGLE_API_KEY": "test_key", "GOOGLE_CSE_ID": "test_cse"}
        ):
            # Mock successful API response
            mock_api_response = Mock()
            mock_api_response.status_code = 200
            mock_api_response.json.return_value = mock_google_image_response
            mock_api_response.raise_for_status.return_value = None

            # Mock failed image download
            mock_image_error = httpx.HTTPStatusError(
                "404 Not Found",
                request=Mock(),
                response=Mock(status_code=404, reason_phrase="Not Found"),
            )

            async def mock_get(url, **kwargs):
                if "googleapis.com" in url:
                    return mock_api_response
                else:
                    raise mock_image_error

            with patch("httpx.AsyncClient.get", side_effect=mock_get):
                result = await image_search_tool.run_json_async(
                    {"query": "test", "save_to_workspace": True}
                )

                # Should still return results, but with None local_path
                assert result["results"]
                assert result["results"][0]["local_path"] is None

    @pytest.mark.asyncio
    async def test_parameter_validation(self, image_search_tool):
        """Test ImageWebSearchArgs parameter validation"""
        # Test valid parameters
        args = ImageWebSearchArgs(
            query="test search",
            num_results=5,
            image_size="large",
            image_type="photo",
            save_to_workspace=True,
        )
        assert args.query == "test search"
        assert args.num_results == 5
        assert args.image_size == "large"
        assert args.image_type == "photo"
        assert args.save_to_workspace is True

        # Test default values
        args = ImageWebSearchArgs(query="test")
        assert args.num_results == 5
        assert args.image_size == "medium"
        assert args.image_type == "photo"
        assert args.save_to_workspace is True

    @pytest.mark.asyncio
    async def test_api_parameter_construction(
        self, image_search_tool, mock_google_image_response
    ):
        """Test that API parameters are constructed correctly"""
        with patch.dict(
            os.environ, {"GOOGLE_API_KEY": "test_key", "GOOGLE_CSE_ID": "test_cse"}
        ):
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = mock_google_image_response
            mock_response.raise_for_status.return_value = None

            with patch("httpx.AsyncClient.get", return_value=mock_response) as mock_get:
                await image_search_tool.run_json_async(
                    {
                        "query": "test search",
                        "num_results": 3,
                        "image_size": "large",
                        "image_type": "clipart",
                    }
                )

                # Check that the API was called with correct parameters
                call_args_list = mock_get.call_args_list

                # Find the API call (not image download calls)
                api_call = None
                for call in call_args_list:
                    if len(call) >= 1 and "googleapis.com" in str(call[0]):
                        api_call = call
                        break

                assert api_call is not None, "API call to Google not found"

                # Extract params from the API call
                if len(api_call) >= 2 and api_call[1]:
                    params = api_call[1].get("params", {})
                else:
                    params = {}
                assert params["q"] == "test search"
                assert params["num"] == 3
                assert params["imgSize"] == "large"
                assert params["imgType"] == "clipart"
                assert params["searchType"] == "image"
                assert params["safe"] == "active"

    @pytest.mark.asyncio
    async def test_filename_generation(
        self, image_search_tool, mock_google_image_response, mock_image_content
    ):
        """Test that filenames are generated correctly for downloaded images"""
        with patch.dict(
            os.environ, {"GOOGLE_API_KEY": "test_key", "GOOGLE_CSE_ID": "test_cse"}
        ):
            # Mock Google API response
            mock_api_response = Mock()
            mock_api_response.status_code = 200
            mock_api_response.json.return_value = mock_google_image_response
            mock_api_response.raise_for_status.return_value = None

            # Mock image download response
            mock_image_response = Mock()
            mock_image_response.status_code = 200
            mock_image_response.content = mock_image_content
            mock_image_response.raise_for_status.return_value = None

            async def mock_get(url, **kwargs):
                if "googleapis.com" in url:
                    return mock_api_response
                else:
                    return mock_image_response

            # Mock file operations to avoid actual file creation
            mock_bytes_io = Mock()
            mock_bytes_io.getvalue.return_value = b"fake_image_data"

            with (
                patch("httpx.AsyncClient.get", side_effect=mock_get),
                patch("pathlib.Path.exists", return_value=True),
                patch("pathlib.Path.write_bytes"),
                patch("builtins.open", mock_open()),
                patch("pathlib.Path.mkdir"),
                patch("io.BytesIO", return_value=mock_bytes_io),
            ):
                result = await image_search_tool.run_json_async(
                    {
                        "query": "test search",
                        "num_results": 1,
                        "save_to_workspace": True,
                    }
                )

                # Check that the filename contains expected components
                local_path = result["results"][0]["local_path"]
                assert local_path is not None
                assert "image_search_1_" in local_path
                assert "Test_Image_1" in local_path or "test_image_1" in local_path
                assert local_path.endswith(
                    ".jpg"
                )  # Default extension for unknown types

    @pytest.mark.asyncio
    async def test_extension_detection(
        self, image_search_tool, mock_google_image_response, mock_image_content
    ):
        """Test that file extensions are detected correctly from URLs"""
        with patch.dict(
            os.environ, {"GOOGLE_API_KEY": "test_key", "GOOGLE_CSE_ID": "test_cse"}
        ):
            # Mock response with PNG URL
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = mock_google_image_response
            mock_response.raise_for_status.return_value = None

            # Mock image download response
            mock_image_response = Mock()
            mock_image_response.status_code = 200
            mock_image_response.content = mock_image_content
            mock_image_response.raise_for_status.return_value = None

            async def mock_get(url, **kwargs):
                if "googleapis.com" in url:
                    return mock_response
                else:
                    return mock_image_response

            # Mock file operations to avoid actual file creation
            mock_bytes_io = Mock()
            mock_bytes_io.getvalue.return_value = b"fake_image_data"

            with (
                patch("httpx.AsyncClient.get", side_effect=mock_get),
                patch("pathlib.Path.exists", return_value=True),
                patch("pathlib.Path.write_bytes"),
                patch("builtins.open", mock_open()),
                patch("pathlib.Path.mkdir"),
                patch("io.BytesIO", return_value=mock_bytes_io),
            ):
                result = await image_search_tool.run_json_async(
                    {
                        "query": "test search",
                        "num_results": 1,
                        "save_to_workspace": True,
                    }
                )

                # Check that the file extension matches the URL
                local_path = result["results"][0]["local_path"]
                assert local_path is not None
                # Should detect .jpg extension from first image URL
                assert local_path.endswith(".jpg")

    @pytest.mark.asyncio
    async def test_workspace_id_parameter(
        self, image_search_tool, mock_google_image_response, mock_image_content
    ):
        """Test that workspace_id parameter works correctly"""
        with patch.dict(
            os.environ, {"GOOGLE_API_KEY": "test_key", "GOOGLE_CSE_ID": "test_cse"}
        ):
            # Mock Google API response
            mock_api_response = Mock()
            mock_api_response.status_code = 200
            mock_api_response.json.return_value = mock_google_image_response
            mock_api_response.raise_for_status.return_value = None

            # Mock image download response
            mock_image_response = Mock()
            mock_image_response.status_code = 200
            mock_image_response.content = mock_image_content
            mock_image_response.raise_for_status.return_value = None

            async def mock_get(url, **kwargs):
                if "googleapis.com" in url:
                    return mock_api_response
                else:
                    return mock_image_response

            # Mock file operations to avoid actual file creation
            mock_bytes_io = Mock()
            mock_bytes_io.getvalue.return_value = b"fake_image_data"

            with (
                patch("httpx.AsyncClient.get", side_effect=mock_get),
                patch("pathlib.Path.exists", return_value=True),
                patch("pathlib.Path.write_bytes"),
                patch("builtins.open", mock_open()),
                patch("pathlib.Path.mkdir"),
                patch("io.BytesIO", return_value=mock_bytes_io),
            ):
                # Test with workspace_id parameter
                result = await image_search_tool.run_json_async(
                    {
                        "query": "test search",
                        "num_results": 1,
                        "save_to_workspace": True,
                        "workspace_id": "test_workspace_123",
                    }
                )

                # Check that the local path contains the workspace directory
                assert result["results"]
                local_path = result["results"][0]["local_path"]
                assert local_path is not None
                assert "workspaces/test_workspace_123/temp/" in local_path

    @pytest.mark.skipif(
        not has_real_google_credentials(),
        reason="Real Google API credentials not available",
    )
    @pytest.mark.asyncio
    async def test_real_google_search_integration(self, image_search_tool):
        """Integration test with real Google API (requires valid credentials)"""
        try:
            result = await image_search_tool.run_json_async(
                {"query": "cat", "num_results": 1, "save_to_workspace": False}
            )
        except ValueError as e:
            if "429 Too Many Requests" in str(e):
                pytest.skip("Google API rate limit exceeded (429 error)")
            elif "400 Bad Request" in str(e):
                pytest.skip(
                    "Google API credentials not configured or invalid (400 error)"
                )
            elif "Network error" in str(e) or "ConnectError" in str(e):
                pytest.skip("Network connection error - skipping integration test")
            else:
                raise
        except Exception as e:
            if "429" in str(e):
                pytest.skip("Google API rate limit exceeded (429 error)")
            elif "400" in str(e):
                pytest.skip(
                    "Google API credentials not configured or invalid (400 error)"
                )
            elif "ConnectError" in str(e) or "Network error" in str(e):
                pytest.skip("Network connection error - skipping integration test")
            else:
                raise

        # Verify basic structure
        assert "results" in result
        assert isinstance(result["results"], list)

        if result["results"]:  # Only verify if we got results
            search_result = result["results"][0]
            assert "title" in search_result
            assert "link" in search_result
            assert "snippet" in search_result
            assert "image_link" in search_result

        print(f"\n🔍 Found {len(result['results'])} search results for 'cat'")
        print("✅ Integration test completed successfully!")


class TestImageWebSearchToolArgs:
    """Test cases for ImageWebSearchArgs"""

    def test_args_model_validation(self):
        """Test ImageWebSearchArgs model validation"""
        # Valid args
        args = ImageWebSearchArgs(query="test search")
        assert args.query == "test search"
        assert args.num_results == 5  # default
        assert args.image_size == "medium"  # default
        assert args.image_type == "photo"  # default
        assert args.save_to_workspace is True  # default

        # Custom args
        args = ImageWebSearchArgs(
            query="custom search",
            num_results=3,
            image_size="large",
            image_type="clipart",
            save_to_workspace=False,
        )
        assert args.query == "custom search"
        assert args.num_results == 3
        assert args.image_size == "large"
        assert args.image_type == "clipart"
        assert args.save_to_workspace is False

    def test_result_model(self):
        """Test ImageWebSearchResult model"""
        results = [
            {
                "title": "Test",
                "link": "https://example.com",
                "snippet": "Test snippet",
                "image_link": "https://example.com/image.jpg",
                "local_path": "/path/to/image.jpg",
            }
        ]

        result = ImageWebSearchResult(results=results)
        assert result.results == results

        # Test model dump
        dumped = result.model_dump()
        assert "results" in dumped
        assert dumped["results"] == results
