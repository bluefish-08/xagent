"""
Browser automation tools for xagent Vibe framework.

Provides browser automation capabilities with proper session lifecycle management.
Browser sessions are automatically cleaned up when tasks complete.
"""

import logging
from typing import Any, Mapping, Optional, Type

from pydantic import BaseModel, Field

from ....tools.core.browser_use import (
    browser_click,
    browser_close,
    browser_evaluate,
    browser_extract_text,
    browser_fill,
    browser_list_sessions,
    browser_navigate,
    browser_screenshot,
    browser_select_option,
    browser_wait_for_selector,
    get_browser_manager,
)
from ....workspace import TaskWorkspace
from .base import AbstractBaseTool, ToolCategory, ToolVisibility

logger = logging.getLogger(__name__)


# ============== Input/Output Schemas ==============


class BrowserNavigateArgs(BaseModel):
    session_id: str = Field(description="Session ID for the browser")
    url: str = Field(description="URL to navigate to")
    headless: bool = Field(default=True, description="Whether to run in headless mode")
    wait_until: str = Field(default="networkidle", description="Wait condition")


class BrowserNavigateResult(BaseModel):
    success: bool = Field(description="Whether navigation succeeded")
    session_id: str = Field(description="Session ID")
    url: str = Field(description="Navigated URL")
    title: str = Field(default="", description="Page title")
    message: str = Field(description="Result message")
    error: str = Field(default="", description="Error message if failed")


class BrowserClickArgs(BaseModel):
    session_id: str = Field(description="Session ID")
    selector: str = Field(description="CSS selector or XPath")
    headless: bool = Field(default=True, description="Whether to run in headless mode")
    timeout: int = Field(default=30000, description="Timeout in milliseconds")


class BrowserClickResult(BaseModel):
    success: bool = Field(description="Whether click succeeded")
    session_id: str = Field(description="Session ID")
    selector: str = Field(description="Clicked selector")
    message: str = Field(description="Result message")
    error: str = Field(default="", description="Error message if failed")


class BrowserFillArgs(BaseModel):
    session_id: str = Field(description="Session ID")
    selector: str = Field(description="CSS selector or XPath")
    value: str = Field(description="Text value to fill")
    headless: bool = Field(default=True, description="Whether to run in headless mode")


class BrowserFillResult(BaseModel):
    success: bool = Field(description="Whether fill succeeded")
    session_id: str = Field(description="Session ID")
    selector: str = Field(description="Filled selector")
    value: str = Field(description="Filled value (preview)")
    message: str = Field(description="Result message")
    error: str = Field(default="", description="Error message if failed")


class BrowserScreenshotArgs(BaseModel):
    session_id: str = Field(description="Session ID")
    full_page: bool = Field(default=False, description="Whether to capture full page")
    headless: bool = Field(
        default=True,
        description="Whether to run in headless mode (default: True, recommended). Only set to False for debugging security pages or visual troubleshooting.",
    )
    width: Optional[int] = Field(
        default=None,
        description="Desired output width in pixels (e.g., 2025, 1080, 1920). Set viewport size before screenshot.",
    )
    height: Optional[int] = Field(
        default=None,
        description="Desired output height in pixels (e.g., 2025, 1080, 1920). Set viewport size before screenshot.",
    )
    wait_for_lazy_load: bool = Field(
        default=False,
        description="Whether to scroll the page to trigger lazy-loaded content before screenshot. Only effective when full_page=True. Use this for pages with infinite scroll, lazy-loaded images, or dynamic content loading.",
    )
    output_filename: Optional[str] = Field(
        default=None,
        description="Output filename for saving to output directory. If provided, saves to output/ instead of temp/. Example: 'result.png'",
    )


class BrowserScreenshotResult(BaseModel):
    success: bool = Field(description="Whether screenshot succeeded")
    session_id: str = Field(description="Session ID")
    screenshot: str = Field(default="", description="Base64 encoded screenshot")
    format: str = Field(description="Image format")
    full_page: bool = Field(description="Whether full page was captured")
    wait_for_lazy_load: bool = Field(description="Whether lazy loading was enabled")
    message: str = Field(description="Result message")
    error: str = Field(default="", description="Error message if failed")


class BrowserExtractTextArgs(BaseModel):
    session_id: str = Field(description="Session ID")
    selector: str = Field(default="body", description="CSS selector or XPath")
    headless: bool = Field(default=True, description="Whether to run in headless mode")


class BrowserExtractTextResult(BaseModel):
    success: bool = Field(description="Whether extraction succeeded")
    session_id: str = Field(description="Session ID")
    selector: str = Field(description="Extracted selector")
    text: str = Field(default="", description="Extracted text")
    length: int = Field(default=0, description="Text length")
    message: str = Field(description="Result message")
    error: str = Field(default="", description="Error message if failed")


class BrowserEvaluateArgs(BaseModel):
    session_id: str = Field(description="Session ID")
    javascript: str = Field(description="JavaScript code to execute")
    headless: bool = Field(default=True, description="Whether to run in headless mode")


class BrowserEvaluateResult(BaseModel):
    success: bool = Field(description="Whether execution succeeded")
    session_id: str = Field(description="Session ID")
    result: Any = Field(default=None, description="Execution result")
    message: str = Field(description="Result message")
    error: str = Field(default="", description="Error message if failed")


class BrowserSelectOptionArgs(BaseModel):
    session_id: str = Field(description="Session ID")
    selector: str = Field(description="CSS selector for the select element")
    value: Optional[str] = Field(default=None, description="Option value to select")
    index: Optional[int] = Field(default=None, description="Option index to select")
    headless: bool = Field(default=True, description="Whether to run in headless mode")


class BrowserSelectOptionResult(BaseModel):
    success: bool = Field(description="Whether selection succeeded")
    session_id: str = Field(description="Session ID")
    selector: str = Field(description="Select element selector")
    selected_value: str = Field(default="", description="Selected option value")
    selected_index: Optional[int] = Field(
        default=None, description="Selected option index"
    )
    message: str = Field(description="Result message")
    error: str = Field(default="", description="Error message if failed")


class BrowserWaitForSelectorArgs(BaseModel):
    session_id: str = Field(description="Session ID")
    selector: str = Field(description="CSS selector or XPath to wait for")
    timeout: int = Field(default=30000, description="Timeout in milliseconds")
    headless: bool = Field(default=True, description="Whether to run in headless mode")


class BrowserWaitForSelectorResult(BaseModel):
    success: bool = Field(description="Whether wait succeeded")
    session_id: str = Field(description="Session ID")
    selector: str = Field(description="Selector that was waited for")
    message: str = Field(description="Result message")
    error: str = Field(default="", description="Error message if failed")


class BrowserCloseArgs(BaseModel):
    session_id: str = Field(description="Session ID to close")


class BrowserCloseResult(BaseModel):
    success: bool = Field(description="Whether close succeeded")
    session_id: str = Field(description="Closed session ID")
    message: str = Field(description="Result message")
    error: str = Field(default="", description="Error message if failed")


# ============== Tool Implementations ==============


class BrowserNavigateTool(AbstractBaseTool):
    category = ToolCategory.BROWSER
    """Navigate to a URL in a browser session."""

    def __init__(
        self, task_id: Optional[str] = None, workspace: Optional["TaskWorkspace"] = None
    ):
        self._visibility = ToolVisibility.PUBLIC
        self._task_id = task_id
        self._workspace = workspace

    @property
    def name(self) -> str:
        return "browser_navigate"

    @property
    def description(self) -> str:
        return """Navigate to a URL in a browser session.

        The browser session is created automatically on first use (lazy initialization).
        Use this tool to visit web pages, open URLs, and navigate between pages.

        WORKSPACE FILE SUPPORT (AUTOMATIC PATH RESOLUTION):
        - You can use relative filenames without specifying the directory
        - The browser will automatically search for files in this order:
          1. input/ directory (e.g., "poster.html" → input/poster.html)
          2. output/ directory (e.g., "poster.html" → output/poster.html)
          3. temp/ directory (e.g., "poster.html" → temp/poster.html)
          4. workspace root (e.g., "poster.html" → workspace/poster.html)

        Examples:
        - url="poster.html" → Automatically finds poster.html in input/output/temp/
        - url="report.html" → Opens report.html from the first matching directory
        - url="temp/screenshot.png" → Opens specific file from temp/ directory

        Supported formats: HTML, SVG, images, PDF (browser may show download prompt for PDF)

        IMPORTANT USAGE NOTES:
        - The browser runs in HEADLESS mode (no GUI) by default
        - If you encounter security verification pages (like "Baidu security verification"), try headless=False
        - Many websites detect headless browsers more easily than visible browsers
        - Setting headless=False will open a visible browser window

        ANTI-DETECTION MEASURES:
        The browser automatically applies anti-detection techniques:
        - Hides navigator.webdriver property
        - Adds realistic browser features (plugins, languages, chrome object)
        - Sets Chinese locale and timezone
        - Disables automation control flags

        Args:
            session_id: Session ID for the browser (typically use task_id)
            url: Target URL to navigate to
                 - http:// or https:// for websites
                 - Relative filename (e.g., "poster.html") for workspace files
                 - Absolute path or file:// URL also supported
            headless: Whether to run in headless mode (default: True)
                    Set to False if you encounter security verification pages
            wait_until: When to consider navigation succeeded (default: "networkidle"):
                - "load": Page is fully loaded (safer for most pages)
                - "domcontentloaded": DOM is ready (RECOMMENDED for modern SPA apps, faster)
                - "networkidle": No network activity for 500ms (may hang on pages with continuous requests)
                - "commit": Received response headers (fastest, but page may still be loading)

        Recommendations:
        - Use wait_until="domcontentloaded" for faster navigation on modern web apps
        - For workspace files, just use the filename (e.g., "index.html") - no need to specify directory
        - If you see security verification, try headless=False and wait again
        - For major platforms (Baidu, Alibaba, etc.), consider using their official APIs instead
        """

    @property
    def tags(self) -> list[str]:
        return ["browser", "automation", "web", "navigation"]

    def args_type(self) -> Type[BaseModel]:
        return BrowserNavigateArgs

    def return_type(self) -> Type[BaseModel]:
        return BrowserNavigateResult

    def run_json_sync(self, args: Mapping[str, Any]) -> Any:
        """
        Synchronous wrapper (not supported - use run_json_async instead).

        Browser tools are async-only. Please call them from async context.
        """
        raise NotImplementedError(
            f"{self.name} is async-only. Use await {self.name}() or call from async context."
        )

    async def run_json_async(self, args: Mapping[str, Any]) -> Any:
        # Convert workspace-relative paths to file:// URLs
        url = args.get("url", "")

        # Debug logging
        import logging

        logger = logging.getLogger(__name__)

        # ALWAYS log to help debugging
        logger.warning("[BrowserNavigateTool] ========== NAVIGATE START ==========")
        logger.warning(f"[BrowserNavigateTool] Input URL: {url}")
        logger.warning(
            f"[BrowserNavigateTool] Workspace available: {self._workspace is not None}"
        )

        if self._workspace is not None:
            logger.warning(
                f"[BrowserNavigateTool] Workspace path: {self._workspace.workspace_dir}"
            )
        else:
            logger.warning(
                "[BrowserNavigateTool] ⚠️  WARNING: Workspace is None! Cannot resolve relative paths."
            )

        if (
            self._workspace
            and url
            and not url.startswith(
                ("http://", "https://", "file://", "about:", "data:")
            )
        ):
            # Use workspace's intelligent file search (input → output → temp → root)
            try:
                resolved_path = self._workspace.resolve_path_with_search(url)
                logger.info(f"[BrowserNavigateTool] Resolved path: {resolved_path}")
                logger.info(
                    f"[BrowserNavigateTool] File exists: {resolved_path.exists()}"
                )

                if resolved_path.exists():
                    args = dict(args)  # Make a mutable copy
                    args["url"] = resolved_path.as_uri()
                    logger.info(
                        f"[BrowserNavigateTool] Converted to file:// URL: {args['url']}"
                    )
                else:
                    # File doesn't exist after search
                    logger.warning(
                        f"[BrowserNavigateTool] File not found after search: {resolved_path}"
                    )
                    # List what directories were searched
                    logger.info(
                        "[BrowserNavigateTool] Searched in: input -> output -> temp -> root"
                    )
                    return {
                        "success": False,
                        "session_id": args.get("session_id", ""),
                        "url": url,
                        "title": "",
                        "message": "",
                        "error": f"File not found: {url}. Searched in workspace directories (input/, output/, temp/). Please check if the file exists.",
                    }
            except ValueError as e:
                # Path is outside workspace
                logger.warning(f"[BrowserNavigateTool] Path error: {e}")
                return {
                    "success": False,
                    "session_id": args.get("session_id", ""),
                    "url": url,
                    "title": "",
                    "message": "",
                    "error": str(e),
                }
        else:
            logger.warning(
                f"[BrowserNavigateTool] Skipping path conversion (workspace={self._workspace is not None}, url={url[:50]})"
            )

        logger.warning(
            f"[BrowserNavigateTool] Calling browser_navigate with URL: {args.get('url', '')[:100]}"
        )

        result = await browser_navigate(**args)

        logger.warning(
            f"[BrowserNavigateTool] Navigation result: success={result.get('success')}"
        )
        if not result.get("success"):
            logger.warning(
                f"[BrowserNavigateTool] Navigation error: {result.get('error', 'Unknown error')[:200]}"
            )

        logger.warning("[BrowserNavigateTool] ========== NAVIGATE END ==========")

        return BrowserNavigateResult(**result).model_dump()

    async def setup(self, task_id: Optional[str] = None) -> None:
        """Setup called when task starts - store task_id for session tracking."""
        if task_id:
            self._task_id = task_id

    async def teardown(self, task_id: Optional[str] = None) -> None:
        """Cleanup browser sessions when task completes."""
        if self._task_id or task_id:
            target_task_id = self._task_id or task_id
            if target_task_id:  # Type guard for mypy
                try:
                    manager = get_browser_manager()
                    # Close the session associated with this task
                    # Session ID is typically the same as task_id in our pattern
                    await manager.close(target_task_id)
                    logger.info(
                        f"Cleaned up browser session for task {target_task_id} via teardown"
                    )
                except Exception as e:
                    logger.error(
                        f"Failed to cleanup browser session for task {target_task_id}: {e}",
                        exc_info=True,
                    )


class BrowserClickTool(AbstractBaseTool):
    category = ToolCategory.BROWSER
    """Click an element on the current page."""

    def __init__(self, task_id: Optional[str] = None):
        self._visibility = ToolVisibility.PUBLIC
        self._task_id = task_id

    @property
    def name(self) -> str:
        return "browser_click"

    @property
    def description(self) -> str:
        return """Click an element on the current page.

        Use this tool to interact with buttons, links, and other clickable elements.
        The browser session is created automatically if it doesn't exist.

        IMPORTANT: The browser runs in HEADLESS mode (no GUI) by default.
        If you encounter security verification or element detection issues, try headless=False.

        Args:
            session_id: Session ID for the browser
            selector: CSS selector or XPath for the element to click
            headless: Whether to run in headless mode (default: True)
            timeout: Maximum time to wait for element to be clickable (default: 30000ms = 30 seconds)

        Tips:
        - If element is not found, try waiting for it first with browser_wait_for_selector
        - If click times out, the element may be hidden, disabled, or not yet loaded
        - For pages with security verification, use headless=False in browser_navigate first

        Example: Click a button with selector 'button.submit'
        """

    @property
    def tags(self) -> list[str]:
        return ["browser", "automation", "interaction"]

    def args_type(self) -> Type[BaseModel]:
        return BrowserClickArgs

    def return_type(self) -> Type[BaseModel]:
        return BrowserClickResult

    def run_json_sync(self, args: Mapping[str, Any]) -> Any:
        """
        Synchronous wrapper (not supported - use run_json_async instead).

        Browser tools are async-only. Please call them from async context.
        """
        raise NotImplementedError(
            f"{self.name} is async-only. Use await {self.name}() or call from async context."
        )

    async def run_json_async(self, args: Mapping[str, Any]) -> Any:
        result = await browser_click(**args)
        return BrowserClickResult(**result).model_dump()


class BrowserFillTool(AbstractBaseTool):
    category = ToolCategory.BROWSER
    """Fill an input field with text."""

    def __init__(self, task_id: Optional[str] = None):
        self._visibility = ToolVisibility.PUBLIC
        self._task_id = task_id

    @property
    def name(self) -> str:
        return "browser_fill"

    @property
    def description(self) -> str:
        return """Fill an input field with text.

        Use this tool to enter text into input fields, text areas, and form fields.
        The browser session is created automatically if it doesn't exist.

        IMPORTANT: The browser runs in HEADLESS mode (no GUI) by default.
        Do NOT set headless=False unless explicitly required by the user.

        Example: Fill an email input with selector 'input[name=\"email\"]'
        """

    @property
    def tags(self) -> list[str]:
        return ["browser", "automation", "input", "form"]

    def args_type(self) -> Type[BaseModel]:
        return BrowserFillArgs

    def return_type(self) -> Type[BaseModel]:
        return BrowserFillResult

    def run_json_sync(self, args: Mapping[str, Any]) -> Any:
        """
        Synchronous wrapper (not supported - use run_json_async instead).

        Browser tools are async-only. Please call them from async context.
        """
        raise NotImplementedError(
            f"{self.name} is async-only. Use await {self.name}() or call from async context."
        )

    async def run_json_async(self, args: Mapping[str, Any]) -> Any:
        result = await browser_fill(**args)
        return BrowserFillResult(**result).model_dump()


class BrowserScreenshotTool(AbstractBaseTool):
    category = ToolCategory.BROWSER
    """Take a screenshot of the current page."""

    def __init__(
        self, task_id: Optional[str] = None, workspace: Optional["TaskWorkspace"] = None
    ):
        self._visibility = ToolVisibility.PUBLIC
        self._task_id = task_id
        self._workspace = workspace

    @property
    def name(self) -> str:
        return "browser_screenshot"

    @property
    def description(self) -> str:
        return """Take a screenshot of the current page.

        Returns a relative path to the saved screenshot.
        By default, saves to workspace/temp/ with auto-generated filename (e.g., temp/screenshot_20250113_123456.png).
        If output_filename is provided, saves to workspace/output/ with the specified filename (e.g., output/result.png).

        You can optionally specify the desired output width using the width parameter.
        This does NOT change the browser viewport size. Instead, you should set the width in your HTML/CSS:
        - HTML: <meta name='viewport' content='width=1080'>
        - CSS: body { width: 1080px; max-width: 1080px; }
        The width parameter is just informational to tell you what width to use in your HTML.

        IMPORTANT USAGE NOTES:
        - The browser runs in HEADLESS mode (no GUI) by default - RECOMMENDED for stability and speed
        - Only use headless=False for debugging security verification pages or visual troubleshooting
        - headless=False may be slower and less reliable for full_page screenshots
        - AFTER taking a screenshot, you MUST use image understanding tools to analyze the screenshot content
        - The screenshot is saved as a PNG file
        - The returned path is relative to workspace (e.g., temp/screenshot_xxx.png or output/result.png)
        - Use output_filename when the screenshot is a final deliverable that should be preserved
        - Use width/height to control the viewport size before taking the screenshot
        - Planning: When planning to use browser_screenshot, also plan to use image understanding tools afterwards

        PARAMETERS:
        - full_page: Set to True to capture the entire scrolling page (not just visible area)
        - wait_for_lazy_load: Set to True (with full_page=True) for pages with infinite scroll or lazy-loaded content.
          This will scroll the page to trigger lazy loading before capturing.

        Example workflow (intermediate screenshot):
        1. browser_navigate to the target URL
        2. browser_screenshot(full_page=True, wait_for_lazy_load=True) to capture the full page
        3. understand_images(images="temp/screenshot_20250113_123456.png", question="What do you see?")

        Example workflow (final output with custom size):
        1. browser_navigate to the result page
        2. browser_screenshot(width=1920, height=1080, output_filename="result.png") (returns "output/result.png")
        """

    @property
    def tags(self) -> list[str]:
        return ["browser", "automation", "screenshot", "image"]

    def args_type(self) -> Type[BaseModel]:
        return BrowserScreenshotArgs

    def return_type(self) -> Type[BaseModel]:
        return BrowserScreenshotResult

    def run_json_sync(self, args: Mapping[str, Any]) -> Any:
        """
        Synchronous wrapper (not supported - use run_json_async instead).

        Browser tools are async-only. Please call them from async context.
        """
        raise NotImplementedError(
            f"{self.name} is async-only. Use await {self.name}() or call from async context."
        )

    async def run_json_async(self, args: Mapping[str, Any]) -> Any:
        # Call async version directly
        import traceback

        try:
            result = await browser_screenshot(**args)
        except Exception as e:
            print(f"[BrowserScreenshotTool] Error type: {type(e).__name__}")
            print(f"[BrowserScreenshotTool] Error: {str(e)}")
            traceback.print_exc()
            raise

        # If workspace is available, save screenshot to file and return relative path
        if self._workspace and result.get("success"):
            try:
                import base64
                from datetime import datetime

                # Extract base64 data from data URI
                screenshot_data = result.get("screenshot", "")
                if screenshot_data.startswith("data:image/png;base64,"):
                    base64_data = screenshot_data.split(",", 1)[1]
                else:
                    base64_data = screenshot_data

                # Decode base64 to bytes
                image_bytes = base64.b64decode(base64_data)

                # Determine filename and save directory
                output_filename = args.get("output_filename")
                if output_filename:
                    filename = output_filename
                    file_path = self._workspace.output_dir / filename
                else:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"screenshot_{timestamp}.png"
                    file_path = self._workspace.temp_dir / filename

                # Save to file
                with open(file_path, "wb") as f:
                    f.write(image_bytes)

                relative_path = str(
                    file_path.relative_to(self._workspace.workspace_dir)
                )
                result["screenshot"] = relative_path
                result["format"] = "file"
                result["message"] = f"Screenshot saved to {relative_path}"
            except Exception as e:
                logger.error(
                    f"Failed to save screenshot to workspace: {e}", exc_info=True
                )
                result["message"] = (
                    f"Screenshot captured (base64 format, file save failed: {e})"
                )

        return BrowserScreenshotResult(**result).model_dump()


class BrowserExtractTextTool(AbstractBaseTool):
    category = ToolCategory.BROWSER
    """Extract text content from the page."""

    def __init__(self, task_id: Optional[str] = None):
        self._visibility = ToolVisibility.PUBLIC
        self._task_id = task_id

    @property
    def name(self) -> str:
        return "browser_extract_text"

    @property
    def description(self) -> str:
        return """Extract text content from the page or a specific element.

        Use this tool to read page content, scrape text, or analyze web pages.
        The browser session is created automatically if it doesn't exist.

        IMPORTANT: The browser runs in HEADLESS mode (no GUI) by default.
        If you encounter security verification or text extraction fails, try headless=False in browser_navigate.

        Args:
            session_id: Session ID for the browser
            selector: CSS selector or XPath (default: "body" extracts entire page)
            headless: Whether to run in headless mode (default: True)

        Timeout: This operation has a 15-second timeout with JavaScript fallback.
        If inner_text() fails, it automatically uses textContent which is more reliable.

        Tips:
        - Use selector="body" to extract all text from the page
        - Use specific selectors to extract text from specific elements
        - If extraction returns empty text, the page might have security verification
        - For security verification pages, navigate again with headless=False

        Example: Extract all text from the page body
        """

    @property
    def tags(self) -> list[str]:
        return ["browser", "automation", "scraping", "text"]

    def args_type(self) -> Type[BaseModel]:
        return BrowserExtractTextArgs

    def return_type(self) -> Type[BaseModel]:
        return BrowserExtractTextResult

    def run_json_sync(self, args: Mapping[str, Any]) -> Any:
        """
        Synchronous wrapper (not supported - use run_json_async instead).

        Browser tools are async-only. Please call them from async context.
        """
        raise NotImplementedError(
            f"{self.name} is async-only. Use await {self.name}() or call from async context."
        )

    async def run_json_async(self, args: Mapping[str, Any]) -> Any:
        result = await browser_extract_text(**args)
        return BrowserExtractTextResult(**result).model_dump()


class BrowserEvaluateTool(AbstractBaseTool):
    category = ToolCategory.BROWSER
    """Execute JavaScript code in the browser."""

    def __init__(self, task_id: Optional[str] = None):
        self._visibility = ToolVisibility.PUBLIC
        self._task_id = task_id

    @property
    def name(self) -> str:
        return "browser_evaluate"

    @property
    def description(self) -> str:
        return """Execute JavaScript code in the browser context.

        Use this tool for advanced interactions, data extraction, or custom logic.
        The browser session is created automatically if it doesn't exist.

        IMPORTANT: The browser runs in HEADLESS mode (no GUI) by default.
        If you encounter security verification or JS execution fails, try headless=False in browser_navigate.

        Example: Execute 'document.title' to get the page title
        """

    @property
    def tags(self) -> list[str]:
        return ["browser", "automation", "javascript", "advanced"]

    def args_type(self) -> Type[BaseModel]:
        return BrowserEvaluateArgs

    def return_type(self) -> Type[BaseModel]:
        return BrowserEvaluateResult

    def run_json_sync(self, args: Mapping[str, Any]) -> Any:
        """
        Synchronous wrapper (not supported - use run_json_async instead).

        Browser tools are async-only. Please call them from async context.
        """
        raise NotImplementedError(
            f"{self.name} is async-only. Use await {self.name}() or call from async context."
        )

    async def run_json_async(self, args: Mapping[str, Any]) -> Any:
        result = await browser_evaluate(**args)
        return BrowserEvaluateResult(**result).model_dump()


class BrowserListSessionsTool(AbstractBaseTool):
    category = ToolCategory.BROWSER
    """List all active browser sessions for debugging."""

    def __init__(self) -> None:
        self._visibility = ToolVisibility.PRIVATE  # Debug tool

    @property
    def name(self) -> str:
        return "browser_list_sessions"

    @property
    def description(self) -> str:
        return """List all active browser sessions (for debugging).

        Returns information about all active browser sessions, including:
        - Session IDs
        - Creation time
        - Last used time
        - Initialization status
        - Headless mode

        Use this tool to diagnose browser state issues.
        """

    @property
    def tags(self) -> list[str]:
        return ["browser", "debug", "internal"]

    def args_type(self) -> Type[BaseModel]:
        return BrowserCloseArgs  # Empty args, reuse for simplicity

    def return_type(self) -> Type[BaseModel]:
        return BrowserCloseResult  # Reuse for simplicity

    def run_json_sync(self, args: Mapping[str, Any]) -> Any:
        """
        Synchronous wrapper (not supported - use run_json_async instead).

        Browser tools are async-only. Please call them from async context.
        """
        raise NotImplementedError(
            f"{self.name} is async-only. Use await {self.name}() or call from async context."
        )

    async def run_json_async(self, args: Mapping[str, Any]) -> Any:
        result = await browser_list_sessions()
        return result


class BrowserSelectOptionTool(AbstractBaseTool):
    category = ToolCategory.BROWSER
    """Select an option from a dropdown."""

    def __init__(self, task_id: Optional[str] = None):
        self._visibility = ToolVisibility.PUBLIC
        self._task_id = task_id

    @property
    def name(self) -> str:
        return "browser_select_option"

    @property
    def description(self) -> str:
        return """Select an option from a select dropdown element.

        Use this tool to interact with dropdown menus and select lists.
        The browser session is created automatically if it doesn't exist.

        IMPORTANT: The browser runs in HEADLESS mode (no GUI) by default.
        If you encounter security verification or interaction issues, try headless=False in browser_navigate.

        Args:
            session_id: Session ID for the browser
            selector: CSS selector for the select element
            value: Option value to select (exclusive with index)
            index: Option index to select (exclusive with value)
            headless: Whether to run in headless mode (default: True)

        Tips:
        - Provide either value OR index, not both
        - Use browser_navigate with headless=False if you encounter security verification
        - Use browser_wait_for_selector to ensure the dropdown is ready before selecting

        Example: Select option with value='option1' from selector 'select#country'
        """

    @property
    def tags(self) -> list[str]:
        return ["browser", "automation", "interaction", "form"]

    def args_type(self) -> Type[BaseModel]:
        return BrowserSelectOptionArgs

    def return_type(self) -> Type[BaseModel]:
        return BrowserSelectOptionResult

    def run_json_sync(self, args: Mapping[str, Any]) -> Any:
        """
        Synchronous wrapper (not supported - use run_json_async instead).

        Browser tools are async-only. Please call them from async context.
        """
        raise NotImplementedError(
            f"{self.name} is async-only. Use await {self.name}() or call from async context."
        )

    async def run_json_async(self, args: Mapping[str, Any]) -> Any:
        result = await browser_select_option(**args)
        return BrowserSelectOptionResult(**result).model_dump()


class BrowserWaitForSelectorTool(AbstractBaseTool):
    category = ToolCategory.BROWSER
    """Wait for an element to appear on the page."""

    def __init__(self, task_id: Optional[str] = None):
        self._visibility = ToolVisibility.PUBLIC
        self._task_id = task_id

    @property
    def name(self) -> str:
        return "browser_wait_for_selector"

    @property
    def description(self) -> str:
        return """Wait for an element to appear on the page.

        Use this tool to ensure an element is ready before interacting with it.
        Useful for dynamic content that loads asynchronously.

        IMPORTANT: The browser runs in HEADLESS mode (no GUI) by default.
        If you encounter issues, try headless=False in browser_navigate.

        Args:
            session_id: Session ID for the browser
            selector: CSS selector or XPath to wait for
            timeout: Maximum time to wait in milliseconds (default: 30000 = 30 seconds)
            headless: Whether to run in headless mode (default: True)

        Use cases:
        - Wait for dynamic content to load
        - Ensure element exists before clicking
        - Wait for page transitions to complete

        Example: Wait for selector '.result-container' to appear
        """

    @property
    def tags(self) -> list[str]:
        return ["browser", "automation", "wait", "async"]

    def args_type(self) -> Type[BaseModel]:
        return BrowserWaitForSelectorArgs

    def return_type(self) -> Type[BaseModel]:
        return BrowserWaitForSelectorResult

    def run_json_sync(self, args: Mapping[str, Any]) -> Any:
        """
        Synchronous wrapper (not supported - use run_json_async instead).

        Browser tools are async-only. Please call them from async context.
        """
        raise NotImplementedError(
            f"{self.name} is async-only. Use await {self.name}() or call from async context."
        )

    async def run_json_async(self, args: Mapping[str, Any]) -> Any:
        result = await browser_wait_for_selector(**args)
        return BrowserWaitForSelectorResult(**result).model_dump()


class BrowserCloseTool(AbstractBaseTool):
    category = ToolCategory.BROWSER
    """Close a browser session and free resources."""

    def __init__(self, task_id: Optional[str] = None):
        self._visibility = ToolVisibility.PRIVATE  # Internal tool
        self._task_id = task_id

    @property
    def name(self) -> str:
        return "browser_close"

    @property
    def description(self) -> str:
        return """Close a browser session and free resources.

        Use this tool to explicitly close a browser session when done.
        Sessions are also automatically cleaned up after 30 minutes of inactivity.

        Example: Close session 'task-123'
        """

    @property
    def tags(self) -> list[str]:
        return ["browser", "cleanup", "internal"]

    def args_type(self) -> Type[BaseModel]:
        return BrowserCloseArgs

    def return_type(self) -> Type[BaseModel]:
        return BrowserCloseResult

    def run_json_sync(self, args: Mapping[str, Any]) -> Any:
        """
        Synchronous wrapper (not supported - use run_json_async instead).

        Browser tools are async-only. Please call them from async context.
        """
        raise NotImplementedError(
            f"{self.name} is async-only. Use await {self.name}() or call from async context."
        )

    async def run_json_async(self, args: Mapping[str, Any]) -> Any:
        result = await browser_close(**args)
        return BrowserCloseResult(**result).model_dump()


# ============== Factory Functions ==============


def create_browser_tools(
    task_id: Optional[str] = None, workspace: Optional["TaskWorkspace"] = None
) -> list:
    """
    Create all browser automation tools for a task.

    Args:
        task_id: Optional task ID for session tracking
        workspace: Optional workspace for saving screenshots

    Returns:
        List of browser tool instances
    """
    return [
        BrowserNavigateTool(task_id=task_id, workspace=workspace),
        BrowserClickTool(task_id=task_id),
        BrowserFillTool(task_id=task_id),
        BrowserScreenshotTool(task_id=task_id, workspace=workspace),
        BrowserExtractTextTool(task_id=task_id),
        BrowserEvaluateTool(task_id=task_id),
        BrowserSelectOptionTool(task_id=task_id),
        BrowserWaitForSelectorTool(task_id=task_id),
        BrowserCloseTool(task_id=task_id),
        BrowserListSessionsTool(),  # Debug tool (no task_id needed)
    ]
