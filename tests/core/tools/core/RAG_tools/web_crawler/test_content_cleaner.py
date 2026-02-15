"""Unit tests for content cleaner."""

from xagent.core.tools.core.RAG_tools.web_crawler.content_cleaner import ContentCleaner


class TestContentCleaner:
    """Test content cleaning functionality."""

    def test_extract_title_from_title_tag(self):
        """Test title extraction from <title> tag."""
        cleaner = ContentCleaner()

        html = """
        <html>
            <head><title>Test Page Title</title></head>
            <body>Content</body>
        </html>
        """

        result = cleaner.clean_and_convert(html, "https://example.com")
        assert result["title"] == "Test Page Title"

    def test_extract_title_from_h1(self):
        """Test title extraction from <h1> tag."""
        cleaner = ContentCleaner()

        html = """
        <html>
            <body>
                <h1>Main Heading</h1>
                <p>Content</p>
            </body>
        </html>
        """

        result = cleaner.clean_and_convert(html, "https://example.com")
        assert result["title"] == "Main Heading"

    def test_extract_title_from_meta(self):
        """Test title extraction from meta tag."""
        cleaner = ContentCleaner()

        html = """
        <html>
            <head>
                <meta property="og:title" content="Meta Title" />
            </head>
            <body>Content</body>
        </html>
        """

        result = cleaner.clean_and_convert(html, "https://example.com")
        assert result["title"] == "Meta Title"

    def test_remove_script_and_style(self):
        """Test removal of script and style elements."""
        cleaner = ContentCleaner()

        html = """
        <html>
            <body>
                <h1>Title</h1>
                <script>alert('test');</script>
                <style>body { color: red; }</style>
                <p>Content</p>
            </body>
        </html>
        """

        result = cleaner.clean_and_convert(html, "https://example.com")
        markdown = result["content_markdown"]
        assert "alert" not in markdown
        assert "color: red" not in markdown
        assert "Content" in markdown

    def test_html_to_markdown_conversion(self):
        """Test basic HTML to Markdown conversion."""
        cleaner = ContentCleaner()

        html = """
        <html>
            <body>
                <h1>Heading 1</h1>
                <h2>Heading 2</h2>
                <p>This is a paragraph.</p>
                <ul>
                    <li>Item 1</li>
                    <li>Item 2</li>
                </ul>
            </body>
        </html>
        """

        result = cleaner.clean_and_convert(html, "https://example.com")
        markdown = result["content_markdown"]

        assert "# Heading 1" in markdown
        assert "## Heading 2" in markdown
        assert "This is a paragraph" in markdown
        assert "* Item 1" in markdown or "- Item 1" in markdown

    def test_custom_remove_selectors(self):
        """Test custom element removal."""
        cleaner = ContentCleaner(remove_selectors=["nav", "footer", ".ad"])

        html = """
        <html>
            <body>
                <nav>Navigation</nav>
                <h1>Title</h1>
                <div class="ad">Advertisement</div>
                <p>Content</p>
                <footer>Footer</footer>
            </body>
        </html>
        """

        result = cleaner.clean_and_convert(html, "https://example.com")
        markdown = result["content_markdown"]

        assert "Navigation" not in markdown
        assert "Advertisement" not in markdown
        assert "Footer" not in markdown
        assert "Content" in markdown

    def test_content_selector_extraction(self):
        """Test extraction using CSS selector."""
        cleaner = ContentCleaner(content_selector="article")

        html = """
        <html>
            <body>
                <nav>Navigation</nav>
                <article>
                    <h1>Article Title</h1>
                    <p>Article content</p>
                </article>
                <footer>Footer</footer>
            </body>
        </html>
        """

        result = cleaner.clean_and_convert(html, "https://example.com")
        markdown = result["content_markdown"]

        assert "Article Title" in markdown
        assert "Article content" in markdown

    def test_content_selector_not_found(self):
        """Test behavior when content selector is not found."""
        cleaner = ContentCleaner(content_selector="article")

        html = """
        <html>
            <body>
                <h1>Title</h1>
                <p>Content</p>
            </body>
        </html>
        """

        # Should not crash, fallback to full content
        result = cleaner.clean_and_convert(html, "https://example.com")
        assert "Title" in result["content_markdown"]

    def test_is_valid_content(self):
        """Test content validation."""
        cleaner = ContentCleaner()

        # Valid content (longer than default min_length=100)
        long_content = "This is valid content with enough text. " * 5  # ~250 chars
        assert cleaner.is_valid_content(long_content) is True

        # Too short
        assert cleaner.is_valid_content("Short") is False

        # Empty
        assert cleaner.is_valid_content("") is False

        # Only whitespace
        assert cleaner.is_valid_content("   \n\n   ") is False

    def test_min_length_parameter(self):
        """Test custom minimum length."""
        cleaner = ContentCleaner()

        # With custom min_length
        assert cleaner.is_valid_content("a" * 50, min_length=50) is True
        assert cleaner.is_valid_content("a" * 50, min_length=100) is False

    def test_links_in_markdown(self):
        """Test that links are preserved in Markdown."""
        cleaner = ContentCleaner()

        html = """
        <html>
            <body>
                <a href="https://example.com">Example Link</a>
            </body>
        </html>
        """

        result = cleaner.clean_and_convert(html, "https://example.com")
        markdown = result["content_markdown"]

        assert "Example Link" in markdown
        assert "https://example.com" in markdown

    def test_images_in_markdown(self):
        """Test that images are preserved in Markdown."""
        cleaner = ContentCleaner()

        html = """
        <html>
            <body>
                <img src="image.jpg" alt="Test Image" />
            </body>
        </html>
        """

        result = cleaner.clean_and_convert(html, "https://example.com")
        markdown = result["content_markdown"]

        assert "image.jpg" in markdown
        assert "Test Image" in markdown

    def test_code_blocks(self):
        """Test handling of code blocks."""
        cleaner = ContentCleaner()

        html = """
        <html>
            <body>
                <pre><code>def hello():
    print("Hello, World!")
</code></pre>
            </body>
        </html>
        """

        result = cleaner.clean_and_convert(html, "https://example.com")
        markdown = result["content_markdown"]

        assert "def hello():" in markdown

    def test_tables(self):
        """Test handling of HTML tables."""
        cleaner = ContentCleaner()

        html = """
        <html>
            <body>
                <table>
                    <tr><th>Header 1</th><th>Header 2</th></tr>
                    <tr><td>Data 1</td><td>Data 2</td></tr>
                </table>
            </body>
        </html>
        """

        result = cleaner.clean_and_convert(html, "https://example.com")
        markdown = result["content_markdown"]

        assert "Header 1" in markdown
        assert "Data 1" in markdown
