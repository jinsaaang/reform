"""Tool for retrieving articles from the database."""

from src.tools.base.database_mixin import DatabaseAwareTool
from src.tools.base.base import ToolResponseMixin
from src.tools.base.output_models import ArticleRetrievalOutput
from src.domain.models import Article
from src.tools.base.schema_helper import pydantic_to_output_schema


class ArticleRetrievalTool(DatabaseAwareTool, ToolResponseMixin):
    """Tool that retrieves full article content by article ID.

    Use this when you need the complete article text for an article
    you've identified (e.g., from event details or article lists).
    """

    name = "article_retrieval"
    description = """Retrieve full article content by article ID.

    Use this tool when you have an article ID and need to read its full content.
    Article IDs can be found in event details (event.article_ids) or other sources.
    """

    inputs = {
        "article_id": {
            "type": "string",
            "description": "The ID of the article to retrieve",
        },
        "query": {
            "type": "string",
            "description": "Optional keywords for returning only relevant excerpts",
            "nullable": True,
        },
        "max_chars": {
            "type": "integer",
            "description": "Maximum returned content characters (default: 3000)",
            "default": 3000,
            "nullable": True,
        },
    }
    output_type = "object"
    output_schema = pydantic_to_output_schema(ArticleRetrievalOutput)

    def __init__(self, db=None, db_path: str = None):
        """Initialize the article retrieval tool.

        Args:
            db: Optional GenericDatabase instance
            db_path: Optional path to database file (creates new GenericDatabase if provided)

        Note:
            If neither db nor db_path is provided, will use default database path
        """
        super().__init__(db=db, db_path=db_path, ensure_tables=[Article])

    def forward(
        self,
        article_id: str,
        query: str = None,
        max_chars: int = 3_000,
    ) -> ArticleRetrievalOutput:
        """Retrieve article by ID.

        Args:
            article_id: Article ID to retrieve

        Returns:
            ArticleRetrievalOutput Pydantic model with full article content
        """
        from src.domain.models import Article

        # Fetch article from database
        article = self.db.get(Article, article_id)

        if not article:
            import json
            error_json = self.not_found_response("Article", article_id, Article)
            try:
                error_dict = json.loads(error_json)
                error_msg = error_dict.get("error", "Not found")
                avail = ", ".join(error_dict.get("available_items", []))
                content_msg = f"Error: {error_msg}. Available items: {avail}"
            except Exception:
                content_msg = f"Error: {error_json}"
            return ArticleRetrievalOutput(
                id="error",
                title="Error",
                url="",
                content=content_msg,
                word_count=0
            )

        original_content = article.content
        content = self._select_content(
            original_content,
            query=query,
            max_chars=max_chars,
        )

        # Return Pydantic output model directly (smolagents passes through as-is)
        return ArticleRetrievalOutput(
            id=article.id,
            title=article.title,
            url=article.url,
            content=content,
            source=article.source,
            published_date=article.published_date.isoformat()
            if article.published_date
            else None,
            word_count=article.word_count,
            original_char_count=len(original_content),
            content_truncated=len(content) < len(original_content),
        )

    @staticmethod
    def _select_content(content: str, query: str = None, max_chars: int = 3_000) -> str:
        """Return a bounded full-text prefix or query-centered excerpts."""
        max_chars = max(500, min(int(max_chars or 3_000), 50_000))
        if len(content) <= max_chars:
            return content

        terms = [term.lower() for term in (query or "").split() if len(term) >= 3]
        if not terms:
            return content[:max_chars] + "\n\n[Content truncated]"

        lowered = content.lower()
        positions = sorted(
            {
                position
                for term in terms
                if (position := lowered.find(term)) >= 0
            }
        )
        if not positions:
            return content[:max_chars] + "\n\n[Content truncated]"

        excerpts = []
        remaining = max_chars
        for position in positions[:5]:
            if remaining <= 0:
                break
            radius = min(1_500, remaining // 2)
            start = max(0, position - radius)
            end = min(len(content), position + radius)
            excerpt = content[start:end].strip()
            if excerpt:
                excerpts.append(excerpt)
                remaining -= len(excerpt)
        joined = "\n\n[... relevant excerpt ...]\n\n".join(excerpts)
        return joined[:max_chars] + "\n\n[Query-focused excerpts]"
