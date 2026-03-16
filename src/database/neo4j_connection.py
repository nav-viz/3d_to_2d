"""Neo4j connection with graceful fallback to an in-memory mock."""
from __future__ import annotations

from typing import Any, Dict, List, Optional
from src.utils.logger import get_logger

logger = get_logger(__name__)

try:
    from neo4j import GraphDatabase as _GraphDatabase
    NEO4J_AVAILABLE = True
    logger.info("neo4j-driver is available.")
except ImportError:
    NEO4J_AVAILABLE = False
    logger.info("neo4j-driver not available – using mock connection.")


class _MockSession:
    """In-memory session that stores query results in a list."""

    def __init__(self, store: List[Dict]) -> None:
        self._store = store

    def run(self, query: str, **params: Any) -> "_MockResult":
        self._store.append({"query": query, "params": params})
        return _MockResult([])

    def close(self) -> None:
        pass

    def __enter__(self) -> "_MockSession":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()


class _MockResult:
    """Minimal mock result that supports list-style iteration."""

    def __init__(self, data: List[Dict]) -> None:
        self._data = data

    def data(self) -> List[Dict]:
        """Return stored result rows."""
        return self._data

    def __iter__(self):
        return iter(self._data)


class Neo4jConnection:
    """
    Manages a connection to a Neo4j database instance.

    When the ``neo4j`` Python driver is not installed (or the server is
    unreachable), all operations silently fall back to an in-memory mock so
    the rest of the pipeline continues to work.

    Usage::

        with Neo4jConnection(uri, user, password) as conn:
            results = conn.run_query("MATCH (n) RETURN n LIMIT 5")
    """

    def __init__(
        self,
        uri: str = "bolt://localhost:7687",
        user: str = "neo4j",
        password: str = "password",
    ) -> None:
        self._uri = uri
        self._user = user
        self._password = password
        self._driver: Any = None
        self._mock_store: List[Dict] = []
        self._log = get_logger(self.__class__.__name__)

    # ------------------------------------------------------------------
    # Connection lifecycle
    # ------------------------------------------------------------------

    def connect(self) -> "Neo4jConnection":
        """
        Open the database connection.

        Returns:
            *self* for chaining / context manager use.
        """
        if NEO4J_AVAILABLE:
            try:
                self._driver = _GraphDatabase.driver(
                    self._uri,
                    auth=(self._user, self._password),
                )
                self._driver.verify_connectivity()
                self._log.info("Connected to Neo4j at %s.", self._uri)
            except Exception as exc:
                self._log.warning(
                    "Cannot reach Neo4j (%s) – falling back to mock.", exc
                )
                self._driver = None
        else:
            self._log.info("Using in-memory mock Neo4j connection.")
        return self

    def close(self) -> None:
        """Close the underlying driver connection if open."""
        if self._driver is not None:
            try:
                self._driver.close()
            except Exception as exc:
                self._log.warning("Error closing Neo4j driver: %s", exc)
            finally:
                self._driver = None

    # ------------------------------------------------------------------
    # Context manager support
    # ------------------------------------------------------------------

    def __enter__(self) -> "Neo4jConnection":
        return self.connect()

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Query execution
    # ------------------------------------------------------------------

    def run_query(
        self,
        query: str,
        parameters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict]:
        """
        Execute a Cypher query and return result rows.

        Args:
            query: Cypher query string.
            parameters: Optional parameter dictionary.

        Returns:
            List of row dictionaries.
        """
        params = parameters or {}
        if self._driver is not None:
            try:
                with self._driver.session() as session:
                    result = session.run(query, **params)
                    return result.data()
            except Exception as exc:
                self._log.error("Query execution failed: %s", exc)
                return []
        else:
            # Mock execution
            self._mock_store.append({"query": query, "params": params})
            return []

    @property
    def is_connected(self) -> bool:
        """Return True when a live driver is active."""
        return self._driver is not None

    @property
    def mock_log(self) -> List[Dict]:
        """Return the list of mock queries executed."""
        return list(self._mock_store)
