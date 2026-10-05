"""Command-line interface: ingest, analyze, index, search, diagram, serve."""

from __future__ import annotations

import argparse
import asyncio
import sys
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path

from log_analytics.application.reporting.markdown import hits_table, render_result
from log_analytics.bootstrap import Container, run_with_container
from log_analytics.config import Settings
from log_analytics.domain.models import AnalysisRequest, Intent, KeywordMode, LogFilter, Severity
from log_analytics.infrastructure.logging import configure_logging


def _filter_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--start", type=datetime.fromisoformat, help="ISO 8601; naive = UTC")
    parser.add_argument("--end", type=datetime.fromisoformat, help="ISO 8601; naive = UTC")
    parser.add_argument("--host", action="append", default=[], dest="hosts", help="repeatable")
    parser.add_argument("--process", action="append", default=[], dest="processes", help="repeatable")
    parser.add_argument("--keyword", action="append", default=[], dest="keywords", help="repeatable")
    parser.add_argument("--keyword-mode", choices=[m.value for m in KeywordMode], default="any")
    parser.add_argument("--min-severity", choices=[s.value for s in Severity])


def _filter(args: argparse.Namespace) -> LogFilter:
    return LogFilter(
        start=args.start,
        end=args.end,
        hostnames=args.hosts,
        processes=args.processes,
        keywords=args.keywords,
        keyword_mode=KeywordMode(args.keyword_mode),
        min_severity=Severity(args.min_severity) if args.min_severity else None,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="log-analytics", description=__doc__)
    parser.add_argument("--parquet", help="override LA_PARQUET_PATH")
    parser.add_argument("--no-llm", action="store_true", help="deterministic analysis only")
    parser.add_argument("--log-level", default=None)
    sub = parser.add_subparsers(dest="command", required=True)

    ingest = sub.add_parser("ingest", help="convert raw syslog file(s) to Parquet (streaming)")
    ingest.add_argument("inputs", nargs="+")
    ingest.add_argument("--output", "-o", required=True)

    analyze = sub.add_parser("analyze", help="run the LangGraph analysis workflow")
    analyze.add_argument("--question", "-q")
    analyze.add_argument("--intent", choices=[i.value for i in Intent])
    analyze.add_argument("--format", choices=["markdown", "json"], default="markdown")
    analyze.add_argument("--output", "-o", help="write to file instead of stdout")
    _filter_args(analyze)

    index = sub.add_parser("index", help="embed log patterns into pgvector")
    _filter_args(index)

    search = sub.add_parser("search", help="semantic search over indexed patterns")
    search.add_argument("query")
    search.add_argument("-k", type=int, default=8)

    sub.add_parser("diagram", help="print the LangGraph workflow as Mermaid")

    serve = sub.add_parser("serve", help="run the API server")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    return parser


def _emit(text: str, output: str | None) -> None:
    if output:
        Path(output).write_text(text, encoding="utf-8")
        print(f"wrote {output}", file=sys.stderr)
    else:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
        sys.stdout.write(text)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    overrides: dict[str, object] = {}
    if args.parquet:
        overrides["parquet_path"] = args.parquet
    if args.no_llm:
        overrides["llm_enabled"] = False
    if args.log_level:
        overrides["log_level"] = args.log_level
    settings = Settings(**overrides)  # type: ignore[arg-type]
    configure_logging(settings.log_level, json=settings.log_json)

    if args.command == "ingest":
        from log_analytics.infrastructure.ingest import ingest_syslog

        rows = ingest_syslog(args.inputs, args.output)
        print(f"wrote {rows:,} rows to {args.output}", file=sys.stderr)
        return 0
    if args.command == "serve":
        import uvicorn

        from log_analytics.api.app import create_app

        uvicorn.run(create_app(settings), host=args.host, port=args.port, log_config=None)
        return 0

    async def analyze(c: Container) -> str:
        request = AnalysisRequest(
            question=args.question, filter=_filter(args), intent=Intent(args.intent) if args.intent else None
        )
        result = await c.workflow.run(request)
        return result.model_dump_json(indent=2) if args.format == "json" else render_result(result)

    async def index(c: Container) -> str:
        if c.indexer is None:
            raise SystemExit("Vector store unavailable: set LA_DATABASE_URL")
        return (await c.indexer.index(_filter(args))).model_dump_json(indent=2) + "\n"

    async def search(c: Container) -> str:
        if c.retriever is None:
            raise SystemExit("Vector store unavailable: set LA_DATABASE_URL")
        return hits_table(await c.retriever.search(args.query, k=args.k)) + "\n"

    async def diagram(c: Container) -> str:
        return c.workflow.mermaid()

    commands = {"analyze": analyze, "index": index, "search": search, "diagram": diagram}
    output = asyncio.run(run_with_container(settings, commands[args.command]))
    _emit(output, getattr(args, "output", None))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
