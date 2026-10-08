"""`support-agent` command line."""

from __future__ import annotations

import asyncio
import json
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Annotated, Any

import typer
from rich.console import Console
from rich.table import Table

from support_agent.cli_drafts import drafts_app
from support_agent.core.logging import configure_logging
from support_agent.core.principal import Principal
from support_agent.core.settings import Settings, get_settings

app = typer.Typer(help="Customer support agent (RAG + MCP data access).", no_args_is_help=True)
app.add_typer(drafts_app, name="drafts")
console = Console()


def _utf8() -> None:
    """Vietnamese text must survive Windows consoles that default to a legacy code page."""
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")


@app.callback()
def _main() -> None:
    _utf8()
    settings = get_settings()
    configure_logging(settings.log_level)


def _settings() -> Settings:
    return get_settings()


# --- init ---------------------------------------------------------------------------


@app.command()
def init() -> None:
    """Create .env from the template and the local data directories."""
    env, example = Path(".env"), Path(".env.example")
    if env.exists():
        console.print("[yellow].env already exists, left untouched.[/yellow]")
    elif example.exists():
        shutil.copy(example, env)
        console.print("[green]Created .env[/green] - add your provider API key.")
    else:
        console.print("[red].env.example not found; run from the repository root.[/red]")
        raise typer.Exit(1)
    for d in ("data", "knowledge", "config"):
        Path(d).mkdir(exist_ok=True)
    drafts_url = get_settings().drafts_db_url
    if drafts_url:  # SPEC 10.1: init creates support_drafts when it can reach the drafts database
        from support_agent.drafts.repository import create_repository

        async def make_table() -> None:
            repo = create_repository(drafts_url)
            try:
                await repo.create_schema()
            finally:
                await repo.close()

        try:
            asyncio.run(make_table())
            console.print("[green]support_drafts table is ready.[/green]")
        except Exception as exc:
            console.print(
                f"[yellow]Could not create support_drafts ({type(exc).__name__}). "
                "Use `support-agent drafts init --url <owner url>`.[/yellow]"
            )
    console.print("Next: edit .env, then `support-agent seed-demo` and `support-agent ingest`.")


# --- ingest -------------------------------------------------------------------------


@app.command()
def ingest(
    full: Annotated[bool, typer.Option("--full", help="Rebuild the whole index")] = False,
    directory: Annotated[Path | None, typer.Option("--dir", help="Knowledge directory")] = None,
) -> None:
    """Load policy documents (.md/.pdf/.docx) into Qdrant. Incremental by content hash."""
    from support_agent.llm.factory import get_embeddings
    from support_agent.rag.index import VectorStore, make_client
    from support_agent.rag.ingest import ingest as run_ingest
    from support_agent.rag.sparse import SparseEncoder

    s = _settings()
    cfg = s.app.retrieval
    root = directory or s.app.knowledge.dir
    store = VectorStore(make_client(s), cfg.collection)
    try:
        sparse = (
            SparseEncoder(cfg.sparse_model, cache_dir=s.app.embeddings.cache_dir)
            if cfg.hybrid
            else None
        )
        report = run_ingest(
            root,
            store=store,
            embeddings=get_embeddings(s),
            sparse=sparse,
            embedding_model=s.embedding_model_name,
            cfg=cfg,
            full=full,
        )
        total = store.count()
    finally:
        store.close()
    console.print(f"[bold]ingest[/bold] {report.summary()} (index now holds {total} chunks)")
    for doc, err in report.failed.items():
        console.print(f"[red]FAILED[/red] {doc}: {err}")
    for doc in report.removed:
        console.print(f"removed: {doc}")
    if report.failed:
        raise typer.Exit(1)


# --- ask ----------------------------------------------------------------------------


def _principal(user: str, role: str) -> Principal:
    if role not in ("customer", "staff"):
        raise typer.BadParameter("role must be customer or staff")
    return Principal(user_id=user, role=role)  # type: ignore[arg-type]


def _print_footer(data: dict[str, Any]) -> None:
    if data.get("citations"):
        console.print("[dim]Sources:[/dim]")
        for c in data["citations"]:
            console.print(f"[dim]  - {c['source']} > {c['section']}[/dim]")
    usage = data.get("usage", {})
    console.print(
        f"[dim]route={data.get('route')} outcome={data.get('outcome')} "
        f"tools={data.get('tool_calls')} steps={data.get('steps')} "
        f"tokens={usage.get('input_tokens')}+{usage.get('output_tokens')} "
        f"{data.get('latency_ms')}ms session={data.get('session_id')}[/dim]"
    )


async def _show_events(events: Any) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    """Print events as they stream. Returns (`done` data, open confirmation)."""
    result: dict[str, Any] | None = None
    stop: dict[str, Any] | None = None
    async for event in events:
        data = event.data
        if event.kind == "tool_start":
            console.print(f"[dim]  -> {data['tool']} {data['args_summary']}[/dim]")
        elif event.kind == "token":
            console.print(data["text"], end="", markup=False, highlight=False, soft_wrap=True)
        elif event.kind == "done":
            result = data
        elif event.kind == "interrupt":
            stop = data
        elif event.kind == "error":
            console.print(f"\n[red]{data['message']}[/red]")
    console.print()
    return result, stop


async def _ask_customer(stop: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """Ask on the terminal what the customer decides. Returns (decision, edits)."""
    from support_agent.cli_drafts import summary_lines

    for line in summary_lines(stop):
        console.print(line, markup=False, highlight=False)
    editable = stop.get("editable_fields", [])
    while True:
        try:
            choice = (await asyncio.to_thread(input, "Submit? [y]es / [n]o / [e]dit > ")).strip()
        except (EOFError, KeyboardInterrupt):
            return "reject", {}
        if choice.lower() in ("y", "yes"):
            return "approve", {}
        if choice.lower() in ("n", "no", ""):
            return "reject", {}
        if choice.lower() in ("e", "edit"):
            edits: dict[str, Any] = {}
            for name in editable:
                value = (await asyncio.to_thread(input, f"{name} (Enter keeps it) > ")).strip()
                if value:
                    edits[name] = value
            return "edit", edits


async def _stream_turn(
    agent: Any,
    question: str,
    principal: Principal,
    *,
    session: str | None,
    lang: str,
    interactive: bool = True,
) -> dict[str, Any] | None:
    """Print one turn as it streams (tool calls, then the answer) and return its `done` data.

    When the turn stops for a confirmation it is asked on the terminal (or, without a terminal,
    left open so `support-agent resume` can answer it later).
    """
    result, stop = await _show_events(
        agent.stream(question, principal, session_id=session, language=lang)
    )
    while stop is not None:
        if not interactive:
            console.print(
                "[yellow]Waiting for the customer's confirmation. Answer it with:[/yellow]\n"
                f"  support-agent resume --user {principal.user_id} --session {stop['session_id']} "
                f"--id {stop['id']} --decision approve|reject"
            )
            return {"session_id": stop["session_id"], "outcome": "confirmation", "interrupt": stop}
        decision, edits = await _ask_customer(stop)
        console.print("assistant> ", end="")
        result, stop = await _show_events(
            agent.resume(principal, stop["session_id"], stop["id"], decision, edits=edits or None)
        )
    return result


@app.command()
def ask(
    question: Annotated[str, typer.Argument(help="The customer's question")],
    user: Annotated[str, typer.Option("--user", "-u", help="Authenticated customer id")] = "u_100",
    role: Annotated[str, typer.Option(help="customer | staff")] = "customer",
    lang: Annotated[str, typer.Option(help="auto | vi | en")] = "auto",
    no_tools: Annotated[bool, typer.Option("--no-tools", help="Policy questions only")] = False,
    as_json: Annotated[bool, typer.Option("--json", help="Print the full response")] = False,
    engine: Annotated[str, typer.Option(help="agent | pipeline (the Phase A baseline)")] = "agent",
    session: Annotated[str | None, typer.Option(help="Continue this conversation")] = None,
) -> None:
    """Ask a question as an authenticated user (stand-in for the JWT identity)."""
    from support_agent.runtime import open_agent, open_pipeline

    principal = _principal(user, role)
    if engine not in ("agent", "pipeline"):
        raise typer.BadParameter("engine must be agent or pipeline")

    async def run_pipeline() -> None:
        async with open_pipeline(_settings(), with_tools=not no_tools) as rag:
            result = await rag.answer(question, principal, language=lang)
        if as_json:
            console.print_json(result.model_dump_json())
            return
        console.print(result.answer)
        _print_footer(
            result.model_dump() | {"citations": [c.model_dump() for c in result.citations]}
        )

    async def run_agent() -> None:
        async with open_agent(_settings(), with_tools=not no_tools) as agent:
            if as_json:
                result = await agent.answer(question, principal, language=lang, session_id=session)
                console.print_json(result.model_dump_json())
                return
            data = await _stream_turn(
                agent,
                question,
                principal,
                session=session,
                lang=lang,
                interactive=sys.stdin.isatty(),
            )
            if data and data.get("outcome") != "confirmation":
                _print_footer(data)

    asyncio.run(run_pipeline() if engine == "pipeline" else run_agent())


@app.command()
def resume(
    session: Annotated[str, typer.Option(help="The conversation that is waiting")],
    confirmation: Annotated[str, typer.Option("--id", help="The confirmation id it printed")],
    decision: Annotated[str, typer.Option(help="approve | reject | edit")],
    user: Annotated[str, typer.Option("--user", "-u", help="Authenticated customer id")] = "u_100",
    edit: Annotated[
        list[str] | None, typer.Option("--set", help="With edit: field=value (repeatable)")
    ] = None,
) -> None:
    """Answer a confirmation an earlier `ask` left open (works across restarts)."""
    from support_agent.runtime import open_agent

    if decision not in ("approve", "reject", "edit"):
        raise typer.BadParameter("decision must be approve, reject or edit")
    edits = dict(pair.split("=", 1) for pair in edit or [] if "=" in pair)
    principal = _principal(user, "customer")

    async def run() -> None:
        async with open_agent(_settings()) as agent:
            events = agent.resume(principal, session, confirmation, decision, edits=edits or None)
            data, stop = await _show_events(events)
            if stop is not None:
                console.print(
                    f"[yellow]Asked again. New id: {stop['id']} (valid until "
                    f"{stop['expires_at']})[/yellow]"
                )
            elif data:
                _print_footer(data)

    asyncio.run(run())


@app.command()
def chat(
    user: Annotated[str, typer.Option("--user", "-u", help="Authenticated customer id")] = "u_100",
    role: Annotated[str, typer.Option(help="customer | staff")] = "customer",
    lang: Annotated[str, typer.Option(help="auto | vi | en")] = "auto",
    no_tools: Annotated[bool, typer.Option("--no-tools", help="Policy questions only")] = False,
    session: Annotated[str | None, typer.Option(help="Resume this conversation")] = None,
) -> None:
    """Talk to the agent over several turns. The conversation is saved and can be resumed."""
    from support_agent.runtime import open_agent

    principal = _principal(user, role)

    async def run() -> None:
        async with open_agent(_settings(), with_tools=not no_tools) as agent:
            sid = session
            console.print("[dim]Empty line or /exit to quit.[/dim]")
            while True:
                try:
                    question = (await asyncio.to_thread(input, "you> ")).strip()
                except (EOFError, KeyboardInterrupt):
                    break
                if not question or question in ("/exit", "/quit"):
                    break
                console.print("assistant> ", end="")
                data = await _stream_turn(agent, question, principal, session=sid, lang=lang)
                if data:
                    sid = data["session_id"]
                    _print_footer(data)
            if sid:
                console.print(
                    f"[dim]Resume: support-agent chat --user {user} --session {sid}[/dim]"
                )

    asyncio.run(run())


# --- seed-demo / validate-mapping ---------------------------------------------------------------


@app.command("seed-demo")
def seed_demo(
    url: Annotated[
        str | None,
        typer.Option(help="DB URL with WRITE access (default: BUSINESS_DB_URL)"),
    ] = None,
    reset: Annotated[bool, typer.Option(help="Drop and recreate the demo tables")] = True,
) -> None:
    """Create the demo shop database. Needs an account that can create tables."""
    from support_agent.mcp_db.factory import normalise_db_url
    from support_agent.seed.demo import seed_mongo, seed_sql

    s = _settings()
    target = url or s.business_db_url
    if not target:
        console.print("[red]Set BUSINESS_DB_URL or pass --url.[/red]")
        raise typer.Exit(1)
    target = normalise_db_url(s.business_db_type, target)
    seeder = seed_mongo if s.business_db_type == "mongodb" else seed_sql
    try:
        counts = asyncio.run(seeder(target, reset=reset))
    except Exception as exc:
        console.print(
            f"[red]Seeding failed ({type(exc).__name__}).[/red] The agent's read-only account "
            "cannot create tables; pass an owner account with --url."
        )
        raise typer.Exit(1) from exc
    console.print(f"[green]Seeded demo data:[/green] {json.dumps(counts)}")
    console.print(
        "Demo customers: u_100, u_101, u_102. Try: support-agent ask 'Where is order #1236?'"
    )


@app.command("validate-mapping")
def validate_mapping() -> None:
    """Check the schema mapping against the live business database."""
    from support_agent.mcp_db.factory import ConfigError, create_adapter
    from support_agent.mcp_db.mapping import MappingError, load_mapping

    s = _settings()
    try:
        mapping = load_mapping(s.app.mapping.path)
        adapter = create_adapter(s, mapping)
    except (MappingError, ConfigError) as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from exc

    async def run() -> list:
        try:
            return await adapter.validate()
        finally:
            await adapter.close()

    issues = asyncio.run(run())
    if not issues:
        console.print(f"[green]Mapping OK[/green] ({s.app.mapping.path}, {mapping.dialect})")
        return
    table = Table("level", "entity", "message")
    for i in issues:
        table.add_row(
            "[red]error[/red]" if i.level == "error" else "[yellow]warning[/yellow]",
            i.entity,
            i.message,
        )
    console.print(table)
    if any(i.level == "error" for i in issues):
        raise typer.Exit(1)


# --- eval ---------------------------------------------------------------------------


@app.command("eval")
def evaluate(
    dataset: Annotated[Path | None, typer.Option(help="JSONL dataset (default: config)")] = None,
    subset: Annotated[
        str | None, typer.Option(help="Only samples tagged with this subset, e.g. ci")
    ] = None,
    limit: Annotated[int | None, typer.Option(help="Only the first N samples")] = None,
    offline: Annotated[
        bool, typer.Option("--offline", help="No LLM: retrieval + heuristic routing")
    ] = False,
    judge: Annotated[
        bool, typer.Option("--judge", help="Also grade answers with the LLM judge")
    ] = False,
    name: Annotated[str | None, typer.Option(help="Report file name (default derived)")] = None,
    fail_under: Annotated[
        bool, typer.Option("--fail-under", help="Exit 1 below any threshold or on a regression")
    ] = False,
    engine: Annotated[str, typer.Option(help="agent | pipeline (the Phase A baseline)")] = "agent",
    compare: Annotated[
        Path | None,
        typer.Option(help="Report JSON to compare against (default: the pipeline baseline)"),
    ] = None,
    concurrency: Annotated[
        int | None, typer.Option(help="Questions evaluated in parallel (default: config)")
    ] = None,
    run_timeout: Annotated[
        float | None,
        typer.Option(help="Seconds allowed per question (raise it on rate-limited API tiers)"),
    ] = None,
    rescore: Annotated[
        Path | None,
        typer.Option(help="Grade a saved report again with the current dataset. Calls no model"),
    ] = None,
) -> None:
    """Evaluate retrieval, routing and (with an LLM) the whole system; write a report.

    With the agent engine the result is compared with the Phase A pipeline baseline, and a
    metric that drops counts as a failure under --fail-under (PLAN Phase 4 acceptance).
    """
    from support_agent.agent.prompts import PROMPT_VERSION
    from support_agent.evals.dataset import DatasetError, check_dataset, load_dataset
    from support_agent.evals.judge import LLMJudge
    from support_agent.evals.report import (
        EvalMeta,
        EvalReport,
        compare_reports,
        now_iso,
        render_markdown,
        trace_session,
        write_report,
    )
    from support_agent.evals.runner import (
        check_thresholds,
        evaluate_heuristic_routing,
        evaluate_retrieval,
        run_pipeline_eval,
        summarise_pipeline,
    )
    from support_agent.llm.factory import get_chat_model
    from support_agent.observability.langfuse import create_tracing
    from support_agent.runtime import build_retriever, open_agent, open_pipeline, open_store

    s = _settings()
    cfg = s.app.evals
    if run_timeout:
        s.app.agent.run_timeout_seconds = run_timeout
    path = dataset or cfg.dataset
    try:
        samples = load_dataset(path, subset=subset, limit=limit)
    except DatasetError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from exc
    if not samples:
        console.print("[red]No samples selected.[/red]")
        raise typer.Exit(1)
    if not subset and not limit:
        for problem in check_dataset(samples):
            console.print(f"[yellow]dataset warning:[/yellow] {problem}")

    if engine not in ("agent", "pipeline"):
        raise typer.BadParameter("engine must be agent or pipeline")

    if rescore is not None:
        from support_agent.evals.runner import rescore_pipeline

        if not rescore.exists():
            console.print(f"[red]Report not found: {rescore}[/red]")
            raise typer.Exit(1)
        saved = json.loads(rescore.read_text(encoding="utf-8"))
        if not (saved.get("pipeline") or {}).get("records"):
            console.print(
                "[red]That report has no per-question records (it predates --rescore); "
                "run the evaluation again.[/red]"
            )
            raise typer.Exit(1)
        pipeline = rescore_pipeline(saved, samples)
        meta = EvalMeta.model_validate(
            {**saved["meta"], "created_at": now_iso(), "rescored_from": rescore.stem}
        )
        meta.n_samples = len(samples)
        report = EvalReport(
            meta=meta,
            pipeline=pipeline,
            thresholds=check_thresholds(pipeline.metrics, cfg.thresholds),
        )
        baseline = compare
        if baseline is not None and baseline.exists():
            report.comparison = compare_reports(
                pipeline, json.loads(baseline.read_text(encoding="utf-8")), against=baseline.stem
            )
        out_name = name or f"{rescore.stem}-rescored"
        md, js = write_report(report, cfg.report_dir, out_name)
        console.print(render_markdown(report))
        console.print(f"[dim]Report written to {md} and {js}[/dim]")
        if fail_under and not report.passed:
            console.print("[red]Below threshold.[/red]")
            raise typer.Exit(1)
        return

    mode = "offline" if offline else "full"
    # The Phase A pipeline keeps its historical names (baseline-full); the agent gets its own.
    partial = bool(subset or limit or dataset)
    tag = subset or "partial"
    if offline or engine == "pipeline":
        run_name = name or (f"{mode}-{tag}" if partial else f"baseline-{mode}")
    else:
        run_name = name or (f"agent-{tag}" if partial else "agent-full")
    tracing = create_tracing(s) if not offline else None
    k_thr = s.app.retrieval.score_threshold
    session = trace_session(run_name)  # unique per run: reruns must not overwrite traces

    async def run() -> EvalReport:
        meta = EvalMeta(
            created_at=now_iso(),
            mode=mode,
            dataset=str(path),
            n_samples=len(samples),
            subset=subset,
            provider=None if offline else s.llm_provider,
            model=None if offline else s.chat_model_name,
            embedding_model=s.embedding_model_name,
            score_threshold=k_thr,
            router="heuristic" if offline else "llm",
            judge=judge and not offline,
            run_name=run_name,
            tracing=bool(tracing and tracing.enabled),
            trace_session=session if tracing and tracing.enabled else None,
            engine="pipeline" if offline else engine,
            run_timeout_seconds=None if offline else s.app.agent.run_timeout_seconds,
            prompt_version=PROMPT_VERSION if engine == "agent" and not offline else None,
        )
        if offline:
            async with open_store(s) as store:
                retriever = build_retriever(s, store)
                retrieval = await asyncio.to_thread(
                    evaluate_retrieval,
                    retriever,
                    samples,
                    ks=cfg.retrieval_ks,
                    configured_threshold=k_thr,
                )
            routing = evaluate_heuristic_routing(samples)
            checks = check_thresholds({"routing": routing.accuracy}, cfg.thresholds)
            return EvalReport(meta=meta, retrieval=retrieval, routing=routing, thresholds=checks)

        # Each evaluated question is its own throwaway session, so agent checkpoints stay in
        # memory instead of filling the real conversation database.
        # Drafts go to a throwaway store too: evaluating requests must never touch real ones.
        scratch = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        drafts_url = f"sqlite+aiosqlite:///{(Path(scratch.name) / 'drafts.db').as_posix()}"
        opener = (
            open_pipeline(s)
            if engine == "pipeline"
            else open_agent(s, checkpoint_url="memory", drafts_url=drafts_url)
        )
        async with opener as rag:
            retrieval = await asyncio.to_thread(
                evaluate_retrieval,
                rag.retriever,
                samples,
                ks=cfg.retrieval_ks,
                configured_threshold=k_thr,
            )
            judge_model = LLMJudge(get_chat_model(streaming=False, settings=s)) if judge else None
            results = await run_pipeline_eval(
                rag,
                samples,
                run_name=session,
                judge=judge_model,
                tracing=tracing,
                concurrency=concurrency or cfg.concurrency,
            )
        pipeline = summarise_pipeline(results)
        report = EvalReport(
            meta=meta,
            retrieval=retrieval,
            pipeline=pipeline,
            thresholds=check_thresholds(pipeline.metrics, cfg.thresholds),
        )
        # The Phase A baseline only says something about the dataset it was measured on.
        baseline_path = compare or (
            cfg.report_dir / "baseline-full.json"
            if engine == "agent" and path == cfg.dataset
            else None
        )
        if baseline_path is not None and baseline_path.exists():
            saved = json.loads(baseline_path.read_text(encoding="utf-8"))
            report.comparison = compare_reports(pipeline, saved, against=baseline_path.stem)
        elif baseline_path is not None:
            console.print(f"[yellow]No baseline to compare with at {baseline_path}[/yellow]")
        return report

    report = asyncio.run(run())
    md, js = write_report(report, cfg.report_dir, run_name)
    console.print(render_markdown(report))
    console.print(f"[dim]Report written to {md} and {js}[/dim]")
    if fail_under and not report.passed:
        console.print("[red]Below threshold.[/red]")
        raise typer.Exit(1)


# --- serve --------------------------------------------------------------------------


@app.command()
def serve(
    host: Annotated[
        str, typer.Option(help="Bind address (0.0.0.0 inside a container)")
    ] = "127.0.0.1",
    port: Annotated[int, typer.Option(help="Port")] = 8000,
) -> None:
    """Run the HTTP API (JWT auth, SSE streaming). OpenAPI docs are served at /docs."""
    import uvicorn

    from support_agent.api.app import create_app
    from support_agent.api.auth import AuthConfigError, JwtVerifier

    s = _settings()
    try:
        JwtVerifier(s)  # fail now, with a clear message, rather than on the first request
    except AuthConfigError as exc:
        console.print(f"[red]{exc}[/red] Set it in .env (see .env.example).")
        raise typer.Exit(1) from exc
    # No reload and a single process: the MCP tool subprocess and embedded Qdrant are per process.
    uvicorn.run(create_app(s), host=host, port=port, log_config=None)


# --- check --------------------------------------------------------------------------


@app.command()
def check(
    skip_llm: Annotated[bool, typer.Option("--skip-llm", help="Do not call the LLM")] = False,
) -> None:
    """Verify configuration, connectivity and the model's capabilities."""
    from support_agent.llm.capabilities import check_capabilities
    from support_agent.llm.factory import MissingCredentials, get_chat_model
    from support_agent.rag.index import VectorStore, make_client

    s = _settings()
    failed = False
    table = Table("component", "status", "detail")

    def row(name: str, ok: bool | None, detail: str) -> None:
        nonlocal failed
        mark = (
            "[green]ok[/green]"
            if ok
            else ("[yellow]skip[/yellow]" if ok is None else "[red]FAIL[/red]")
        )
        failed = failed or ok is False
        table.add_row(name, mark, detail)

    row(
        "config",
        True,
        f"provider={s.llm_provider} model={s.chat_model_name} embeddings={s.embedding_model_name}",
    )

    try:
        store = VectorStore(make_client(s), s.app.retrieval.collection)
        try:
            n = store.count()
        finally:
            store.close()
        row(
            "qdrant",
            n > 0,
            f"{n} chunks indexed" if n else "collection empty: run `support-agent ingest`",
        )
    except Exception as exc:
        row("qdrant", False, type(exc).__name__)

    try:
        from support_agent.mcp_db.factory import create_adapter
        from support_agent.mcp_db.mapping import load_mapping

        adapter = create_adapter(s, load_mapping(s.app.mapping.path))

        async def db() -> list:
            try:
                return await adapter.validate()
            finally:
                await adapter.close()

        issues = asyncio.run(db())
        errors = [i for i in issues if i.level == "error"]
        row(
            "database",
            not errors,
            "mapping valid" if not errors else f"{len(errors)} mapping error(s)",
        )
    except Exception as exc:
        row("database", False, f"{type(exc).__name__}: {exc}"[:120])

    if skip_llm:
        row("llm", None, "skipped")
    else:
        try:
            report = asyncio.run(check_capabilities(get_chat_model(streaming=False, settings=s)))
            detail = (
                f"{report.status}: tools={report.tool_calling} "
                f"structured={report.structured_output} "
                f"{report.latency_ms}ms"
            )
            row(
                "llm",
                report.status != "failed",
                detail + (f" ({'; '.join(report.warnings)})" if report.warnings else ""),
            )
        except MissingCredentials as exc:
            row("llm", False, str(exc))

    console.print(table)
    if failed:
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
