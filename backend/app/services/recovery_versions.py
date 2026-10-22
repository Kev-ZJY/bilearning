"""Candidate-generation identity for safe recovery decisions.

The fingerprint is built from already-loaded Python code objects and exact
rendered prompt bytes. It deliberately does not read source files from disk:
those files can change while a worker is still running the older imported
implementation.
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
from types import CodeType, FunctionType
from typing import Any, Callable, Mapping


FINGERPRINT_SCHEMA_VERSION = 1
INITIAL_DRAFT_PROMPT_STAGE = "initial_draft"
INITIAL_COVERAGE_PROMPT_PREFIX = "initial_coverage_"


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def loaded_code_sha256(component: Callable[..., Any]) -> str:
    """Hash stable fields of the code object Python actually loaded.

    ``marshal.dumps(code)`` can include interpreter-specialized state that
    changes after a function runs. Serialize the immutable bytecode contract
    explicitly so a checkpoint remains comparable within one loaded process.
    """
    function = getattr(component, "__func__", component)
    code = getattr(function, "__code__", None)
    if not isinstance(code, CodeType):
        raise ValueError("候选生成依赖没有可校验的已加载代码对象")
    payload = _stable_code_payload(code)
    payload["defaults"] = _stable_constant(getattr(function, "__defaults__", None))
    payload["keyword_defaults"] = _stable_constant(
        getattr(function, "__kwdefaults__", None) or {}
    )
    return _sha256(_canonical_json(payload))


def _stable_constant(value: Any) -> Any:
    if isinstance(value, CodeType):
        return {"code": _stable_code_payload(value)}
    if isinstance(value, FunctionType):
        return {
            "function_code": _stable_code_payload(value.__code__),
            "defaults": _stable_constant(value.__defaults__),
            "keyword_defaults": _stable_constant(value.__kwdefaults__ or {}),
        }
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        return {"float_hex": value.hex()}
    if isinstance(value, complex):
        return {"complex_hex": [value.real.hex(), value.imag.hex()]}
    if isinstance(value, bytes):
        return {"bytes_hex": value.hex()}
    if value is Ellipsis:
        return {"ellipsis": True}
    if isinstance(value, tuple):
        return ["tuple", *(_stable_constant(child) for child in value)]
    if isinstance(value, list):
        return ["list", *(_stable_constant(child) for child in value)]
    if isinstance(value, dict):
        pairs = [
            [_stable_constant(key), _stable_constant(child)]
            for key, child in value.items()
        ]
        return {"dict": sorted(pairs, key=lambda pair: _canonical_json(pair[0]))}
    if isinstance(value, (set, frozenset)):
        normalized = [_stable_constant(child) for child in value]
        kind = "frozenset" if isinstance(value, frozenset) else "set"
        return {kind: sorted(normalized, key=_canonical_json)}
    raise TypeError(f"加载代码包含未支持的常量类型: {type(value).__name__}")


def _stable_code_payload(code: CodeType) -> dict[str, Any]:
    return {
        "name": code.co_name,
        "qualname": code.co_qualname,
        "argcount": code.co_argcount,
        "posonlyargcount": code.co_posonlyargcount,
        "kwonlyargcount": code.co_kwonlyargcount,
        "nlocals": code.co_nlocals,
        "stacksize": code.co_stacksize,
        "flags": code.co_flags,
        "bytecode": code.co_code.hex(),
        "exception_table": code.co_exceptiontable.hex(),
        "constants": [_stable_constant(value) for value in code.co_consts],
        "names": list(code.co_names),
        "varnames": list(code.co_varnames),
        "freevars": list(code.co_freevars),
        "cellvars": list(code.co_cellvars),
    }


def constant_sha256(value: str | bytes) -> str:
    """Hash the exact UTF-8 or byte representation of a prompt constant."""
    raw = value.encode("utf-8") if isinstance(value, str) else value
    if not isinstance(raw, bytes):
        raise TypeError("提示常量必须是 str 或 bytes")
    return _sha256(raw)


def _loaded_constant_sha256(value: Any) -> str:
    """Hash a scoped prompt/validation constant from its loaded value."""
    if isinstance(value, re.Pattern):
        payload = {"regex_pattern": value.pattern, "regex_flags": value.flags}
    else:
        payload = _stable_constant(value)
    return _sha256(_canonical_json(payload))


def _loaded_module_callables(module: Any, prefix: str) -> dict[str, Callable[..., Any]]:
    """Collect locally implemented functions, including methods on local classes."""
    result: dict[str, Callable[..., Any]] = {}
    for name, value in vars(module).items():
        if getattr(value, "__module__", None) == module.__name__ and hasattr(value, "__code__"):
            result[f"{prefix}.{name}"] = value
        if isinstance(value, type) and value.__module__ == module.__name__:
            for method_name, method in vars(value).items():
                if isinstance(method, (staticmethod, classmethod)):
                    method = method.__func__
                if getattr(method, "__module__", None) == module.__name__ and hasattr(method, "__code__"):
                    result[f"{prefix}.{name}.{method_name}"] = method
    return result


def rendered_prompt_identity(stage: str, rendered_prompt: str) -> dict[str, Any]:
    """Return a digest and byte length for the exact prompt material sent."""
    if not isinstance(stage, str) or not stage.strip():
        raise ValueError("候选生成提示阶段名称不能为空")
    if not isinstance(rendered_prompt, str):
        raise TypeError("候选生成提示必须是 str")
    raw = rendered_prompt.encode("utf-8")
    return {"stage": stage, "sha256": _sha256(raw), "size_bytes": len(raw)}


def rendered_prompt_identity_from_digest(
    stage: str, sha256: str, size_bytes: int,
) -> dict[str, Any]:
    """Accept a prompt digest emitted by the exact call-site renderer."""
    identity = {"stage": stage, "sha256": sha256, "size_bytes": size_bytes}
    if not _valid_prompt_identity(identity):
        raise ValueError("候选生成提示摘要格式无效")
    return identity


def _valid_digest(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        char in "0123456789abcdef" for char in value
    )


def _valid_prompt_identity(value: Any) -> bool:
    return (
        isinstance(value, dict)
        and isinstance(value.get("stage"), str)
        and bool(value["stage"].strip())
        and _valid_digest(value.get("sha256"))
        and type(value.get("size_bytes")) is int
        and value["size_bytes"] >= 0
    )


def default_candidate_generation_code_components() -> dict[str, str]:
    """Digest the loaded implementation and prompt contracts used for drafts.

    This is a deliberately scoped dependency set: initial drafting, strict
    document checks, and the first complete-coverage generation. It does not
    claim to fingerprint model weights, provider behavior, or every process
    dependency.
    """
    from app.agent.knowledge import document_contract, draft_schema, grounding_schema, extractor, grounding
    from app.agent.review import independent_review
    from app.agent.workflow import pipeline
    from app.agent.tooling import web_research
    from app.agent.review import blind_review
    from app.agent.media import illustrations
    from app.agent.runtime import model_limits, public_progress, tool_context
    from app.agent.knowledge import grounding_scope
    from app.agent.tooling import tool_definitions
    from app.agent.runtime import llm as llm_module, http_transport
    from app.agent.runtime import context as agent_context
    from app.agent.runtime.context import (
        CONTEXT_BUDGET_TOKENS,
        _messages_tokens as app_context_messages_tokens,
        document_for_model,
        estimate_tokens,
        estimate_llm_input_tokens,
        validate_messages_budget,
    )
    from app.agent.runtime.heartbeat import chat_with_heartbeat
    from app.agent.runtime.llm import LLMClient, extract_json
    from app.agent.runtime.task_budget import ModelAttemptBudgetExceeded, TaskBudget

    callables: dict[str, Callable[..., Any]] = {
        "pipeline.Pipeline.run": pipeline.Pipeline.run,
        "pipeline.Pipeline._run": pipeline.Pipeline._run,
        "pipeline.Pipeline.resume": pipeline.Pipeline.resume,
        "pipeline.Pipeline._chat_with_heartbeat": pipeline.Pipeline._chat_with_heartbeat,
        "pipeline.Pipeline._grounding_call": pipeline.Pipeline._grounding_call,
        "pipeline.Pipeline._grounding_callback": pipeline.Pipeline._grounding_callback,
        "pipeline.Pipeline._draft_activity_callbacks": pipeline.Pipeline._draft_activity_callbacks,
        "pipeline.Pipeline._model_input_budget": pipeline.Pipeline._model_input_budget,
        "pipeline.Pipeline._input_budget_kwargs": pipeline.Pipeline._input_budget_kwargs,
        "pipeline.Pipeline._publish_public_progress_segment": pipeline.Pipeline._publish_public_progress_segment,
        "pipeline.Pipeline._load_existing_doc": pipeline.Pipeline._load_existing_doc,
        "pipeline.Pipeline._init_task_budget": pipeline.Pipeline._init_task_budget,
        "pipeline.Pipeline._checkpoint": pipeline.Pipeline._checkpoint,
        "pipeline.dumps": pipeline.dumps,
        "pipeline._json_default": pipeline._json_default,
        "pipeline.build_extraction_prompt": pipeline.build_extraction_prompt,
        "context.document_for_model": document_for_model,
        "context.estimate_tokens": estimate_tokens,
        "context.estimate_llm_input_tokens": estimate_llm_input_tokens,
        "context._messages_tokens": app_context_messages_tokens,
        "context.validate_messages_budget": validate_messages_budget,
        "llm.extract_json": extract_json,
        "llm.LLMClient._post": LLMClient._post,
        "llm.LLMClient._tool_protocol_options": LLMClient._tool_protocol_options,
        "llm.LLMClient._model_request_compatibility_options": LLMClient._model_request_compatibility_options,
        "llm.LLMClient.input_budget_tokens": LLMClient.input_budget_tokens,
        "llm.LLMClient._effective_output_tokens": LLMClient._effective_output_tokens,
        "llm.LLMClient._try_once_tools": LLMClient._try_once_tools,
        "llm.LLMClient._run": LLMClient._run,
        "llm.LLMClient._try_once": LLMClient._try_once,
        "llm.LLMClient.chat_tools": LLMClient.chat_tools,
        "llm.LLMClient.chat": LLMClient.chat,
        "llm.LLMClient.__init__": LLMClient.__init__,
        "task_budget.TaskBudget.__init__": TaskBudget.__init__,
        "task_budget.TaskBudget._snapshot_locked": TaskBudget._snapshot_locked,
        "task_budget.TaskBudget.snapshot": TaskBudget.snapshot,
        "task_budget.TaskBudget.reserve_attempt": TaskBudget.reserve_attempt,
        "task_budget.ModelAttemptBudgetExceeded.__init__": ModelAttemptBudgetExceeded.__init__,
        "heartbeat.chat_with_heartbeat": chat_with_heartbeat,
        "grounding.complete_document": grounding.complete_document,
        "independent_review._citation_issues": independent_review._citation_issues,
        "independent_review.document_hash": independent_review.document_hash,
        "independent_review._points": independent_review._points,
        "independent_review._prose_strings": independent_review._prose_strings,
        "independent_review._image": independent_review._image,
        "independent_review._bounded_figure": independent_review._bounded_figure,
    }
    callables.update(_loaded_module_callables(document_contract, "document_contract"))
    callables.update(_loaded_module_callables(grounding_schema, "grounding_schema"))
    callables.update(_loaded_module_callables(extractor, "extractor"))
    callables.update(_loaded_module_callables(grounding, "grounding"))
    callables.update(_loaded_module_callables(grounding_scope, "grounding_scope"))
    callables.update(_loaded_module_callables(tool_definitions, "tool_definitions"))
    callables.update(_loaded_module_callables(web_research, "web_research"))
    callables.update(_loaded_module_callables(model_limits, "model_limits"))
    callables.update(_loaded_module_callables(public_progress, "public_progress"))
    callables.update(_loaded_module_callables(tool_context, "tool_context"))
    callables.update(_loaded_module_callables(blind_review, "blind_review"))
    callables.update(_loaded_module_callables(illustrations, "illustrations"))
    callables.update(_loaded_module_callables(llm_module, "llm"))
    callables.update(_loaded_module_callables(http_transport, "http_transport"))
    callables.update(_loaded_module_callables(agent_context, "context"))
    # Mandatory entry points remain visible when instrumentation rebinds a
    # module attribute to a loaded implementation from a different module.
    callables.update({
        "model_limits.resolve_model_limits": model_limits.resolve_model_limits,
        "model_limits.input_budget_for_llm": model_limits.input_budget_for_llm,
        "public_progress.PublicProgressReporter.report": public_progress.PublicProgressReporter.report,
        "tool_context.compact_tool_context": tool_context.compact_tool_context,
        "blind_review.build_projection": blind_review.build_projection,
        "illustrations.enrich_illustrations": illustrations.enrich_illustrations,
        "http_transport.post": http_transport.post,
        "http_transport.request_controls.generator": http_transport.request_controls.__wrapped__,
        "http_transport._BufferedResponse.text": http_transport._BufferedResponse.text.fget,
        "http_transport._StreamingResponse.text": http_transport._StreamingResponse.text.fget,
        "http_transport._StreamingResponse.content": http_transport._StreamingResponse.content.fget,
        "web_research._google_dns_public_a": web_research._google_dns_public_a,
        "web_research._parse_google_dns_a_response": web_research._parse_google_dns_a_response,
    })
    schema_builder = getattr(draft_schema, "build_draft_response_format", None)
    if not callable(schema_builder) or not isinstance(
        getattr(schema_builder, "__code__", None), CodeType,
    ):
        raise ValueError("初稿结构化输出 schema builder 没有可校验的已加载代码对象")
    schema_policy = getattr(draft_schema, "DRAFT_SCHEMA_POLICY", None)
    if not isinstance(schema_policy, str) or not schema_policy.strip():
        raise ValueError("初稿结构化输出 schema policy 缺失或无效")
    callables["draft_schema.build_draft_response_format"] = schema_builder
    for name, component in vars(draft_schema).items():
        if (getattr(component, "__module__", None) == draft_schema.__name__
                and hasattr(component, "__code__")):
            callables[f"draft_schema.{name}"] = component
    grounding_schema_builder = getattr(
        grounding_schema, "build_grounding_response_format", None,
    )
    if not callable(grounding_schema_builder) or not isinstance(
        getattr(grounding_schema_builder, "__code__", None), CodeType,
    ):
        raise ValueError("grounding schema builder 没有可校验的已加载代码对象")
    grounding_schema_policy = getattr(
        grounding_schema, "GROUNDING_SCHEMA_POLICY", None,
    )
    if not isinstance(grounding_schema_policy, str) or not grounding_schema_policy.strip():
        raise ValueError("grounding schema policy 缺失或无效")
    grounding_reference_policy = getattr(
        grounding_schema, "GROUNDING_EVIDENCE_REFERENCE_POLICY", None,
    )
    if not isinstance(grounding_reference_policy, str) or not grounding_reference_policy.strip():
        raise ValueError("grounding evidence reference policy 缺失或无效")
    grounding_reference_limit = getattr(
        grounding_schema, "MAX_GROUNDING_EVIDENCE_IDS_PER_ITEM", None,
    )
    if type(grounding_reference_limit) is not int or grounding_reference_limit < 1:
        raise ValueError("grounding evidence reference 上限缺失或无效")
    callables["grounding_schema.build_grounding_response_format"] = grounding_schema_builder
    constants: dict[str, Any] = {
        "runtime.python": (
            f"{sys.implementation.name}:{sys.version_info.major}."
            f"{sys.version_info.minor}.{sys.version_info.micro}"
        ),
        "context.CONTEXT_BUDGET_TOKENS": str(CONTEXT_BUDGET_TOKENS),
        "model_limits.INPUT_SAFETY_MARGIN_TOKENS": str(model_limits.INPUT_SAFETY_MARGIN_TOKENS),
        "model_limits.DEFAULT_INPUT_BUDGET_TOKENS": str(model_limits.DEFAULT_INPUT_BUDGET_TOKENS),
        "model_limits._DEEPSEEK_BASE_URLS": model_limits._DEEPSEEK_BASE_URLS,
        "model_limits._OPENROUTER_BASE_URL": model_limits._OPENROUTER_BASE_URL,
        "public_progress.REPORT_PROGRESS_TOOL": public_progress.REPORT_PROGRESS_TOOL,
        "public_progress.MAX_PROGRESS_SUMMARY_CHARS": str(public_progress.MAX_PROGRESS_SUMMARY_CHARS),
        "public_progress.MAX_PROGRESS_CALLS_PER_ROUND": str(public_progress.MAX_PROGRESS_CALLS_PER_ROUND),
        "public_progress.MAX_PROGRESS_CALLS_PER_LOOP": str(public_progress.MAX_PROGRESS_CALLS_PER_LOOP),
        "public_progress._ALLOWED_KINDS": public_progress._ALLOWED_KINDS,
        "public_progress._ALLOWED_ACTORS": public_progress._ALLOWED_ACTORS,
        "public_progress._ALLOWED_EXECUTIONS": public_progress._ALLOWED_EXECUTIONS,
        "document_contract.WRITING_RULES": document_contract.WRITING_RULES,
        "document_contract.SOURCE_AUDIT_NARRATIVE": document_contract.SOURCE_AUDIT_NARRATIVE,
        "document_contract.META_NARRATIVE": document_contract.META_NARRATIVE,
        "document_contract.MAX_FIGURE_BYTES": str(document_contract.MAX_FIGURE_BYTES),
        "extractor._CONJUNCT_RE": extractor._CONJUNCT_RE,
        "extractor._HE_INSIDE_WORDS": extractor._HE_INSIDE_WORDS,
        "extractor._HARD_BAD_RE": extractor._HARD_BAD_RE,
        "extractor.POINT_TITLE_BAD": extractor.POINT_TITLE_BAD,
        "independent_review.MAX_IMAGE_BYTES": str(independent_review.MAX_IMAGE_BYTES),
        "grounding.GROUNDING_MIN_OUTPUT_RESERVE_TOKENS": str(grounding.GROUNDING_MIN_OUTPUT_RESERVE_TOKENS),
        "grounding.GROUNDING_OUTPUT_MULTIPLIER": str(grounding.GROUNDING_OUTPUT_MULTIPLIER),
        "grounding.GROUNDING_RETRY_OVERHEAD_TOKENS": str(grounding.GROUNDING_RETRY_OVERHEAD_TOKENS),
        "grounding.GROUNDING_MIN_SAFETY_MARGIN_TOKENS": str(grounding.GROUNDING_MIN_SAFETY_MARGIN_TOKENS),
        "grounding.GROUNDING_SAFETY_MARGIN_RATIO": str(grounding.GROUNDING_SAFETY_MARGIN_RATIO),
        "grounding.CONTEXT_BUDGET_TOKENS": str(grounding.CONTEXT_BUDGET_TOKENS),
        "grounding._GROUNDING_AUDIT_FIELDS": grounding._GROUNDING_AUDIT_FIELDS,
        "grounding._FIGURE_TEXT_FIELDS": grounding._FIGURE_TEXT_FIELDS,
        "web_research.MAX_BYTES": str(web_research.MAX_BYTES),
        "web_research.MAX_TEXT": str(web_research.MAX_TEXT),
        "web_research.TIMEOUT": str(web_research.TIMEOUT),
        "web_research.MAX_REDIRECTS": str(web_research.MAX_REDIRECTS),
        "web_research.DNS_JSON_MAX_BYTES": str(web_research.DNS_JSON_MAX_BYTES),
        "web_research.PUBLIC_DNS_JSON_ENDPOINT": web_research.PUBLIC_DNS_JSON_ENDPOINT,
        "web_research._ProxyConnectionFailure": {
            "module": web_research._ProxyConnectionFailure.__module__,
            "qualname": web_research._ProxyConnectionFailure.__qualname__,
            "mro": [
                f"{base.__module__}.{base.__qualname__}"
                for base in web_research._ProxyConnectionFailure.__mro__
            ],
        },
        "independent_review.TOOLS": json.dumps(
            independent_review.TOOLS, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ),
    }

    # These loaded policy constants affect which evidence history the initial
    # author receives after compaction; a changed policy invalidates old drafts.
    for module, prefix in (
        (tool_context, "tool_context"),
        (blind_review, "blind_review"),
        (illustrations, "illustrations"),
        (http_transport, "http_transport"),
        (draft_schema, "draft_schema"),
        (grounding_schema, "grounding_schema"),
    ):
        for name, value in vars(module).items():
            if name.isupper() and isinstance(value, (str, int, float, tuple, list, dict, set, frozenset, re.Pattern)):
                constants[f"{prefix}.{name}"] = value

    # These two constraints govern the bounded reference output contract and
    # must remain explicit fingerprint inputs even if policy collection changes.
    constants["grounding_schema.GROUNDING_EVIDENCE_REFERENCE_POLICY"] = grounding_reference_policy
    constants["grounding_schema.MAX_GROUNDING_EVIDENCE_IDS_PER_ITEM"] = str(
        grounding_reference_limit
    )

    # Task1's initial writer is an explicit generator dependency once installed.
    try:
        from app.agent.knowledge import draft_writer
    except ImportError:
        draft_writer = None
    if draft_writer is not None:
        # Fingerprint every local function in the bounded writer module; its
        # helpers enforce the evidence tools and citation/frame constraints.
        for name, component in vars(draft_writer).items():
            if getattr(component, "__module__", None) == draft_writer.__name__ and hasattr(component, "__code__"):
                callables[f"draft_writer.{name}"] = component
        # This entry is mandatory even in instrumented runtimes where the
        # module attribute was rebound to another loaded Python implementation.
        # Never silently omit the actual generator from a recovery comparison.
        generator = getattr(draft_writer, "generate_draft", None)
        if not callable(generator) or not isinstance(getattr(generator, "__code__", None), CodeType):
            raise ValueError("初稿生成入口没有可校验的已加载代码对象")
        callables["draft_writer.generate_draft"] = generator
        instructions = getattr(draft_writer, "DRAFT_WRITER_INSTRUCTIONS", None)
        if isinstance(instructions, str):
            constants["draft_writer.DRAFT_WRITER_INSTRUCTIONS"] = instructions
        for name in (
            "MAX_DRAFT_ROUNDS", "MAX_DRAFT_TOOL_CALLS",
            "MAX_DOCUMENT_FORMAT_REPAIRS", "MAX_TOOL_BATCH",
            "IMAGE_CONTEXT_RESERVE_TOKENS", "MAX_IMAGE_BYTES",
            "MAX_SCALAR_PROSE_PATCH_PATHS", "MAX_SCALAR_PROSE_PATCH_TEXT_CHARS",
        ):
            value = getattr(draft_writer, name, None)
            if type(value) is int:
                constants[f"draft_writer.{name}"] = str(value)
        summary_instructions = getattr(draft_writer, "CONTEXT_SUMMARY_SYSTEM", None)
        if isinstance(summary_instructions, str):
            constants["draft_writer.CONTEXT_SUMMARY_SYSTEM"] = summary_instructions
        prose_patch_instructions = getattr(draft_writer, "SCALAR_PROSE_PATCH_SYSTEM", None)
        if isinstance(prose_patch_instructions, str):
            constants["draft_writer.SCALAR_PROSE_PATCH_SYSTEM"] = prose_patch_instructions
        tool_argument_keys = getattr(draft_writer, "_TOOL_ARGUMENT_KEYS", None)
        if isinstance(tool_argument_keys, dict):
            constants["draft_writer._TOOL_ARGUMENT_KEYS"] = json.dumps(
                {
                    name: sorted(keys)
                    for name, keys in sorted(tool_argument_keys.items())
                }, ensure_ascii=False, sort_keys=True,
                separators=(",", ":"),
            )

    result = {name: loaded_code_sha256(value) for name, value in callables.items()}
    result.update({name: _loaded_constant_sha256(value) for name, value in constants.items()})
    return dict(sorted(result.items()))


def candidate_generation_intent(
    *,
    prompt_identities: list[dict[str, Any]],
    code_dependencies: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Capture loaded code and prompt digests before the candidate call."""
    if not isinstance(prompt_identities, list) or not prompt_identities:
        raise ValueError("候选生成至少需要一个真实初稿提示摘要")
    if any(not _valid_prompt_identity(row) for row in prompt_identities):
        raise ValueError("候选生成提示摘要格式无效")
    if prompt_identities[0].get("stage") != INITIAL_DRAFT_PROMPT_STAGE:
        raise ValueError("候选生成提示摘要必须以 initial_draft 开始")
    names = [row["stage"] for row in prompt_identities]
    if len(names) != len(set(names)):
        raise ValueError("候选生成提示阶段不能重复")
    components = dict(code_dependencies or default_candidate_generation_code_components())
    if not components or any(
        not isinstance(name, str) or not name.strip() or not _valid_digest(digest)
        for name, digest in components.items()
    ):
        raise ValueError("候选生成代码依赖摘要格式无效")
    return {
        "schema_version": FINGERPRINT_SCHEMA_VERSION,
        "code_components": dict(sorted(components.items())),
        "prompt_identities": [dict(row) for row in prompt_identities],
    }


def capture_candidate_generation_fingerprint(
    rendered_prompt: str,
    *,
    code_dependencies: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Capture the exact initial writer prompt before its first model call."""
    return candidate_generation_intent(
        prompt_identities=[rendered_prompt_identity(INITIAL_DRAFT_PROMPT_STAGE, rendered_prompt)],
        code_dependencies=code_dependencies,
    )


def capture_initial_draft_intent(
    user_prompt: str,
    *,
    raw_sources: list[tuple[str, str]] | None = None,
    code_dependencies: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Capture Task1's canonical actual system+user prompt before generation."""
    from app.agent.knowledge import draft_writer

    rendered = draft_writer.render_initial_draft_prompt(
        user_prompt, raw_sources=raw_sources,
    )
    identity = rendered_prompt_identity(INITIAL_DRAFT_PROMPT_STAGE, rendered)
    helper_identity = draft_writer.rendered_prompt_identity(
        user_prompt, raw_sources=raw_sources,
    )
    if identity["sha256"] != helper_identity["sha256"] or identity["size_bytes"] != helper_identity["utf8_bytes"]:
        raise ValueError("初稿 helper 的实际提示与恢复指纹渲染结果不一致")
    return candidate_generation_intent(
        prompt_identities=[identity], code_dependencies=code_dependencies,
    )


def add_generation_prompt_identity(
    intent: dict[str, Any], stage: str, rendered_prompt: str,
) -> dict[str, Any]:
    """Record an exact later prompt, such as a complete-coverage call."""
    if not is_candidate_generation_intent(intent):
        raise ValueError("候选生成意图摘要无效")
    result = json.loads(_canonical_json(intent))
    identity = rendered_prompt_identity(stage, rendered_prompt)
    if any(row["stage"] == stage for row in result["prompt_identities"]):
        raise ValueError("候选生成提示阶段不能重复")
    result["prompt_identities"].append(identity)
    return result


def is_candidate_generation_intent(value: Any) -> bool:
    return (
        isinstance(value, dict)
        and set(value) == {"schema_version", "code_components", "prompt_identities"}
        and value.get("schema_version") == FINGERPRINT_SCHEMA_VERSION
        and isinstance(value.get("code_components"), dict)
        and bool(value["code_components"])
        and all(
            isinstance(name, str) and bool(name.strip()) and _valid_digest(digest)
            for name, digest in value["code_components"].items()
        )
        and isinstance(value.get("prompt_identities"), list)
        and bool(value["prompt_identities"])
        and all(_valid_prompt_identity(row) for row in value["prompt_identities"])
        and value["prompt_identities"][0].get("stage") == INITIAL_DRAFT_PROMPT_STAGE
        and len({row["stage"] for row in value["prompt_identities"]})
        == len(value["prompt_identities"])
    )


def has_actual_initial_coverage_prompt(value: Any) -> bool:
    return is_candidate_generation_intent(value) and any(
        row["stage"].startswith(INITIAL_COVERAGE_PROMPT_PREFIX)
        for row in value["prompt_identities"]
    )


def bind_candidate_artifact(intent: dict[str, Any], artifact: bytes) -> dict[str, Any]:
    """Bind the post-coverage candidate bytes to a pre-captured generation intent."""
    if not has_actual_initial_coverage_prompt(intent):
        raise ValueError("候选生成意图摘要无效")
    if not isinstance(artifact, bytes):
        raise TypeError("候选文档工件必须是 bytes")
    value = {
        **json.loads(_canonical_json(intent)),
        "candidate_artifact": {"sha256": _sha256(artifact), "size_bytes": len(artifact)},
    }
    value["fingerprint_sha256"] = _sha256(_canonical_json(value))
    return value


def validate_candidate_generation_fingerprint(
    value: Any, *, candidate_artifact: bytes | None = None,
) -> bool:
    if not isinstance(value, dict) or set(value) != {
        "schema_version", "code_components", "prompt_identities",
        "candidate_artifact", "fingerprint_sha256",
    }:
        return False
    artifact = value.get("candidate_artifact")
    if (
        value.get("schema_version") != FINGERPRINT_SCHEMA_VERSION
        or not has_actual_initial_coverage_prompt({
            key: value.get(key) for key in ("schema_version", "code_components", "prompt_identities")
        })
        or not isinstance(artifact, dict)
        or set(artifact) != {"sha256", "size_bytes"}
        or not _valid_digest(artifact.get("sha256"))
        or type(artifact.get("size_bytes")) is not int
        or artifact["size_bytes"] < 0
        or not _valid_digest(value.get("fingerprint_sha256"))
    ):
        return False
    payload = {key: child for key, child in value.items() if key != "fingerprint_sha256"}
    if _sha256(_canonical_json(payload)) != value["fingerprint_sha256"]:
        return False
    if candidate_artifact is not None:
        return artifact == {"sha256": _sha256(candidate_artifact), "size_bytes": len(candidate_artifact)}
    return True


def compare_candidate_generation(
    source: Any,
    target_intent: Any,
    *,
    candidate_artifact: bytes | None = None,
) -> tuple[str, str]:
    """Classify candidate reuse; unknown or different dependencies require regen."""
    if source is None:
        return "unknown", "候选生成指纹缺失，旧候选不可复用"
    if not validate_candidate_generation_fingerprint(source, candidate_artifact=candidate_artifact):
        return "invalid", "候选生成指纹或候选工件哈希无效，旧候选不可复用"
    if not is_candidate_generation_intent(target_intent):
        return "unknown", "当前候选生成实现/提示摘要不可用，旧候选不可复用"
    if source["code_components"] != target_intent["code_components"]:
        return "mismatch", "候选生成代码依赖与当前已加载实现不匹配，需重新生成初稿"
    source_prompt = next(
        (row for row in source["prompt_identities"] if row.get("stage") == INITIAL_DRAFT_PROMPT_STAGE),
        None,
    )
    target_prompt = next(
        (row for row in target_intent["prompt_identities"] if row.get("stage") == INITIAL_DRAFT_PROMPT_STAGE),
        None,
    )
    if source_prompt is None or target_prompt is None:
        return "unknown", "初稿提示摘要缺失，旧候选不可复用"
    if source_prompt != target_prompt:
        return "mismatch", "初稿提示与当前输入重构结果不匹配，需重新生成初稿"
    if not any(
        isinstance(row, dict)
        and isinstance(row.get("stage"), str)
        and row["stage"].startswith(INITIAL_COVERAGE_PROMPT_PREFIX)
        for row in source["prompt_identities"]
    ):
        return "unknown", "来源候选缺少实际初始覆盖提示摘要，旧候选不可复用"
    return "matched", "候选生成代码依赖及精确初稿提示摘要匹配"


def candidate_generation_prefix_matches(source_intent: Any, target_intent: Any) -> bool:
    """Check code and initial prompt equality while regeneration is pending."""
    if not is_candidate_generation_intent(source_intent) or not is_candidate_generation_intent(target_intent):
        return False
    return (
        source_intent["code_components"] == target_intent["code_components"]
        and source_intent["prompt_identities"][0] == target_intent["prompt_identities"][0]
    )


def initial_prompt_identity_matches(intent: Any, generation_report: Any) -> bool:
    """Require Task1 to confirm the canonical prompt it actually rendered."""
    if not is_candidate_generation_intent(intent) or not isinstance(generation_report, dict):
        return False
    expected = intent["prompt_identities"][0]
    return (
        generation_report.get("actual_prompt_sha256") == expected["sha256"]
        and type(generation_report.get("actual_prompt_utf8_bytes")) is int
        and generation_report["actual_prompt_utf8_bytes"] == expected["size_bytes"]
    )


def current_candidate_generation_intent(snapshot: Any) -> dict[str, Any] | None:
    """Rebuild the current implementation and exact initial prompt.

    Later complete-coverage call prompts are stored verbatim-by-digest in the
    source fingerprint as observed at call time. Their sequence is model-output
    dependent, so compatibility checks the loaded coverage implementation code
    plus the exact initial prompt and does not invent a replayed target sequence.
    """
    try:
        import app.agent.workflow.pipeline as pipeline
        from app.services.options import ProcessingOptions

        transcript = "\n\n".join(f"【{name}】\n{text}" for name, text in snapshot.sources)
        prompt_options = ProcessingOptions(**dict(snapshot.options)).model_dump()
        prompt_options.pop("max_model_attempts", None)
        historical_context = prompt_options.pop("historical_context", None)
        prompt_kwargs = {"options": prompt_options}
        if historical_context is not None:
            prompt_kwargs["historical_context"] = historical_context
        user_prompt = pipeline.build_extraction_prompt(
            transcript,
            snapshot.terms,
            existing_summary=pipeline.dumps(pipeline.document_for_model(snapshot.baseline_doc)),
            intent=snapshot.intent,
            **prompt_kwargs,
        )
        return capture_initial_draft_intent(
            user_prompt, raw_sources=getattr(snapshot, "sources", None),
        )
    except Exception:
        # Callers treat an unavailable target fingerprint as unknown and fail closed.
        return None
