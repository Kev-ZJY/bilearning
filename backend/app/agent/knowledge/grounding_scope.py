"""Pure, conservative checks for scope strengthening in source-supported prose.

This heuristic detects selected quantifier/condition changes. It is not a
general entailment solver and does not access an LLM or original media.
"""
import re


def _grounding_visible_text_fields(item_id, value):
    """Yield only independently visible prose fields and their display paths."""
    visible_keys = {
        "alt", "anchor", "body", "caption", "content", "description", "example",
        "examples", "explanation", "headers", "items", "label", "language", "name",
        "nodes", "result", "rows", "steps", "text", "title", "value", "warnings",
    }

    def leaves(current, path):
        if isinstance(current, str):
            yield path, current
        elif isinstance(current, list):
            for index, child in enumerate(current):
                yield from leaves(child, f"{path}[{index}]")
        elif isinstance(current, dict):
            for key, child in current.items():
                if key in visible_keys:
                    yield from leaves(child, f"{path}.{key}")

    fields = []
    if item_id == "appendix":
        for field in ("title", "headers", "rows"):
            if field in value:
                fields.extend(leaves(value[field], f"appendix.{field}"))
        tables = value.get("cheatsheets")
        if isinstance(tables, list):
            for index, table in enumerate(tables):
                if not isinstance(table, dict):
                    continue
                prefix = f"appendix.cheatsheets[{index}]"
                for field in ("title", "headers", "rows"):
                    if field in table:
                        fields.extend(leaves(table[field], f"{prefix}.{field}"))
        return fields

    for field in ("title", "anchor", "body", "examples", "warnings"):
        if field in value:
            fields.extend(leaves(value[field], f"{item_id}.{field}"))
    figures = value.get("figures")
    if isinstance(figures, list):
        for index, figure in enumerate(figures):
            if not isinstance(figure, dict):
                continue
            for field in ("alt", "caption"):
                if isinstance(figure.get(field), str):
                    fields.append((f"{item_id}.figures[{index}].{field}", figure[field]))
    return fields

def _grounding_scope_core(text):
    markers = (
        "大多数", "多数", "大部分", "部分", "有些", "一些", "通常", "往往", "一般",
        "经常", "常常", "所有", "全部", "每个", "每一", "每条", "每项", "每种",
        "每位", "每次", "各个", "逐一", "逐项", "全都", "全体", "任一", "一切",
        "一一对应", "无一例外", "总是", "始终", "必然", "可能", "如果", "假如",
        "若", "当", "一旦", "则", "那么", "只有", "唯有", "仅当", "仅在", "仅有",
        "才能", "才可以", "才会", "才达到", "才谈", "才需要", "方可", "方能",
        "必要条件", "必须", "必需", "一定", "都", "均", "皆", "可以", "能够",
        "进行", "是", "有", "的", "most", "usually", "often", "generally",
        "typically", "some", "all", "every", "each", "always", "one-to-one",
        "if", "then", "only", "when", "necessary", "must", "can", "may", "the",
    )
    pattern = "|".join(re.escape(marker) for marker in sorted(markers, key=len, reverse=True))
    core = re.sub(pattern, "", text, flags=re.IGNORECASE)
    return re.sub(r"[^\w\u3400-\u9fff]+", "", core).casefold()

def _grounding_scope_related(left, right):
    """Require a shared topic phrase before comparing scope markers."""
    left_core = _grounding_scope_core(left)
    right_core = _grounding_scope_core(right)
    left_han = re.findall(r"[\u3400-\u9fff]+", left_core)
    right_han = re.findall(r"[\u3400-\u9fff]+", right_core)
    left_bigrams = {
        run[index:index + 2]
        for run in left_han
        for index in range(len(run) - 1)
    }
    right_bigrams = {
        run[index:index + 2]
        for run in right_han
        for index in range(len(run) - 1)
    }
    if left_bigrams & right_bigrams:
        return True
    left_tokens = set(re.findall(r"[a-z][a-z0-9_-]*|\d+", left_core))
    right_tokens = set(re.findall(r"[a-z][a-z0-9_-]*|\d+", right_core))
    common_tokens = left_tokens & right_tokens
    return len(common_tokens) >= 2 or any(len(token) >= 4 for token in common_tokens)

def _grounding_scope_clauses(text):
    return [clause.strip() for clause in re.split(r"[，,。；;！？?!\n]+", text) if clause.strip()]

def _explicit_quantified_heads(text, *, qualified):
    """Extract short explicit Chinese noun heads; ambiguity keeps the old gate."""
    marker = (r"(?:大多数|多数|大部分|部分)(?:的)?" if qualified else
              r"(?:每(?:一)?(?:个|条|项|种|位|次)?|所有|全部)(?:的)?")
    heads = set()
    for match in re.finditer(marker, text):
        if re.search(r"(?:并非|不是|未必|不一定|无需|不需要|不必|不)$",
                     text[max(0, match.start()-8):match.start()].strip()):
            continue
        noun = re.match(r"[\u3400-\u9fffA-Za-z0-9_]+", text[match.end():].lstrip())
        if not noun:
            continue
        head = re.split(r"单独|进行|拥有|包含|包括|支持|适用|要求|对应|如何|记录|必须|都|均|会|可|能|需|应|在|有|是|为",
                        noun.group(0), maxsplit=1)[0]
        if 1 <= len(head) <= 8:
            heads.add(head)
    return heads


def _different_explicit_quantified_subject(candidate_clause, source_clause, source_contexts):
    """Exempt only a literal supported proposition about a different noun.

    A noun mentioned universally elsewhere cannot borrow a majority-scoped
    predicate. Ambiguous paraphrases still go through the correction loop.
    """
    candidate_heads = _explicit_quantified_heads(candidate_clause, qualified=False)
    limited_heads = _explicit_quantified_heads(source_clause, qualified=True)
    if not candidate_heads or not limited_heads or candidate_heads & limited_heads:
        return False

    def literal(text):
        # Normalize only equivalent classifier spelling and layout, never
        # remove modality, negation or predicates to manufacture support.
        text = re.sub(r"每一(?=[个条项种位次])", "每", text)
        return re.sub(r"[\s，,。；;！？?!]+", "", text)

    candidate = literal(candidate_clause)

    def complete_assertion(clause):
        source = literal(clause)
        if not source.endswith(candidate):
            return False
        prefix = source[:-len(candidate)]
        # Only omit the speaker and plain future auxiliary. Never slice a
        # modal/negative prefix or a trailing second proposition off a match.
        return re.fullmatch(r"(?:我们|我)?(?:会)?", prefix) is not None

    return any(
        not _grounding_has_qualified_scope(clause)
        and _grounding_has_unqualified_universal(clause)
        and candidate_heads <= _explicit_quantified_heads(clause, qualified=False)
        and complete_assertion(clause)
        for context in source_contexts
        for clause in _grounding_scope_clauses(context)
    )

def _grounding_has_qualified_scope(text):
    return bool(re.search(
        r"大多数|多数|大部分|部分|有些|一些|通常|往往|一般|经常|常常|"
        r"\b(?:most|usually|often|generally|typically|some|many)\b",
        text,
        re.IGNORECASE,
    ))

def _grounding_has_unqualified_universal(text):
    if _grounding_has_qualified_scope(text):
        return False
    universal = re.compile(
        r"大多数|多数|大部分|所有|全部|每个|每一|每条|每项|每种|每位|每次|各个|逐一|逐项|全都|全体|任一|一切|一一对应|无一例外|总是|始终|必然|"
        r"\ball\b|\bevery\b|\beach\b|\balways\b|\bone-to-one\b|"
        r"均(?=可|能|为|有|是|包含|包括|支持|适用|应|会|需|必须|更新|运行|执行|迁移|匹配)|皆(?=可|能|为|有|是|包含|包括|支持|适用|应|会|需|必须)",
        re.IGNORECASE,
    )
    negated = re.compile(r"(?:并非|不是|并不|不一定|未必|无需|不需要|不必|没有|非|不)$")
    return any(
        not negated.search(text[max(0, match.start() - 8):match.start()].strip())
        for match in universal.finditer(text)
    )

def _grounding_has_if_then(text):
    return bool(
        re.search(r"(?:如果|若|假如|倘若|当|一旦)[\s\S]{1,48}?(?:则|就|那么)[\s\S]{1,72}", text)
        or re.search(r"\bif\b[\s\S]{1,80}?\bthen\b[\s\S]{1,96}", text, re.IGNORECASE)
    )

def _grounding_has_only_if_scope(text):
    patterns = (
        r"(?:只有|只有在|只有当|唯有|仅当|仅在|仅有)[\s\S]{0,48}?(?:才|方可|方能|必须|必需)",
        r"(?<!天)(?:才达到|才谈|才能|才可|才可以|才会|才需|才需要|方可|方能)",
        r"必要条件|必要前提|必不可少|不可或缺|\bonly if\b|\bonly when\b|\bif and only if\b|\bnecessary condition\b",
    )
    if not any(re.search(pattern, text, re.IGNORECASE) for pattern in patterns):
        return False
    # Negated exclusivity phrases (for example, “并非只有 A 才 B”) do not
    # assert that A is necessary. Check the complete local clause so a second
    # marker such as “才能” cannot accidentally undo the negation.
    if re.search(
        r"(?:并非|不是|并不|不一定|未必|不|非)[^。；;！？?!\n]{0,24}"
        r"(?:只有|唯有|仅当|仅在|必要条件|必要前提)[^。；;！？?!\n]{0,56}"
        r"(?:才|方可|方能|必须|必需)",
        text,
    ):
        return False
    negated = re.compile(r"(?:并非|不是|并不|不一定|未必|无需|不需要|不必|没有|非|不)$")
    return not any(
        negated.search(text[max(0, match.start() - 8):match.start()].strip())
        for match in re.finditer(r"才|只有|唯有|仅当|仅在|必要|\bonly\b|\bnecessary\b", text, re.IGNORECASE)
    )

def _grounding_if_then_scope_parts(text):
    match = re.search(
        r"(?:如果|若|假如|倘若|当|一旦)\s*(.{1,64}?)[，,]?\s*(?:则|就|那么)\s*(.{1,96})",
        text,
    )
    if not match:
        match = re.search(
            r"\bif\s+(.{1,80}?)\s*,?\s*then\s+(.{1,96})",
            text,
            re.IGNORECASE,
        )
    if not match:
        return None
    condition = re.sub(r"[，,。；;！？?!\s]+$", "", match.group(1)).strip()
    consequence = re.sub(r"^[，,。；;！？?!\s]+", "", match.group(2)).strip()
    consequence = re.sub(r"^(?:可以|能够|能|会|must|can|may)\s*", "", consequence, flags=re.IGNORECASE)
    if not condition or not consequence:
        return None
    return condition, consequence

def _grounding_only_if_scope_parts(text):
    match = re.search(
        r"(?:只有(?:在|当)?|唯有|仅当|仅在|仅有)(.{1,112}?)(才|方可|方能|必须|必需)(.{1,112})",
        text,
    )
    if not match:
        match = re.search(
            r"\bonly\s+(?:if|when)\s+(.{1,80}?)\s*,?\s*(?:then|must|can|may)\s+(.{1,96})",
            text,
            re.IGNORECASE,
        )
        if not match:
            return None
        condition = match.group(1).strip()
        consequence = match.group(2).strip()
    else:
        before_necessity = match.group(1).strip()
        condition = before_necessity
        prefix_consequence = ""
        condition_boundary = re.search(r"(?:时|情况下|之时)[，,]?", before_necessity)
        if condition_boundary:
            condition = before_necessity[:condition_boundary.start()].strip()
            prefix_consequence = before_necessity[condition_boundary.end():].strip(" ，,")
        elif "，" in before_necessity or "," in before_necessity:
            delimiter = max(before_necessity.rfind("，"), before_necessity.rfind(","))
            condition = before_necessity[:delimiter].strip()
            prefix_consequence = before_necessity[delimiter + 1:].strip()
        consequence = " ".join(part for part in (prefix_consequence, match.group(3).strip()) if part)

    condition = re.sub(r"(?:时|情况下|之时)$", "", condition.strip(" ，,。；;" )).strip()
    consequence = re.sub(r"^[，,。；;！？?!\s]+", "", consequence).strip()
    consequence = re.sub(
        r"^(?:才|方可|方能|必须|必需|可以|能够|能|会|may|can|must)\s*",
        "",
        consequence,
        flags=re.IGNORECASE,
    )
    if not condition or not consequence:
        return None
    return condition, consequence

def _grounding_scope_pair_related(left, right):
    if not left or not right:
        return False

    def phrase_related(left_phrase, right_phrase):
        left_core = _grounding_scope_core(left_phrase)
        right_core = _grounding_scope_core(right_phrase)
        if not left_core or not right_core:
            return False
        if left_core == right_core:
            return True
        left_han = re.findall(r"[\u3400-\u9fff]+", left_core)
        right_han = re.findall(r"[\u3400-\u9fff]+", right_core)
        left_bigrams = {
            run[index:index + 2]
            for run in left_han
            for index in range(len(run) - 1)
        }
        right_bigrams = {
            run[index:index + 2]
            for run in right_han
            for index in range(len(run) - 1)
        }
        if len(left_bigrams & right_bigrams) >= 2:
            return True
        if min(len(left_core), len(right_core)) >= 2 and (
            left_core in right_core or right_core in left_core
        ):
            return True
        left_tokens = set(re.findall(r"[a-z][a-z0-9_-]*|\d+", left_core))
        right_tokens = set(re.findall(r"[a-z][a-z0-9_-]*|\d+", right_core))
        return any(len(token) >= 4 for token in left_tokens & right_tokens)

    return bool(
        phrase_related(left[0], right[0])
        and phrase_related(left[1], right[1])
    )

def _grounding_source_proposition_contexts(quotes, chunk):
    """Build bounded diagnostic contexts from selected, adjacent source lines."""
    def strip_timestamp(text):
        return re.sub(r"^\[\d{2}:\d{2}:\d{2}[,.]\d{3}\]\s*", "", text.strip())

    def sentence_parts(text):
        parts = []
        cursor = 0
        explicit_clause_start = (
            r"(?:若|如果|假如|倘若|只有(?:在|当)?|唯有|仅当|仅在|"
            r"此外|另外|但是|不过|然而|相反|另一方面|需要注意|注意|"
            r"if\b|only\b)"
        )
        boundary_pattern = re.compile(
            r"[。；;！？?!\n]+|[，,]\s*(?=" + explicit_clause_start + r")",
            re.IGNORECASE,
        )
        for boundary in boundary_pattern.finditer(text):
            value = text[cursor:boundary.start()].strip()
            if value:
                parts.append((value, True))
            cursor = boundary.end()
        tail = text[cursor:].strip()
        if tail:
            parts.append((tail, False))
        return parts

    raw_quotes = [quote.strip() for quote in quotes]
    normalized_quotes = [strip_timestamp(quote) for quote in raw_quotes]
    contexts = [
        sentence
        for quote in normalized_quotes
        for sentence, _ended in sentence_parts(quote)
    ]
    if not isinstance(chunk, str):
        return list(dict.fromkeys(contexts))

    source_lines = chunk.splitlines()
    positions = set()

    def matching_sequences(quote_lines, *, normalized):
        matches = []
        if not quote_lines:
            return matches
        for start in range(len(source_lines) - len(quote_lines) + 1):
            source_slice = source_lines[start:start + len(quote_lines)]
            if normalized:
                equal = all(
                    strip_timestamp(source_line) == strip_timestamp(quote_line)
                    for source_line, quote_line in zip(source_slice, quote_lines)
                )
            else:
                equal = all(
                    source_line.strip() == quote_line.strip()
                    for source_line, quote_line in zip(source_slice, quote_lines)
                )
            if equal:
                matches.extend(range(start, start + len(quote_lines)))
        return matches

    for raw_quote in raw_quotes:
        raw_lines = raw_quote.splitlines() or [raw_quote]
        exact_positions = matching_sequences(raw_lines, normalized=False)
        if exact_positions:
            positions.update(exact_positions)
            continue
        # Timestamp-free legacy quotes can still identify source text. Keep all
        # matches because normalization alone cannot distinguish repeated lines.
        positions.update(matching_sequences(raw_lines, normalized=True))

    selected_lines = {}
    for line_index in positions:
        # A source line is included only when the exact full line was selected
        # as evidence for this item. The joined text is diagnostic context only.
        selected_lines[line_index] = strip_timestamp(source_lines[line_index])

    continuation_openers = re.compile(
        r"^(?:并且|而且|并|且|那你|那就|那么|则|就|都有|均有|都能|都可|"
        r"从而|因此|所以|进而|称|then)"
    )
    new_clause_openers = re.compile(
        r"^(?:若|如果|假如|倘若|只有|只有在|只有当|唯有|仅当|仅在|"
        r"此外|另外|但是|不过|然而|相反|另一方面|需要注意|注意)"
    )

    def join_parts(left, right):
        separator = " " if re.search(r"[A-Za-z0-9]$", left) and re.match(r"[A-Za-z0-9]", right) else ""
        return left + separator + right

    # Keep independently complete sentences separate. For incomplete ASR lines,
    # allow at most four exact adjacent selected lines to form one local context;
    # sentence starters that introduce a new condition or contrast end the window.
    for start in sorted(selected_lines):
        start_parts = sentence_parts(selected_lines[start])
        if len(start_parts) != 1:
            continue
        context, ended = start_parts[0]
        if ended:
            continue
        contexts_for_start = context
        for offset in range(1, 4):
            next_index = start + offset
            if next_index not in selected_lines or ended:
                break
            next_parts = sentence_parts(selected_lines[next_index])
            if len(next_parts) != 1:
                break
            next_text, next_ended = next_parts[0]
            if new_clause_openers.match(next_text) and not continuation_openers.match(next_text):
                break
            contexts_for_start = join_parts(contexts_for_start, next_text)
            contexts.append(contexts_for_start)
            context = next_text
            ended = next_ended
            if ended:
                break

    return list(dict.fromkeys(value for value in contexts if value))

def _grounding_scope_conflicts(item_id, value, quotes, chunk=None):
    conflicts = []
    source_contexts = _grounding_source_proposition_contexts(quotes, chunk)
    for path, candidate_text in _grounding_visible_text_fields(item_id, value):
        for candidate_clause in _grounding_scope_clauses(candidate_text):
            if _grounding_has_unqualified_universal(candidate_clause) and any(
                _grounding_has_qualified_scope(source_clause)
                and _grounding_scope_related(source_clause, candidate_clause)
                and not _different_explicit_quantified_subject(candidate_clause, source_clause, source_contexts)
                for source_clause in source_contexts
            ):
                conflicts.append(f"{path}（超出来源的量词范围）：{candidate_clause[:160]}")
        candidate_condition_clauses = [
            clause.strip()
            for clause in re.split(r"[。；;！？?!\n]+", candidate_text)
            if clause.strip() and _grounding_has_only_if_scope(clause)
        ]
        for candidate_clause in candidate_condition_clauses:
            matching_source_contexts = [
                source_context
                for source_context in source_contexts
                if _grounding_has_if_then(source_context)
                and _grounding_scope_related(source_context, candidate_clause)
            ]
            candidate_parts = _grounding_only_if_scope_parts(candidate_clause)
            necessity_matches_condition = any(
                _grounding_scope_pair_related(
                    candidate_parts,
                    _grounding_if_then_scope_parts(conditional_context),
                )
                and any(
                    _grounding_has_only_if_scope(necessity_context)
                    and _grounding_scope_pair_related(
                        candidate_parts,
                        _grounding_only_if_scope_parts(necessity_context),
                    )
                    for necessity_context in source_contexts
                )
                for conditional_context in matching_source_contexts
            )
            if matching_source_contexts and not necessity_matches_condition:
                conflicts.append(f"{path}（将若则关系加强为必要条件）")
    return list(dict.fromkeys(conflicts))
