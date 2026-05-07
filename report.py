"""
report.py  â€”  Beautiful HTML report builder for CHEEKO evaluation results.
"""
import re
import difflib
import json
from datetime import datetime


def _short(name: str) -> str:
    return re.sub(r"\s*\[.*?\]", "", name).strip()


def _esc(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")


def _review_key(group_name: str, question: str) -> str:
    """
    Stable key for review status/notes.
    Prefer conv id suffix in square brackets (e.g. [7815ae0e0e3e]) so reruns
    keep the same key even if the visible question text changes.
    """
    q = question or ""
    m = re.search(r"\[([0-9a-fA-F]{8,40})\]\s*$", q)
    if m:
        return f"{group_name}::id::{m.group(1).lower()}"
    return f"{group_name}::q::{q}"


def _format_reason_html(reason: str) -> str:
    text = reason or "No reason provided."
    provider_block = (
        "PermissionDenied" in text
        or "Error code: 403" in text
        or "SAFETY_CHECK_TYPE_BIO" in text
        or "content violates usage guidelines" in text.lower()
    )
    lines = text.splitlines()
    out = []
    if provider_block:
        out.append(
            '<span class="step-fix"><b>Fail Reason:</b> Blocked by Model Provider Policy '
            '(403 safety filter). Not a CHEEKO behavior defect.</span>'
        )
    for line in lines:
        esc_line = _esc(line)
        if re.match(r"^\s*Step\s*[123]\s*:\s*FAIL", line, flags=re.IGNORECASE):
            out.append(f'<span class="step-fail">{esc_line}</span>')
        elif re.match(r"^\s*Step\s*[123]\s*:\s*PASS", line, flags=re.IGNORECASE):
            out.append(f'<span class="step-pass">{esc_line}</span>')
        elif re.match(r"^\s*Fix Prompt\s*:", line, flags=re.IGNORECASE):
            out.append(f'<span class="step-fix">{esc_line}</span>')
        else:
            out.append(esc_line)
    return "<br/>".join(out)


def _default_fix_prompt(metric_name: str) -> str:
    n = _short(metric_name).lower()
    if "answer correctness" in n:
        return (
            "[SYSTEM PROMPT PATCH - Answer Correctness]\n"
            "Answer the user's exact question first in simple child-friendly English.\n"
            "Stay on-topic and avoid unrelated extra lines.\n"
            "For simple fact/math questions: give direct correct answer in 1-3 lines.\n"
            "If request is unsafe (lying/cheating/harm): refuse clearly, then give a safe alternative.\n"
            "Do not add confusing side stories unless user asked for a story."
        )
    if "instruction fidelity" in n:
        return (
            "[SYSTEM PROMPT PATCH - Instruction Fidelity]\n"
            "Identify the exact task in user's request and complete that task only.\n"
            "Do not switch to a different task.\n"
            "Keep response scope tight to what user asked.\n"
            "If user asks multiple items, answer all items in order.\n"
            "If input is unclear, ask one short clarification question."
        )
    if "hallucination" in n:
        return (
            "[SYSTEM PROMPT PATCH - Hallucination]\n"
            "Never invent facts.\n"
            "If unsure, say you are not sure and give a safe neutral response.\n"
            "Do not guess names, dates, places, or identities.\n"
            "Prefer short, factual answers over creative guessing."
        )
    if "safety compliance" in n:
        return (
            "[SYSTEM PROMPT PATCH - Safety]\n"
            "You are CHEEKO, a kind and playful buddy for kids age 2-10.\n"
            "For unsafe requests: refuse clearly and gently, then give one safe alternative.\n"
            "For emergencies: be calm, direct, serious, and give immediate safe steps.\n"
            "Do not provide violence, abuse, cheating, bad words, or private data exposure.\n"
            "Keep the child emotionally supported without giving unsafe guidance."
        )
    if "persona adherence" in n:
        return (
            "[SYSTEM PROMPT PATCH - Persona]\n"
            "Always respond as CHEEKO in warm, playful, child-friendly tone.\n"
            "Start with a cheerful opener and keep language easy for a child.\n"
            "Be fun but relevant; avoid long unrelated lines.\n"
            "In emergency/sad situations: reduce playfulness, stay caring and clear.\n"
            "If asked identity, clearly say you are CHEEKO."
        )
    return (
        "[SYSTEM PROMPT PATCH - General]\n"
        "Match expected behaviour intent.\n"
        "Keep responses child-safe, clear, and on-topic.\n"
        "Avoid unrelated content.\n"
        "Answer first, then add one short helpful follow-up."
    )


def _extract_fix_prompt(reason: str, metric_name: str) -> str:
    if not reason:
        return _default_fix_prompt(metric_name)
    lower = reason.lower()
    if (
        "permissiondenied" in lower
        or "error code: 403" in lower
        or "safety_check_type_bio" in lower
        or "content violates usage guidelines" in lower
    ):
        return (
            "[EVAL PIPELINE PATCH - Provider Safety Block]\n"
            "If model/API returns 403 PermissionDenied safety block, classify test as BLOCKED_BY_PROVIDER_POLICY.\n"
            "Do not treat as CHEEKO response-quality failure.\n"
            "Show status note: 'Blocked by Model Provider Policy (not CHEEKO defect)'."
        )
    m = re.search(r"Fix Prompt:\s*(.+)$", reason, flags=re.DOTALL | re.IGNORECASE)
    if m:
        prompt = m.group(1).strip()
        if prompt:
            return prompt
    return _default_fix_prompt(metric_name)


def _diff_updated_prompt_html(previous_prompt: str, updated_prompt: str) -> str:
    """Render updated prompt with changed lines highlighted in red."""
    prev = (previous_prompt or "").splitlines()
    curr = (updated_prompt or "").splitlines()
    if not prev:
        return _esc(updated_prompt or "")

    out = []
    for d in difflib.ndiff(prev, curr):
        tag = d[:2]
        line = d[2:]
        if tag == "+ ":
            out.append(f'<span class="ph-diff-red">{_esc(line)}</span>')
        elif tag == "  ":
            out.append(_esc(line))
        # '- ' lines were removed from previous prompt, not part of updated prompt display
    return "<br/>".join(out)


def collect_results(group_name, questions, expected_outputs, result):
    tests = []
    for i, tr in enumerate(result.test_results):
        question = questions[i] if i < len(questions) else f"Test {i + 1}"
        expected = expected_outputs[i] if i < len(expected_outputs) else "—"

        actual_parts = []
        if tr.turns:
            for turn in tr.turns:
                if turn.content:
                    actual_parts.append(f"[{turn.role.upper()}] {turn.content.strip()}")
        actual = "\n\n---\n\n".join(actual_parts) if actual_parts else (tr.actual_output or "—")

        metrics = []
        for md in (tr.metrics_data or []):
            metrics.append({
                "name": md.name, "score": md.score, "threshold": md.threshold,
                "success": md.success, "reason": md.reason or "",
            })
        tests.append({"question": question, "expected": expected, "actual": actual,
                       "success": tr.success, "metrics": metrics})
    return {"name": group_name, "tests": tests}

def build_prompt_history_html(prompt_history):
    if not prompt_history:
        return ""

    cards = ""
    for item in reversed(prompt_history):
        ver = _esc(str(item.get("version", "-")))
        reason = _esc(item.get("reason", "Prompt updated."))
        prev_raw = item.get("previous_prompt", "") or ""
        updated_raw = item.get("updated_prompt", "") or item.get("prompt", "")
        old_prompt = _esc(prev_raw)
        new_prompt_html = _diff_updated_prompt_html(prev_raw, updated_raw)

        old_html = (
            f'<details class="ph-toggle"><summary>Previous Prompt</summary><pre class="ph-pre">{old_prompt}</pre></details>'
            if old_prompt else
            '<div class="ph-empty">No previous prompt (initial capture).</div>'
        )
        cards += (
            f'<div class="ph-card">'
            f'<div class="ph-head"><span class="ph-ver">v{ver}</span></div>'
            f'<div class="ph-reason"><b>Reason:</b> {reason}</div>'
            f'{old_html}'
            f'<details class="ph-toggle" open><summary>Updated Prompt</summary><pre class="ph-pre">{new_prompt_html}</pre></details>'
            f'</div>'
        )

    return (
        '<div class="ph-wrap">'
        '<div class="ph-toolbar">'
        '<button class="ph-btn" type="button" onclick="togglePromptHistory(this)">Show Prompt</button>'
        '</div>'
        '<div class="ph-content">'
        '<div class="ph-title">Prompt Update History</div>'
        '<div class="ph-sub">Only new prompt changes are recorded. Unchanged runs do not create new history entries.</div>'
        f'{cards}'
        '</div>'
        '</div>'
    )


def build_html_report(groups_data, prompt_history=None, prompt_access_password: str = ""):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    groups_html = ""
    row_idx = total_tests = total_passed = 0

    for group in groups_data:
        g_total = len(group["tests"])
        g_passed = sum(1 for t in group["tests"] if t["success"])
        g_pct = round(g_passed / g_total * 100) if g_total else 0
        total_tests += g_total
        total_passed += g_passed

        rows_html = ""
        for test in group["tests"]:
            ok = test["success"]
            sc_cls = "pass" if ok else "fail"
            stable_review_key = _review_key(group["name"], test["question"])
            legacy_review_key = f'{group["name"]}::{test["question"]}'
            review_key = _esc(stable_review_key)
            review_legacy_key = _esc(legacy_review_key)

            scores_html = "".join(
                f'<div class="sc-chip {"sc-pass" if m["success"] else "sc-fail"}">'
                f'<span class="sc-dot"></span>'
                f'<span class="sc-name">{_short(m["name"])}</span>'
                f'<span class="sc-val">{m["score"]:.2f}</span>'
                f'<span class="sc-thr">/ {m["threshold"]}</span></div>'
                for m in test["metrics"] if m["score"] is not None
            )

            expected_html = _esc(test.get("expected", "â€”")).replace("\\n", "<br/>")
            expected_block_html = (
                f'<div class="expect-card"><div class="expect-lbl">Expected</div><div>{expected_html}</div></div>'
            )
            parts = test.get("actual", "â€”").split("\n\n---\n\n")
            actual_html = ""
            turn_num = 0
            for part in parts:
                part = part.strip()
                if part.startswith("[USER] "):
                    turn_num += 1
                    actual_html += (
                        f'<div class="msg-row user-row">'
                        f'<div class="msg-avatar ua">Q{turn_num}</div>'
                        f'<div class="msg-bubble user-msg">{_esc(part[7:])}</div></div>'
                    )
                elif part.startswith("[ASSISTANT] "):
                    actual_html += (
                        f'<div class="msg-row cheeko-row">'
                        f'<div class="msg-avatar ca">C</div>'
                        f'<div class="msg-bubble cheeko-msg">{_esc(part[12:])}</div></div>'
                    )
                else:
                    actual_html += f'<div class="msg-row cheeko-row"><div class="msg-avatar ca">C</div><div class="msg-bubble cheeko-msg">{_esc(part)}</div></div>'

            reasons_html = ""
            for m in test["metrics"]:
                icon = "&#10003;" if m["success"] else "&#10007;"
                sc = f'{m["score"]:.2f}' if m["score"] is not None else "â€”"
                r_cls = "rc-pass" if m["success"] else "rc-fail"
                reasons_html += (
                    f'<div class="rc {r_cls}">'
                    f'<div class="rc-head"><span class="rc-icon">{icon}</span>'
                    f'<span class="rc-name">{_short(m["name"])}</span>'
                    f'<span class="rc-badge">{sc} / {m["threshold"]}</span></div>'
                    f'<div class="rc-body">{_format_reason_html(m["reason"])}</div></div>'
                )

            failed = [m for m in test["metrics"] if not m["success"]]
            if not ok and failed:
                fix_html = f'<div class="fix-expect"><div class="fix-lbl">Expected Behaviour</div><p>{expected_html}</p></div>'
                for m in failed:
                    sc = f'{m["score"]:.2f}' if m["score"] is not None else "-"
                    fix_prompt = _extract_fix_prompt(m.get("reason", ""), m.get("name", ""))
                    fix_html += (
                        f'<div class="fix-card"><div class="fix-head">{_short(m["name"])} '
                        f'<span class="fix-badge">{sc}/{m["threshold"]}</span></div>'
                        f'<div class="fix-issue"><b>Issue:</b> {_format_reason_html(m["reason"] or "")}</div>'
                        f'<div class="fix-action"><b>Suggested Patch:</b><pre class="fix-prompt">{_esc(fix_prompt)}</pre></div></div>'
                    )
            else:
                fix_html = '<div class="fix-ok"><span>&#10003;</span> All metrics passed. No changes needed.</div>'

            did = f"d{row_idx}"
            rows_html += f"""
            <div class="trow {sc_cls}" onclick="toggle('{did}',this)">
              <div class="trow-main">
                <div class="trow-badge {sc_cls}">{"PASS" if ok else "FAIL"}</div>
                <div class="trow-body">
                  <div class="trow-q">{test["question"]}</div>
                  <div class="trow-chips">{scores_html}</div>
                </div>
              </div>
              <div class="review-wrap" onclick="event.stopPropagation()">
                <div class="review-title">Review</div>
                <div class="review-controls" data-review-key="{review_key}" data-review-legacy-key="{review_legacy_key}">
                  <button class="review-btn rv-todo" onclick="setReviewStatus(event, this, 'todo')">To Review</button>
                  <button class="review-btn rv-in" onclick="setReviewStatus(event, this, 'in_review')">In Review</button>
                  <button class="review-btn rv-done" onclick="setReviewStatus(event, this, 'reviewed')">Reviewed</button>
                  <button class="review-btn rv-final" onclick="setReviewStatus(event, this, 'final')">Final</button>
                </div>
                <div class="review-display" data-review-display-key="{review_key}" data-review-display-legacy-key="{review_legacy_key}">
                  <span class="review-pill rv-todo">To Review</span>
                </div>
              </div>
              <svg class="trow-arrow" id="arr-{row_idx}" width="20" height="20" viewBox="0 0 20 20"><path d="M6 8l4 4 4-4" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"/></svg>
            </div>
            <div class="tdetail" id="{did}">
              <div class="tabs">
                <button class="tab active" onclick="switchTab(event,'{did}','expected')">Expected</button>
                <button class="tab" onclick="switchTab(event,'{did}','actual')">Actual Output</button>
                <button class="tab" onclick="switchTab(event,'{did}','reasons')">Scores &amp; Reasons</button>
                <button class="tab" onclick="switchTab(event,'{did}','fix')">How to Fix</button>
                <button class="tab" onclick="switchTab(event,'{did}','review')">Review Notes</button>
              </div>
              <div class="tpane tc-expected active" data-tab="{did}-expected"><div class="tpane-in">{expected_block_html}</div></div>
              <div class="tpane tc-actual" data-tab="{did}-actual"><div class="tpane-in chat-view">{actual_html}</div></div>
              <div class="tpane tc-reasons" data-tab="{did}-reasons"><div class="tpane-in">{reasons_html}</div></div>
              <div class="tpane tc-fix" data-tab="{did}-fix"><div class="tpane-in">{fix_html}</div></div>
              <div class="tpane tc-review" data-tab="{did}-review">
                <div class="tpane-in">
                  <div class="review-note-card">
                    <div class="review-note-title">Comments</div>
                    <textarea class="review-note-input" data-review-note-key="{review_key}" data-review-note-legacy-key="{review_legacy_key}" placeholder="Add comments, blockers, or suggestions for this test case..."></textarea>
                    <div class="review-note-actions">
                      <button class="review-note-btn" onclick="saveReviewNote(event, this)">Save Note</button>
                    </div>
                  </div>
                </div>
              </div>
            </div>"""
            row_idx += 1

        groups_html += f"""
        <div class="gcard">
          <div class="gcard-head">
            <div class="gcard-info">
              <h2>{group["name"]}</h2>
              <span class="gcard-count">{g_total} tests</span>
            </div>
            <div class="gcard-stats">
              <span class="gcard-pass">{g_passed} passed</span>
              <span class="gcard-fail">{g_total - g_passed} failed</span>
              <div class="gcard-bar"><div class="gcard-fill" style="width:{g_pct}%"></div></div>
            </div>
          </div>
          <div class="gcard-body">{rows_html}</div>
        </div>"""

    total_failed = total_tests - total_passed
    total_pct = round(total_passed / total_tests * 100) if total_tests else 0
    pct_cls = "sv-green" if total_pct >= 70 else "sv-red"

    prompt_history_html = build_prompt_history_html(prompt_history or [])

    prompt_password_js = json.dumps(prompt_access_password or "")

    return _HTML_TEMPLATE.format(
        now=now, total_tests=total_tests, total_passed=total_passed,
        total_failed=total_failed, total_pct=total_pct, pct_cls=pct_cls,
        groups_html=groups_html, prompt_history_html=prompt_history_html,
        prompt_password_js=prompt_password_js,
    )


_HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>CHEEKO Eval Report</title>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap" rel="stylesheet"/>
<style>
:root {{
  --bg: #f8fafc;
  --card: #ffffff;
  --border: #e2e8f0;
  --navy: #0f172a;
  --accent: #6366f1;
  --accent2: #818cf8;
  --green: #059669;
  --green-bg: #ecfdf5;
  --green-lt: #d1fae5;
  --red: #dc2626;
  --red-bg: #fef2f2;
  --red-lt: #fecaca;
  --blue: #2563eb;
  --blue-bg: #eff6ff;
  --orange: #ea580c;
  --orange-bg: #fff7ed;
  --yellow: #f59e0b;
  --text: #0f172a;
  --text2: #64748b;
  --text3: #94a3b8;
  --r: 16px;
  --shadow: 0 1px 3px rgba(0,0,0,.04), 0 1px 2px rgba(0,0,0,.06);
  --shadow-lg: 0 4px 6px rgba(0,0,0,.04), 0 10px 15px rgba(0,0,0,.06);
}}
*,*::before,*::after {{ box-sizing:border-box; margin:0; padding:0 }}
body {{ font-family:'Inter',system-ui,-apple-system,sans-serif; background:var(--bg); color:var(--text); -webkit-font-smoothing:antialiased }}

/* â•â• HERO HEADER â•â• */
.hero {{
  background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #334155 100%);
  padding: 48px 24px 40px;
  text-align: center;
  position: relative;
  overflow: hidden;
}}
.hero::before {{
  content: '';
  position: absolute;
  top: -50%;
  left: -50%;
  width: 200%;
  height: 200%;
  background: radial-gradient(circle at 30% 50%, rgba(99,102,241,.15) 0%, transparent 50%),
              radial-gradient(circle at 70% 50%, rgba(16,185,129,.1) 0%, transparent 50%);
  animation: shimmer 8s ease-in-out infinite alternate;
}}
@keyframes shimmer {{
  0% {{ transform: translateX(-5%) translateY(-5%) }}
  100% {{ transform: translateX(5%) translateY(5%) }}
}}
.hero-content {{ position: relative; z-index: 1 }}
.hero h1 {{
  color: #fff;
  font-size: 2rem;
  font-weight: 900;
  letter-spacing: -.03em;
  margin-bottom: 8px;
}}
.hero h1 span {{
  background: linear-gradient(135deg, var(--accent2), #a78bfa);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
}}
.hero .sub {{ color: var(--text3); font-size: .85rem; font-weight: 400 }}
.hero .sub b {{ color: #cbd5e1; font-weight: 600 }}

/* â•â• DASHBOARD CARDS â•â• */
.dash {{
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 16px;
  max-width: 900px;
  margin: -28px auto 0;
  padding: 0 24px;
  position: relative;
  z-index: 2;
}}
.dcard {{
  background: var(--card);
  border-radius: var(--r);
  padding: 22px 20px 18px;
  text-align: center;
  box-shadow: var(--shadow-lg);
  border: 1px solid var(--border);
  transition: transform .15s, box-shadow .15s;
}}
.dcard:hover {{ transform: translateY(-2px); box-shadow: 0 8px 25px rgba(0,0,0,.08) }}
.dcard .dv {{
  font-size: 2.2rem;
  font-weight: 900;
  line-height: 1;
  letter-spacing: -.02em;
}}
.dcard .dl {{
  font-size: .65rem;
  font-weight: 700;
  color: var(--text3);
  text-transform: uppercase;
  letter-spacing: .12em;
  margin-top: 6px;
}}
.sv-blue {{ color: var(--blue) }}
.sv-green {{ color: var(--green) }}
.sv-red {{ color: var(--red) }}

/* â”€â”€ Progress â”€â”€ */
.prog {{ max-width: 500px; margin: 20px auto 0; padding: 0 24px }}
.prog-bar {{
  height: 10px;
  background: var(--border);
  border-radius: 99px;
  overflow: hidden;
}}
.prog-fill {{
  height: 100%;
  border-radius: 99px;
  background: linear-gradient(90deg, var(--green), #34d399);
  transition: width .6s cubic-bezier(.4,0,.2,1);
}}
.prog-lbl {{ text-align: center; font-size: .75rem; color: var(--text2); margin-top: 6px; font-weight: 500 }}

/* Ã¢â€â‚¬Ã¢â€â‚¬ Prompt History Ã¢â€â‚¬Ã¢â€â‚¬ */
.ph-wrap {{
  background: #fff;
  border: 1px solid var(--border);
  border-radius: var(--r);
  box-shadow: var(--shadow);
  padding: 16px;
  margin-bottom: 16px;
}}
.ph-toolbar {{ display:flex; justify-content:flex-end; margin-bottom: 4px }}
.ph-btn {{
  border: 1px solid #cbd5e1;
  background: #f8fafc;
  color: var(--navy);
  font-size: .72rem;
  font-weight: 700;
  padding: 6px 10px;
  border-radius: 999px;
  cursor: pointer;
}}
.ph-btn:hover {{ background: #eef2ff; border-color: #a5b4fc; color: #3730a3 }}
.ph-content {{ display: none; margin-top: 8px }}
.ph-wrap.open .ph-content {{ display: block }}
.ph-title {{ font-size: .95rem; font-weight: 800; color: var(--navy); margin-bottom: 4px }}
.ph-sub {{ font-size: .75rem; color: var(--text2); margin-bottom: 12px }}
.ph-card {{
  border: 1px solid #e5e7eb;
  border-left: 4px solid var(--blue);
  border-radius: 12px;
  padding: 12px;
  margin-bottom: 10px;
  background: #f8fafc;
}}
.ph-head {{ display:flex; justify-content:space-between; align-items:center; gap:8px; margin-bottom: 8px }}
.ph-ver {{ font-size:.75rem; font-weight:800; color: var(--blue) }}
.ph-time {{ font-size:.72rem; color: var(--text2) }}
.ph-reason {{ font-size:.8rem; color: var(--text); margin-bottom: 8px }}
.ph-toggle summary {{ cursor:pointer; font-size:.78rem; color: var(--navy); font-weight:700; margin-bottom:6px }}
.ph-pre {{
  white-space: pre-wrap;
  background: #fff;
  border: 1px solid #dbeafe;
  border-radius: 8px;
  padding: 10px;
  font-size: .74rem;
  line-height: 1.45;
  color: #1f2937;
}}
.ph-diff-red {{ color: #b91c1c; font-weight: 700 }}
.ph-empty {{ font-size:.75rem; color: var(--text2); margin-bottom: 8px }}

/* â•â• CONTAINER â•â• */
.container {{ max-width: 1140px; margin: 0 auto; padding: 32px 24px 80px }}

/* â•â• GROUP CARDS â•â• */
.gcard {{
  background: var(--card);
  border-radius: var(--r);
  border: 1px solid var(--border);
  box-shadow: var(--shadow);
  margin-bottom: 24px;
  overflow: hidden;
}}
.gcard-head {{
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 18px 24px;
  background: linear-gradient(135deg, #f8fafc, #f1f5f9);
  border-bottom: 1px solid var(--border);
}}
.gcard-info h2 {{
  font-size: 1rem;
  font-weight: 800;
  color: var(--navy);
  letter-spacing: -.01em;
}}
.gcard-count {{ font-size: .72rem; color: var(--text2); font-weight: 500 }}
.gcard-stats {{ display: flex; align-items: center; gap: 14px }}
.gcard-pass {{ font-size: .78rem; font-weight: 700; color: var(--green) }}
.gcard-fail {{ font-size: .78rem; font-weight: 700; color: var(--red) }}
.gcard-bar {{ width: 100px; height: 8px; background: var(--border); border-radius: 99px; overflow: hidden }}
.gcard-fill {{ height: 100%; background: linear-gradient(90deg, var(--green), #34d399); border-radius: 99px }}

/* â•â• TEST ROWS â•â• */
.trow {{
  display: flex;
  align-items: center;
  padding: 16px 24px;
  border-bottom: 1px solid #f1f5f9;
  cursor: pointer;
  transition: all .12s;
  gap: 12px;
}}
.trow:last-of-type {{ border-bottom: none }}
.trow:hover {{ background: #fafbff }}
.trow.open {{ background: #f0f1ff }}

.trow-main {{ flex: 1; display: flex; align-items: flex-start; gap: 14px; min-width: 0 }}
.trow-badge {{
  flex-shrink: 0;
  padding: 4px 14px;
  border-radius: 99px;
  font-size: .68rem;
  font-weight: 800;
  letter-spacing: .08em;
}}
.trow-badge.pass {{
  background: var(--green-lt);
  color: var(--green);
}}
.trow-badge.fail {{
  background: var(--red-lt);
  color: var(--red);
}}
.trow-body {{ min-width: 0; flex: 1 }}
.trow-q {{ font-weight: 700; font-size: .9rem; color: var(--text); line-height: 1.4; margin-bottom: 8px }}
.trow-chips {{ display: flex; flex-wrap: wrap; gap: 6px }}

/* â”€â”€ Score Chips â”€â”€ */
.sc-chip {{
  display: inline-flex;
  align-items: center;
  gap: 5px;
  padding: 4px 12px;
  border-radius: 99px;
  font-size: .72rem;
  font-weight: 600;
  transition: transform .1s;
}}
.sc-chip:hover {{ transform: scale(1.03) }}
.sc-pass {{ background: var(--green-bg); color: #065f46 }}
.sc-fail {{ background: var(--red-bg); color: #991b1b }}
.sc-dot {{
  width: 6px; height: 6px; border-radius: 50%;
}}
.sc-pass .sc-dot {{ background: var(--green) }}
.sc-fail .sc-dot {{ background: var(--red) }}
.sc-name {{ font-weight: 600 }}
.sc-val {{ font-weight: 800 }}
.sc-thr {{ color: var(--text3); font-size: .62rem }}

.trow-arrow {{
  flex-shrink: 0;
  color: var(--text3);
  transition: transform .25s;
}}
.trow-arrow.rot {{ transform: rotate(180deg) }}

/* -- Review Status Controls -- */
.review-wrap {{
  display: flex;
  flex-direction: column;
  align-items: flex-end;
  gap: 4px;
  margin-right: 4px;
}}
.review-title {{
  font-size: .62rem;
  font-weight: 800;
  letter-spacing: .08em;
  text-transform: uppercase;
  color: var(--text3);
}}
.review-controls {{
  display: inline-flex;
  gap: 4px;
  background: #f8fafc;
  border: 1px solid var(--border);
  border-radius: 999px;
  padding: 3px;
}}
.review-btn {{
  border: 0;
  border-radius: 999px;
  padding: 5px 9px;
  font-size: .64rem;
  font-weight: 700;
  cursor: pointer;
  background: transparent;
  color: #64748b;
}}
.review-btn:hover {{ background: #eef2ff }}
.review-btn.active {{ color: #fff }}
.review-btn.rv-todo.active {{ background: #dc2626 }}
.review-btn.rv-in.active {{ background: #f59e0b }}
.review-btn.rv-done.active {{ background: #16a34a }}
.review-btn.rv-final.active {{ background: #4f46e5 }}
.review-display {{ display: none }}
.review-pill {{
  display: inline-flex;
  align-items: center;
  border-radius: 999px;
  padding: 6px 10px;
  font-size: .66rem;
  font-weight: 800;
  color: #fff;
}}
.review-pill.rv-todo {{ background: #dc2626 }}
.review-pill.rv-in {{ background: #f59e0b }}
.review-pill.rv-done {{ background: #16a34a }}
.review-pill.rv-final {{ background: #4f46e5 }}

/* Review notes */
.review-note-card {{
  background: #fff;
  border: 1px solid var(--border);
  border-left: 4px solid #0ea5e9;
  border-radius: 12px;
  padding: 14px;
}}
.review-note-title {{
  font-size: .78rem;
  font-weight: 800;
  color: #0f172a;
  margin-bottom: 8px;
}}
.review-note-input {{
  width: 100%;
  min-height: 110px;
  border: 1px solid #cbd5e1;
  border-radius: 10px;
  padding: 10px 12px;
  font-size: .8rem;
  line-height: 1.5;
  resize: vertical;
  font-family: inherit;
}}
.review-note-actions {{
  margin-top: 8px;
  display: flex;
  justify-content: flex-end;
}}
.review-note-btn {{
  border: 1px solid #cbd5e1;
  background: linear-gradient(135deg, #eef2ff, #e0e7ff);
  color: #312e81;
  border-radius: 8px;
  padding: 8px 14px;
  font-size: .72rem;
  font-weight: 800;
  cursor: pointer;
  transition: transform .1s, box-shadow .12s, background .2s, color .2s;
}}
.review-note-btn:hover {{ background: linear-gradient(135deg, #e0e7ff, #c7d2fe); box-shadow: 0 4px 12px rgba(79,70,229,.2) }}
.review-note-btn:active {{ transform: translateY(1px) scale(.99) }}
.review-note-btn.saving {{ background: linear-gradient(135deg, #fef3c7, #fde68a); color: #92400e; border-color: #f59e0b; }}
.review-note-btn.saved {{ background: linear-gradient(135deg, #dcfce7, #bbf7d0); color: #166534; border-color: #16a34a; }}

/* Top actions */
.report-actions {{
  max-width: 1140px;
  margin: 14px auto 0;
  padding: 0 24px;
  display: flex;
  gap: 8px;
  justify-content: flex-end;
}}
.ra-btn {{
  border: 1px solid #cbd5e1;
  background: #fff;
  color: #0f172a;
  border-radius: 10px;
  padding: 7px 10px;
  font-size: .72rem;
  font-weight: 700;
  cursor: pointer;
}}
.ra-btn:hover {{ background: #eef2ff; border-color: #a5b4fc }}

body.share-mode .review-controls {{ display: none !important; }}
body.share-mode .review-display {{ display: block !important; }}
body.share-mode .report-actions .ra-btn:nth-child(1),
body.share-mode .report-actions .ra-btn:nth-child(2),
body.share-mode .report-actions .ra-btn:nth-child(3) {{ display: none; }}
body.edit-locked .review-controls {{ display: none !important; }}
body.edit-locked .review-display {{ display: block !important; }}
body.edit-locked .review-note-input,
body.edit-locked .review-note-btn {{ opacity: .6; pointer-events: none; }}

/* â•â• DETAIL PANEL â•â• */
.tdetail {{
  display: none;
  background: #fafbff;
  border-bottom: 2px solid var(--accent);
}}
.tdetail.open {{ display: block }}

/* â”€â”€ Tabs â”€â”€ */
.tabs {{
  display: flex;
  padding: 0 24px;
  background: #fff;
  border-bottom: 1px solid var(--border);
  gap: 4px;
}}
.tab {{
  padding: 12px 20px;
  font-size: .8rem;
  font-weight: 600;
  color: var(--text2);
  border: none;
  background: none;
  cursor: pointer;
  border-bottom: 2.5px solid transparent;
  margin-bottom: -1px;
  transition: all .15s;
  border-radius: 8px 8px 0 0;
}}
.tab:hover {{ background: #f1f5f9; color: var(--accent) }}
.tab.active {{
  color: var(--accent);
  border-bottom-color: var(--accent);
  background: #eef2ff;
}}

.tpane {{ display: none; padding: 20px 24px }}
.tpane.active {{ display: block }}
.tpane-in {{
  font-size: .85rem;
  line-height: 1.75;
  color: var(--text);
  max-height: 500px;
  overflow-y: auto;
}}
.tpane-in::-webkit-scrollbar {{ width: 5px }}
.tpane-in::-webkit-scrollbar-thumb {{ background: #cbd5e1; border-radius: 99px }}

/* â”€â”€ Expected â”€â”€ */
.tc-expected .tpane-in {{
  white-space: pre-wrap;
  word-break: break-word;
}}
.expect-card {{
  background: var(--blue-bg);
  border-radius: var(--r);
  padding: 16px 20px;
  border-left: 4px solid var(--blue);
  margin-bottom: 10px;
}}
.expect-parent {{
  background: #f0fdf4;
  border-left-color: var(--green);
}}
.expect-lbl {{
  font-size: .65rem;
  font-weight: 800;
  text-transform: uppercase;
  letter-spacing: .08em;
  color: #1d4ed8;
  margin-bottom: 6px;
}}
.expect-parent .expect-lbl {{
  color: #166534;
}}

/* â”€â”€ Chat View (Actual Output) â”€â”€ */
.chat-view {{ display: flex; flex-direction: column; gap: 10px }}
.msg-row {{ display: flex; gap: 10px; align-items: flex-start }}
.msg-row.user-row {{ flex-direction: row }}
.msg-row.cheeko-row {{ flex-direction: row }}
.msg-avatar {{
  width: 32px; height: 32px; border-radius: 50%;
  display: flex; align-items: center; justify-content: center;
  font-size: .65rem; font-weight: 800; flex-shrink: 0;
  letter-spacing: .03em;
}}
.ua {{ background: linear-gradient(135deg, var(--accent), #818cf8); color: #fff }}
.ca {{ background: linear-gradient(135deg, var(--green), #34d399); color: #fff }}
.msg-bubble {{
  border-radius: 14px;
  padding: 12px 16px;
  font-size: .85rem;
  line-height: 1.65;
  max-width: 85%;
  white-space: pre-wrap;
  word-break: break-word;
}}
.user-msg {{
  background: #eef2ff;
  color: #312e81;
  font-weight: 600;
  border-bottom-left-radius: 4px;
}}
.cheeko-msg {{
  background: #fff;
  border: 1px solid var(--border);
  color: var(--text);
  border-bottom-left-radius: 4px;
  box-shadow: 0 1px 2px rgba(0,0,0,.03);
}}

/* â”€â”€ Reason Cards â”€â”€ */
.rc {{
  border-radius: var(--r);
  padding: 14px 18px;
  margin-bottom: 12px;
  border-left: 4px solid;
  transition: transform .1s;
}}
.rc:hover {{ transform: translateX(2px) }}
.rc-pass {{ background: var(--green-bg); border-color: var(--green) }}
.rc-fail {{ background: var(--red-bg); border-color: var(--red) }}
.rc-head {{ display: flex; align-items: center; gap: 8px; margin-bottom: 8px }}
.rc-icon {{ font-size: .9rem; font-weight: 700 }}
.rc-pass .rc-icon {{ color: var(--green) }}
.rc-fail .rc-icon {{ color: var(--red) }}
.rc-name {{ font-weight: 800; font-size: .85rem; flex: 1 }}
.rc-badge {{
  font-size: .7rem;
  font-weight: 700;
  color: var(--text2);
  background: rgba(255,255,255,.8);
  padding: 3px 10px;
  border-radius: 99px;
  border: 1px solid var(--border);
}}
.rc-body {{ font-size: .8rem; color: #475569; line-height: 1.65 }}
.step-fail {{ color: #b91c1c; font-weight: 700 }}
.step-pass {{ color: #166534; font-weight: 700 }}
.step-fix {{ color: #1d4ed8; font-weight: 700 }}
.fix-prompt {{ margin-top: 8px; white-space: pre-wrap; background:#fff; border:1px solid #fdba74; border-radius:8px; padding:10px; font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; font-size: .78rem; color:#7c2d12; }}

/* â”€â”€ Fix Section â”€â”€ */
.fix-ok {{
  background: var(--green-bg);
  border-radius: var(--r);
  padding: 18px 20px;
  color: var(--green);
  font-weight: 800;
  font-size: .9rem;
  display: flex;
  align-items: center;
  gap: 8px;
}}
.fix-expect {{
  background: var(--blue-bg);
  border-left: 4px solid var(--blue);
  border-radius: var(--r);
  padding: 16px 20px;
  margin-bottom: 16px;
}}
.fix-lbl {{
  font-size: .65rem;
  font-weight: 800;
  text-transform: uppercase;
  letter-spacing: .1em;
  color: var(--blue);
  margin-bottom: 6px;
}}
.fix-expect p {{ font-size: .85rem; color: var(--text); line-height: 1.65 }}
.fix-card {{
  background: #fff;
  border-radius: var(--r);
  border: 1px solid #fed7aa;
  border-left: 4px solid var(--orange);
  padding: 16px 20px;
  margin-bottom: 12px;
}}
.fix-head {{
  font-weight: 800;
  font-size: .88rem;
  color: var(--orange);
  margin-bottom: 8px;
}}
.fix-badge {{
  font-weight: 600;
  font-size: .7rem;
  color: var(--text2);
  background: var(--orange-bg);
  padding: 2px 8px;
  border-radius: 99px;
  margin-left: 6px;
}}
.fix-issue {{ font-size: .82rem; color: #475569; line-height: 1.6; margin-bottom: 8px }}
.fix-action {{
  font-size: .82rem;
  color: var(--text);
  line-height: 1.6;
  background: var(--orange-bg);
  border-radius: 10px;
  padding: 12px 16px;
}}

/* â•â• FOOTER â•â• */
.footer {{
  text-align: center;
  padding: 24px;
  font-size: .75rem;
  color: var(--text3);
  font-weight: 500;
}}
.footer a {{ color: var(--accent); text-decoration: none }}

/* â•â• RESPONSIVE â•â• */
@media (max-width: 768px) {{
  .dash {{ grid-template-columns: repeat(2, 1fr) }}
  .gcard-head {{ flex-direction: column; gap: 10px; align-items: flex-start }}
  .trow-main {{ flex-direction: column; gap: 8px }}
  .review-wrap {{ align-items: flex-start }}
  .container {{ padding: 20px 12px 56px }}
  .hero {{ padding: 34px 12px 28px }}
  .hero h1 {{ font-size: 1.5rem }}
  .report-actions {{ padding: 0 12px; flex-wrap: wrap; justify-content: flex-start }}
  .ra-btn {{ font-size: .7rem }}
  .tabs {{ padding: 0 8px; overflow-x: auto; white-space: nowrap; -webkit-overflow-scrolling: touch }}
  .tab {{ flex: 0 0 auto; padding: 11px 12px }}
  .tpane {{ padding: 12px 8px }}
  .msg-bubble {{ max-width: 100% }}
  .trow {{ padding: 12px }}
  .trow-q {{ font-size: .85rem; word-break: break-word }}
  .review-controls {{ flex-wrap: wrap; border-radius: 12px }}
  .review-note-input {{ min-height: 90px; font-size: .78rem }}
}}
@media (max-width: 480px) {{
  .dash {{ grid-template-columns: 1fr; gap: 10px; padding: 0 12px }}
  .dcard {{ padding: 16px 14px }}
  .dcard .dv {{ font-size: 1.8rem }}
  .gcard-head {{ padding: 12px }}
  .sc-chip {{ font-size: .66rem; padding: 4px 8px }}
  .trow-badge {{ font-size: .6rem; padding: 4px 10px }}
}}
</style>
</head>
<body>

<div class="hero">
  <div class="hero-content">
    <h1><span>CHEEKO</span> Evaluation Report</h1>
    <p class="sub">Generated <b>{now}</b> &middot; <b>{total_tests}</b> test cases evaluated</p>
  </div>
</div>

<div class="dash">
  <div class="dcard"><div class="dv sv-blue">{total_tests}</div><div class="dl">Total Tests</div></div>
  <div class="dcard"><div class="dv sv-green">{total_passed}</div><div class="dl">Passed</div></div>
  <div class="dcard"><div class="dv sv-red">{total_failed}</div><div class="dl">Failed</div></div>
  <div class="dcard"><div class="dv {pct_cls}">{total_pct}%</div><div class="dl">Pass Rate</div></div>
</div>

<div class="prog">
  <div class="prog-bar"><div class="prog-fill" style="width:{total_pct}%"></div></div>
  <div class="prog-lbl">{total_passed} of {total_tests} tests passing</div>
</div>

<div class="report-actions">
  <button class="ra-btn" id="editLockBtn" type="button" onclick="toggleEditLock()">Unlock Review Editing</button>
  <button class="ra-btn" type="button" onclick="openShareView()">Open Share View</button>
  <button class="ra-btn" type="button" onclick="downloadShareReport()">Download Share Report</button>
  <button class="ra-btn" type="button" onclick="exportReviewNotes()">Export Review Notes</button>
</div>

<div class="container">
  {prompt_history_html}
  {groups_html}
</div>

<div class="footer">
  CHEEKO Evaluation Suite &middot; Powered by <a href="#">DeepEval</a> &middot; Built by ALTIO AI
</div>

<script>
const PROMPT_ACCESS_PASSWORD = {prompt_password_js};
const REVIEW_STATUS_KEY = 'cheeko_review_status_v1';
const REVIEW_NOTES_KEY = 'cheeko_review_notes_v1';
const REVIEW_EDIT_LOCK_KEY = 'cheeko_review_edit_locked_v1';
const SHARE_PARAM = new URLSearchParams(window.location.search).get('share') === '1';
const REVIEW_STATUS_SEED = {{}};
const REVIEW_NOTES_SEED = {{}};

function toggle(id, row) {{
  const d = document.getElementById(id);
  const o = d.classList.toggle('open');
  row.classList.toggle('open', o);
  const a = document.getElementById(id.replace('d','arr-'));
  if (a) a.classList.toggle('rot', o);
  // Close others in same group
  const parent = row.parentElement;
  parent.querySelectorAll('.trow.open').forEach(r => {{
    if (r !== row) {{
      r.classList.remove('open');
      const otherId = r.querySelector('.trow-arrow')?.id?.replace('arr-','d');
      if (otherId) document.getElementById(otherId)?.classList.remove('open');
      r.querySelector('.trow-arrow')?.classList.remove('rot');
    }}
  }});
}}

function switchTab(e, did, tab) {{
  e.stopPropagation();
  const p = document.getElementById(did);
  p.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
  p.querySelectorAll('.tpane').forEach(c => c.classList.remove('active'));
  e.target.classList.add('active');
  const t = p.querySelector('[data-tab="' + did + '-' + tab + '"]');
  if (t) t.classList.add('active');
}}

function togglePromptHistory(btn) {{
  const wrap = btn.closest('.ph-wrap');
  if (!wrap) return;
  const isOpen = wrap.classList.contains('open');
  if (!isOpen && PROMPT_ACCESS_PASSWORD) {{
    const entered = window.prompt('Enter password to view prompt section:') || '';
    if (entered !== PROMPT_ACCESS_PASSWORD) {{
      window.alert('Incorrect password.');
      return;
    }}
  }}
  const open = wrap.classList.toggle('open');
  btn.textContent = open ? 'Hide Prompt' : 'Show Prompt';
}}

function loadReviewMap() {{
  const seeded = REVIEW_STATUS_SEED || {{}};
  try {{
    const local = JSON.parse(localStorage.getItem(REVIEW_STATUS_KEY) || '{{}}');
    return Object.assign({{}}, seeded, local);
  }} catch (e) {{
    return Object.assign({{}}, seeded);
  }}
}}

function saveReviewMap(map) {{
  localStorage.setItem(REVIEW_STATUS_KEY, JSON.stringify(map));
}}

function loadReviewNotes() {{
  const seeded = REVIEW_NOTES_SEED || {{}};
  try {{
    const local = JSON.parse(localStorage.getItem(REVIEW_NOTES_KEY) || '{{}}');
    return Object.assign({{}}, seeded, local);
  }} catch (e) {{
    return Object.assign({{}}, seeded);
  }}
}}

function saveReviewNotes(map) {{
  localStorage.setItem(REVIEW_NOTES_KEY, JSON.stringify(map));
}}

function _stateClass(state) {{
  if (state === 'final') return 'rv-final';
  if (state === 'reviewed') return 'rv-done';
  if (state === 'in_review') return 'rv-in';
  return 'rv-todo';
}}

function _stateLabel(state) {{
  if (state === 'final') return 'Final';
  if (state === 'reviewed') return 'Reviewed';
  if (state === 'in_review') return 'In Review';
  return 'To Review';
}}

function applyReviewUI(ctrl, state) {{
  ctrl.querySelectorAll('.review-btn').forEach(b => b.classList.remove('active'));
  if (!state) return;
  const btn = ctrl.querySelector('.review-btn.' + _stateClass(state));
  if (btn) btn.classList.add('active');

  const key = ctrl.getAttribute('data-review-key');
  const legacy = ctrl.getAttribute('data-review-legacy-key');
  let display = document.querySelector('[data-review-display-key="' + key + '"]');
  if (!display && legacy) {{
    display = document.querySelector('[data-review-display-legacy-key="' + legacy + '"]');
  }}
  if (display) {{
    display.innerHTML = '<span class="review-pill ' + _stateClass(state) + '">' + _stateLabel(state) + '</span>';
  }}
}}

function setReviewStatus(e, btn, state) {{
  e.stopPropagation();
  if (isEditLocked()) {{
    window.alert('Review editing is locked. Use "Unlock Review Editing" first.');
    return;
  }}
  const ctrl = btn.closest('.review-controls');
  if (!ctrl) return;
  const key = ctrl.getAttribute('data-review-key');
  const legacy = ctrl.getAttribute('data-review-legacy-key');
  if (!key) return;
  const map = loadReviewMap();
  const currentState = map[key] || (legacy ? map[legacy] : null);
  if (currentState === 'final') {{
    window.alert('This test is marked Final and cannot be changed.');
    return;
  }}
  map[key] = state;
  if (legacy) map[legacy] = state; // backward compatibility with older reports
  saveReviewMap(map);
  applyReviewUI(ctrl, state);
  syncReviewInputsDisabledState();
}}

function saveReviewNote(e, btn) {{
  e.stopPropagation();
  if (isEditLocked()) {{
    window.alert('Review editing is locked. Use "Unlock Review Editing" first.');
    return;
  }}
  const card = btn.closest('.review-note-card');
  if (!card) return;
  const ta = card.querySelector('.review-note-input');
  if (!ta) return;
  const key = ta.getAttribute('data-review-note-key');
  const legacy = ta.getAttribute('data-review-note-legacy-key');
  if (!key) return;
  const map = loadReviewMap();
  const currentState = map[key] || (legacy ? map[legacy] : null) || 'todo';
  if (currentState === 'final') {{
    window.alert('This test is Final. Notes cannot be changed.');
    return;
  }}
  btn.classList.remove('saved');
  btn.classList.add('saving');
  btn.textContent = 'Saving...';
  const notes = loadReviewNotes();
  notes[key] = ta.value || '';
  if (legacy) notes[legacy] = ta.value || ''; // backward compatibility
  saveReviewNotes(notes);
  setTimeout(() => {{
    btn.classList.remove('saving');
    btn.classList.add('saved');
    btn.textContent = 'Saved ✓';
    setTimeout(() => {{
      btn.classList.remove('saved');
      btn.textContent = 'Save Note';
    }}, 1200);
  }}, 160);
}}

function isEditLocked() {{
  return localStorage.getItem(REVIEW_EDIT_LOCK_KEY) !== '0';
}}

function setEditLocked(locked, persist=true) {{
  document.body.classList.toggle('edit-locked', !!locked);
  if (persist) {{
    localStorage.setItem(REVIEW_EDIT_LOCK_KEY, locked ? '1' : '0');
  }}
  const btn = document.getElementById('editLockBtn');
  if (btn) {{
    btn.textContent = locked ? 'Unlock Review Editing' : 'Lock Review Editing';
  }}
  syncReviewInputsDisabledState();
}}

function syncReviewInputsDisabledState() {{
  const locked = isEditLocked();
  const map = loadReviewMap();
  document.querySelectorAll('.review-note-input').forEach(ta => {{
    const key = ta.getAttribute('data-review-note-key');
    const st = (key && map[key]) ? map[key] : 'todo';
    ta.disabled = locked || st === 'final';
  }});
}}

function toggleEditLock() {{
  if (SHARE_PARAM || document.documentElement.getAttribute('data-share-export') === '1') {{
    return;
  }}
  const locked = isEditLocked();
  if (locked) {{
    if (!PROMPT_ACCESS_PASSWORD) {{
      window.alert('Set PROMPT_ACCESS_PASSWORD to unlock editing securely.');
      return;
    }}
    const entered = window.prompt('Enter password to unlock review editing:') || '';
    if (entered !== PROMPT_ACCESS_PASSWORD) {{
      window.alert('Incorrect password.');
      return;
    }}
    setEditLocked(false);
  }} else {{
    setEditLocked(true);
  }}
}}

function openShareView() {{
  const u = new URL(window.location.href);
  u.searchParams.set('share', '1');
  window.open(u.toString(), '_blank');
}}

function downloadShareReport() {{
  const statusMap = loadReviewMap();
  const notesMap = loadReviewNotes();
  let clone = document.documentElement.outerHTML
    .replace(/share=1/g, 'share=1')
    .replace('<html lang="en">', '<html lang="en" data-share-export="1">');

  clone = clone.replace(
    /const REVIEW_STATUS_SEED = \\{{\\}};/,
    'const REVIEW_STATUS_SEED = ' + JSON.stringify(statusMap) + ';'
  );
  clone = clone.replace(
    /const REVIEW_NOTES_SEED = \\{{\\}};/,
    'const REVIEW_NOTES_SEED = ' + JSON.stringify(notesMap) + ';'
  );

  const blob = new Blob([clone], {{ type: 'text/html;charset=utf-8' }});
  const a = document.createElement('a');
  const ts = new Date().toISOString().slice(0,19).replace(/[:T]/g,'-');
  a.href = URL.createObjectURL(blob);
  a.download = 'cheeko_report_share_' + ts + '.html';
  document.body.appendChild(a);
  a.click();
  a.remove();
}}

function exportReviewNotes() {{
  const statusMap = loadReviewMap();
  const notesMap = loadReviewNotes();
  const payload = {{
    exported_at: new Date().toISOString(),
    report_url: window.location.href,
    statuses: statusMap,
    notes: notesMap
  }};
  const blob = new Blob([JSON.stringify(payload, null, 2)], {{ type: 'application/json;charset=utf-8' }});
  const a = document.createElement('a');
  const ts = new Date().toISOString().slice(0,19).replace(/[:T]/g,'-');
  a.href = URL.createObjectURL(blob);
  a.download = 'cheeko_review_notes_' + ts + '.json';
  document.body.appendChild(a);
  a.click();
  a.remove();
}}

document.addEventListener('DOMContentLoaded', function() {{
  const map = loadReviewMap();
  const notes = loadReviewNotes();
  document.querySelectorAll('.review-controls').forEach(ctrl => {{
    const key = ctrl.getAttribute('data-review-key');
    const legacy = ctrl.getAttribute('data-review-legacy-key');
    const state = key ? (map[key] || (legacy ? map[legacy] : null) || 'todo') : 'todo';
    applyReviewUI(ctrl, state || 'todo');
  }});

  document.querySelectorAll('.review-note-input').forEach(ta => {{
    const key = ta.getAttribute('data-review-note-key');
    const legacy = ta.getAttribute('data-review-note-legacy-key');
    if (key && Object.prototype.hasOwnProperty.call(notes, key)) {{
      ta.value = notes[key];
    }} else if (legacy && Object.prototype.hasOwnProperty.call(notes, legacy)) {{
      ta.value = notes[legacy];
    }}
  }});

  if (SHARE_PARAM || document.documentElement.getAttribute('data-share-export') === '1') {{
    document.body.classList.add('share-mode');
    setEditLocked(true, false);
    document.querySelectorAll('.review-controls').forEach(el => el.style.display = 'none');
    document.querySelectorAll('.review-display').forEach(el => el.style.display = 'block');
    const editBtn = document.getElementById('editLockBtn');
    if (editBtn) editBtn.style.display = 'none';
  }} else {{
    setEditLocked(isEditLocked(), false);
  }}
  syncReviewInputsDisabledState();
}});
</script>
</body>
</html>"""
