"""Frontend: verdict banner, side-by-side engine panes, repair table, execution output."""
import pandas as pd
import streamlit as st
from Backend.Engine1_Deterministic_Parser.syntax_analyzer import GRAMMAR_BNF


def show_lines(text, line):
    st.code("\n".join(f"{'>' if i == line else ' '} {i:>2} | {l}" for i, l in enumerate(text.split("\n"), 1)), language=None)


def _problems(r1):
    if r1["error"]:
        e = r1["error"]
        return [("Lexical error" if r1["stage"] == "lexer" else "Syntax error", e["line"], e["col"], e["msg"])]
    return [("Name error", i["line"], i["col"], i["msg"]) for i in r1["semantic"]["issues"]] + \
           [("Type error", e["line"], e["col"], e["msg"]) for e in r1["types"]["errors"]]


def describe(c):
    if c["op"] == "insert":
        return f"a '{c['symbol']}' is probably missing after character {c['col']} of line {c['line']}"
    return f"the '{c['symbol']}' at character {c['col'] + 1} of line {c['line']} is probably extra"


def render_verdict(res):
    probs = _problems(res["r1"])
    if not probs:
        st.success("Engine 1 found no problems (syntax, names, types)."); return
    st.error(f"{len(probs)} problem(s) found")
    for kind, line, col, msg in probs:
        st.markdown(f"**{kind}**, line {line}, col {col}: {msg}")
    show_lines(res["text"], probs[0][1])


def _engine1(res):
    r1 = res["r1"]
    st.subheader("Engine 1: CFG + name + type checks")
    if not r1["valid"]:
        st.error(f"The {'lexer' if r1['stage'] == 'lexer' else 'parser'} rejected the code")
    elif _problems(r1):
        st.warning("Syntax valid (derivable from the grammar), but name/type rules are violated")
        st.caption("A CFG cannot express these rules: name and type constraints are context-sensitive.")
    else:
        st.success("Valid: grammar, names and types all pass")
    st.caption(f"{res['lat']['engine1_ms']:.2f} ms")
    with st.expander("Derivation trace (rules entered, up to the failure)"):
        st.code("\n".join(r1["trace"][-60:]) or "(lexer failed before parsing)", language=None)
    with st.expander("Pushdown automaton: bracket stack"):
        (st.success if r1["pda"]["ok"] else st.error)(r1["pda"]["problem"] or "Stack empty at end: brackets balanced")
        if r1["pda"]["steps"]:
            st.dataframe(pd.DataFrame(r1["pda"]["steps"]), hide_index=True)
    with st.expander("Lexical tokens"):
        if r1["tokens"]:
            st.dataframe(pd.DataFrame(r1["tokens"]), hide_index=True)


def _engine2(res):
    r2, text, fix = res["r2"], res["text"], res["fix"]
    st.subheader("Engine 2: CodeBERT")
    if r2 is None:
        st.info("Enable CodeBERT in the sidebar."); return
    if "error_msg" in r2:
        st.error(r2["error_msg"]); return
    if r2["detected"]:
        st.warning(f"Likely error on line {r2['predicted_line']}: {describe(r2['candidates'][0])}")
        show_lines(text, r2["predicted_line"])
    else:
        st.success("No confident error found")
    st.caption(f"{res['lat']['engine2_ms']:.0f} ms | " + (r2.get("note") or f"lowest token probability {r2['min_prob']:.4f}"))
    if r2["candidates"]:
        st.dataframe(pd.DataFrame(r2["candidates"]).drop(columns="fixed_text"), hide_index=True)
        if fix:
            st.success(f"Hybrid: candidate #{fix['rank']} ({fix['op']} '{fix['symbol']}' on line {fix['line']}) passes the parser")
            st.code(fix["fixed_text"], language="python")
        else:
            st.warning("No top candidate passes the parser.")
    with st.expander("Most surprising tokens"):
        st.dataframe(pd.DataFrame(r2["most_surprising"]), hide_index=True)


def render_dual_pane(res):
    c1, c2 = st.columns(2)
    with c1: _engine1(res)
    with c2: _engine2(res)
    if res["sem_cands"]:
        st.subheader("Name/type repair (candidates verified by Engine 1)")
        st.dataframe(pd.DataFrame(res["sem_cands"]).drop(columns="fixed_text"), hide_index=True)
        f = res["sem_fix"]
        if f:
            st.success(f"Verified fix #{f['rank']}: {f['op']} for '{f['name']}' (source: {f['source']})")
            st.code(f["fixed_text"], language="python")
    agg = res["agg"]
    if "agree_on_validity" in agg:
        st.info(f"Agree on validity: {agg['agree_on_validity']} | Agree on error line: {agg['agree_on_line']}")


def render_execution(res):
    run = res.get("run")
    if run is None:
        return
    st.subheader("Execution (sandboxed subprocess)")
    if not run["ran"]:
        st.info(run["reason"]); return
    if run["stdout"]:
        st.caption("Program output"); st.code(run["stdout"], language=None)
    if run["timed_out"]:
        st.error("Timed out after 3 s (possible infinite loop). Engine 1 cannot predict this: it is the halting problem."); return
    e = run["error"]
    if not e:
        st.success(f"Finished without error in {run['ms']:.0f} ms"); return
    st.error(f"{e['type']}: {e['msg']}" + (f" (line {e['line']})" if e["line"] else ""))
    predicted = {"NameError" for i in res["r1"]["semantic"]["issues"]} | {"TypeError" for _ in res["r1"]["types"]["errors"]}
    if e["type"] in predicted:
        st.success(f"Engine 1 predicted this {e['type']} statically, before running.")
    else:
        st.info(f"{e['type']} is not detectable by the static name/type rules.")


def render_grammar():
    with st.expander("Grammar (BNF)"):
        st.code(GRAMMAR_BNF, language=None)
