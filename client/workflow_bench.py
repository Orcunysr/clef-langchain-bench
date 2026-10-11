"""Stock-brokerage workflow, implemented twice and benchmarked on the same requests.

A) agent  - LangChain `create_agent` with Qwen3.5-9B (no thinking). The LLM decides
            which tool to call and with which arguments, then writes the reply.
B) clef   - LangGraph `StateGraph`. One clef-flash call classifies intent and ticker,
            code extracts the quantity and runs the tool, the reply is a template.
            Only open-ended requests (intent "other" or low confidence) reach the LLM.

Both talk to server/clef_server.py through the SSH tunnel (scripts/start_server.sh).
Tools are fake; no real orders are placed.

    python client/workflow_bench.py [--repeats 3] [--out results/workflow]
"""

import argparse
import json
import os
import re
import statistics
import time
from pathlib import Path
from typing import Optional, TypedDict

os.environ.setdefault("TYPESAFE_BASE_URL", "http://127.0.0.1:30000")
os.environ.setdefault("TYPESAFE_API_KEY", "local")  # required by the client, ignored by the local server

from langchain.agents import create_agent  # noqa: E402
from langchain_core.messages import AIMessage  # noqa: E402
from langchain_core.tools import tool  # noqa: E402
from langchain_openai import ChatOpenAI  # noqa: E402
from langchain_typesafe import Choice, TypeSafeClassifier  # noqa: E402
from langgraph.graph import END, START, StateGraph  # noqa: E402

BASE_URL = os.environ["TYPESAFE_BASE_URL"].rstrip("/") + "/v1"

# ------------------------------------------------------------------ fake brokerage backend

STOCKS = {
    "THYAO": ("Türk Hava Yolları", 312.50),
    "ASELS": ("Aselsan", 148.20),
    "GARAN": ("Garanti BBVA", 131.90),
    "BIMAS": ("BİM Birleşik Mağazalar", 540.00),
    "TUPRS": ("Tüpraş", 171.30),
    "KCHOL": ("Koç Holding", 198.75),
}
PORTFOLIO = {"THYAO": 200, "BIMAS": 40, "TUPRS": 25, "GARAN": 300}
CALLS = []  # (tool, args) executed during the current request


@tool
def get_price(ticker: str) -> str:
    """Get the current price of a BIST stock by its ticker (e.g. THYAO)."""
    CALLS.append(("get_price", {"ticker": ticker}))
    name, price = STOCKS.get(ticker.upper(), (None, None))
    return f"{ticker.upper()} ({name}): {price} TL" if name else f"Unknown ticker {ticker}"


@tool
def buy_stock(ticker: str, quantity: int) -> str:
    """Place a BUY order for `quantity` lots of the stock `ticker`."""
    CALLS.append(("buy_stock", {"ticker": ticker, "quantity": int(quantity)}))
    return f"Alım emri iletildi: {quantity} lot {ticker.upper()}"


@tool
def sell_stock(ticker: str, quantity: int) -> str:
    """Place a SELL order for `quantity` lots of the stock `ticker`."""
    CALLS.append(("sell_stock", {"ticker": ticker, "quantity": int(quantity)}))
    return f"Satış emri iletildi: {quantity} lot {ticker.upper()}"


@tool
def get_portfolio() -> str:
    """List the stocks and lot counts in the user's portfolio."""
    CALLS.append(("get_portfolio", {}))
    return json.dumps(PORTFOLIO)


TOOLS = [get_price, buy_stock, sell_stock, get_portfolio]
TICKER_HELP = ", ".join(f"{t} = {n}" for t, (n, _) in STOCKS.items())

# ------------------------------------------------------------------ test set (Turkish requests)
# gold: the (tool, args) that must run first; None = no tool may run (open-ended request).

CASES = [
    ("THYAO ne kadar?", ("get_price", {"ticker": "THYAO"})),
    ("Aselsan hissesinin fiyatı nedir?", ("get_price", {"ticker": "ASELS"})),
    ("Koç Holding şu an kaç lira?", ("get_price", {"ticker": "KCHOL"})),
    ("Türk Hava Yollarından 100 lot al", ("buy_stock", {"ticker": "THYAO", "quantity": 100})),
    ("Garanti'den 50 adet satın al", ("buy_stock", {"ticker": "GARAN", "quantity": 50})),
    ("BIMAS 5 lot alım emri ver", ("buy_stock", {"ticker": "BIMAS", "quantity": 5})),
    ("Elimdeki BİM hisselerinden 20 tanesini sat", ("sell_stock", {"ticker": "BIMAS", "quantity": 20})),
    ("Tüpraş'tan 10 lot satmak istiyorum", ("sell_stock", {"ticker": "TUPRS", "quantity": 10})),
    ("Portföyümde neler var?", ("get_portfolio", {})),
    ("Hesabımdaki hisseleri göster", ("get_portfolio", {})),
    ("Garanti hisselerimi satmayacağım, sadece fiyatına bakacaktım", ("get_price", {"ticker": "GARAN"})),
    ("Hisse senedi nedir, nasıl çalışır?", None),
    ("Aselsan alınır mı sence?", None),
    ("Temettü verimi yüksek şirketlerle uzun vadeli bir strateji önerir misin?", None),
    ("Merhaba, nasılsın?", None),
]

llm = ChatOpenAI(model="qwen-fast", base_url=BASE_URL, api_key="local", timeout=300)

# ------------------------------------------------------------------ A) classic agent

agent = create_agent(
    llm,
    tools=TOOLS,
    system_prompt=(
        "Sen bir aracı kurum asistanısın. Kısa ve Türkçe cevap ver. Tickers: " + TICKER_HELP + ". "
        "Sadece kullanıcı açıkça emir verirse alım/satım yap; fikir sorularında işlem yapma."
    ),
)


def run_agent(text):
    result = agent.invoke({"messages": [{"role": "user", "content": text}]}, {"recursion_limit": 8})
    reply = next((m.content for m in reversed(result["messages"]) if isinstance(m, AIMessage) and m.content), "")
    return str(reply), ""


# ------------------------------------------------------------------ B) clef-routed LangGraph

classifier = TypeSafeClassifier()
QUESTIONS = {
    "intent": Choice(
        instructions="What does the user want the brokerage system to do right now?",
        criteria={
            "price": "Look up the current price of a stock",
            "buy": "An explicit order to BUY a stock now",
            "sell": "An explicit order to SELL a stock now",
            "portfolio": "Show the user's own holdings / portfolio",
            "other": "General question, opinion or advice, analysis, chit-chat; no order and no lookup",
        },
    ),
    "ticker": Choice(
        instructions="Which stock does the message refer to?",
        criteria={**{t: n for t, (n, _) in STOCKS.items()}, "none": "No specific stock"},
    ),
}
QUANTITY = re.compile(r"(\d+)\s*(?:lot|adet|tane|hisse)?", re.IGNORECASE)
CONFIDENCE_THRESHOLD = 0.6


class FlowState(TypedDict, total=False):
    request: str
    intent: str
    intent_probability: float
    ticker: str
    quantity: Optional[int]
    reply: str


def classify(state: FlowState) -> FlowState:
    answers = classifier.invoke({"state": state["request"], "questions": QUESTIONS}).choices
    intent, ticker = answers["intent"], answers["ticker"]
    match = QUANTITY.search(state["request"])
    return {
        "intent": intent.choice,
        "intent_probability": intent.probabilities[intent.choice],
        "ticker": ticker.choice,
        "quantity": int(match.group(1)) if match else None,
    }


def route(state: FlowState) -> str:
    intent = state["intent"]
    if intent == "other" or state["intent_probability"] < CONFIDENCE_THRESHOLD:
        return "llm_fallback"
    if intent == "portfolio":
        return "portfolio"
    if state["ticker"] == "none" or (intent in ("buy", "sell") and state["quantity"] is None):
        return "ask_missing"
    return intent


def price(state: FlowState) -> FlowState:
    return {"reply": get_price.invoke({"ticker": state["ticker"]})}


def buy(state: FlowState) -> FlowState:
    return {"reply": buy_stock.invoke({"ticker": state["ticker"], "quantity": state["quantity"]})}


def sell(state: FlowState) -> FlowState:
    return {"reply": sell_stock.invoke({"ticker": state["ticker"], "quantity": state["quantity"]})}


def portfolio(state: FlowState) -> FlowState:
    holdings = json.loads(get_portfolio.invoke({}))
    return {"reply": "Portföyünüz: " + ", ".join(f"{t} {q} lot" for t, q in holdings.items())}


def ask_missing(state: FlowState) -> FlowState:
    if state["ticker"] == "none":
        return {"reply": "Hangi hisse için işlem yapmak istiyorsunuz?"}
    return {"reply": f"{state['ticker']} için kaç lot işlem yapılsın?"}


def llm_fallback(state: FlowState) -> FlowState:
    reply = llm.invoke([("system", "Kısa ve Türkçe cevap ver. İşlem yapma."), ("user", state["request"])])
    return {"reply": str(reply.content)}


builder = StateGraph(FlowState)
for node in (classify, price, buy, sell, portfolio, ask_missing, llm_fallback):
    builder.add_node(node.__name__, node)
builder.add_edge(START, "classify")
builder.add_conditional_edges(
    "classify", route,
    {"price": "price", "buy": "buy", "sell": "sell", "portfolio": "portfolio",
     "ask_missing": "ask_missing", "llm_fallback": "llm_fallback"},
)
for node in ("price", "buy", "sell", "portfolio", "ask_missing", "llm_fallback"):
    builder.add_edge(node, END)
clef_graph = builder.compile()


def run_clef(text):
    state = clef_graph.invoke({"request": text})
    detail = f"intent={state['intent']}({state['intent_probability']:.2f}) ticker={state['ticker']}"
    return state["reply"], detail


# ------------------------------------------------------------------ scoring and reporting

def is_correct(calls, gold):
    if gold is None:
        return not calls
    if not calls:
        return False
    name, args = calls[0]
    normalized = {k: (v.upper() if isinstance(v, str) else v) for k, v in args.items()}
    trading = [c for c in calls if c[0] in {"buy_stock", "sell_stock"}]
    allowed_trades = 1 if gold[0] in {"buy_stock", "sell_stock"} else 0
    return name == gold[0] and normalized == gold[1] and len(trading) <= allowed_trades


def bench(name, runner, repeats):
    runner(CASES[0][0])  # warm-up, not measured
    rows = []
    for text, gold in CASES:
        runs = []
        for _ in range(repeats):
            CALLS.clear()
            started = time.perf_counter()
            reply, detail = runner(text)
            runs.append({"seconds": time.perf_counter() - started, "calls": list(CALLS),
                         "ok": is_correct(list(CALLS), gold), "reply": reply, "detail": detail})
        row = {
            "approach": name, "request": text, "gold": gold,
            "correct_runs": sum(r["ok"] for r in runs), "repeats": repeats,
            "median_s": round(statistics.median(r["seconds"] for r in runs), 3),
            "runs": [{**r, "seconds": round(r["seconds"], 3)} for r in runs],
        }
        rows.append(row)
        last = runs[-1]
        print(f"[{name:5}] {row['correct_runs']}/{repeats} ok  {row['median_s']:6.2f}s  "
              f"{text[:52]:<52} calls={last['calls']} {last['detail']}", flush=True)
    return rows


def summarize(rows):
    tool_rows = [r for r in rows if r["gold"] is not None]
    open_rows = [r for r in rows if r["gold"] is None]
    return {
        "correct": f"{sum(r['correct_runs'] for r in rows)}/{sum(r['repeats'] for r in rows)}",
        "tool_median_s": round(statistics.median(r["median_s"] for r in tool_rows), 2),
        "tool_mean_s": round(statistics.mean(r["median_s"] for r in tool_rows), 2),
        "open_median_s": round(statistics.median(r["median_s"] for r in open_rows), 2),
        "all_median_s": round(statistics.median(r["median_s"] for r in rows), 2),
        "total_s": round(sum(r["median_s"] for r in rows), 1),
    }


def write_report(out: Path, results, summaries, repeats):
    lines = [
        "| Approach | Correct (all runs) | Tool requests, median | Tool requests, mean | "
        "Open-ended, median | All, median | Sum of medians (15 req) |",
        "|---|---|---|---|---|---|---|",
    ]
    for name, s in summaries.items():
        lines.append(f"| {name} | {s['correct']} | {s['tool_median_s']} s | {s['tool_mean_s']} s | "
                     f"{s['open_median_s']} s | {s['all_median_s']} s | {s['total_s']} s |")
    lines += ["", f"Per request (median of {repeats} runs):", "",
              "| Request | Expected | clef | agent |", "|---|---|---|---|"]
    by_request = {}
    for row in results:
        by_request.setdefault(row["request"], {})[row["approach"]] = row
    for request, pair in by_request.items():
        gold = pair["clef"]["gold"]
        expected = "no tool" if gold is None else f"`{gold[0]}({', '.join(f'{k}={v}' for k, v in gold[1].items())})`"
        cells = []
        for name in ("clef", "agent"):
            row = pair[name]
            mark = "✅" if row["correct_runs"] == row["repeats"] else ("❌" if row["correct_runs"] == 0 else "⚠️")
            cells.append(f"{mark} {row['median_s']:.2f} s")
        lines.append(f"| {request} | {expected} | {cells[0]} | {cells[1]} |")
    (out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--out", default="results/workflow")
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    (out / "graph_clef.mmd").write_text(clef_graph.get_graph().draw_mermaid(), encoding="utf-8")
    (out / "graph_agent.mmd").write_text(agent.get_graph().draw_mermaid(), encoding="utf-8")

    results = bench("clef", run_clef, args.repeats) + bench("agent", run_agent, args.repeats)
    summaries = {name: summarize([r for r in results if r["approach"] == name]) for name in ("clef", "agent")}
    (out / "results.json").write_text(json.dumps({"summary": summaries, "results": results},
                                                 ensure_ascii=False, indent=1), encoding="utf-8")
    write_report(out, results, summaries, args.repeats)
    print("\n=== SUMMARY ===")
    for name, s in summaries.items():
        print(name, s)
