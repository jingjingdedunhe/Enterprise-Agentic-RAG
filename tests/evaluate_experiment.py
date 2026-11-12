"""阶段 4 自动化评估：三种检索方案 + LLM-as-a-Judge。

- 从 354 条 Velvera 问题中固定随机种子抽取 20 条；
- A：基础向量 Top-5；B：混合检索 + 重排 Top-5；C：Agentic RAG 最终 chunks；
- 指标：Recall@5、MRR、平均检索/工作流耗时；
- 对前 5 题用 Agentic RAG 的答案做 1–5 分忠实度评审。

用法（项目根目录）：
    C:\\RAG\\P1\\Prag\\Scripts\\python.exe tests\\evaluate_experiment.py
"""

from __future__ import annotations

import json
import random
import re
import sys
import time
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import pandas as pd  # noqa: E402

from backend.agent.workflow import invoke_workflow  # noqa: E402
from backend.generation.llm_client import LLMClient  # noqa: E402
from backend.retrieval.bm25_retriever import rebuild_index  # noqa: E402
from backend.retrieval.reranker import get_reranker  # noqa: E402
from backend.services.rag_service import RAGService  # noqa: E402
from config.settings import settings  # noqa: E402

CSV_PATH = BASE_DIR / "data" / "Dataset categories - queries.csv"
CATEGORY = "Velvera Technologies"
SEED = 42
SAMPLE_SIZE = 20
TOP_K = 5
JUDGE_COUNT = 5

JUDGE_SYSTEM_PROMPT = """你是严格的企业 RAG 答案评审。请根据【标准事实】判断【Agent 答案】是否忠实、准确、完整地回答了【用户问题】。

评分标准（1-5）：
5 = 完全正确，关键事实完整，无幻觉；
4 = 基本正确，缺少少量细节；
3 = 部分正确，有明显遗漏或轻微不准确；
2 = 大部分错误或不完整；
1 = 完全错误、存在幻觉，或本可回答却拒答。

只输出一个 JSON 对象，不要 Markdown 代码块，不要额外解释：
{"score": 1, "reason": "简要中文理由"}
score 必须是 1 到 5 的整数。"""


def load_queries() -> list[dict]:
    """读取全部 Velvera 问题，解析标准来源与 Supporting Facts 文本。"""
    df = pd.read_csv(CSV_PATH, encoding="gb18030")
    df.columns = [str(column).strip() for column in df.columns]
    mask = (
        df["Enterprise Name"].astype(str).str.strip().str.casefold()
        == CATEGORY.casefold()
    )
    df = df.loc[mask].reset_index(drop=True)

    rows: list[dict] = []
    for index, row in df.iterrows():
        raw_facts = row.get("Supporting Facts")
        facts: list[dict] = []
        if isinstance(raw_facts, str) and raw_facts.strip():
            try:
                parsed = json.loads(raw_facts)
                if isinstance(parsed, list):
                    facts = [
                        fact for fact in parsed if isinstance(fact, dict)
                    ]
            except json.JSONDecodeError:
                facts = []

        expected = {
            Path(str(fact.get("filename", ""))).with_suffix(".pdf").name
            for fact in facts
            if fact.get("filename")
        }
        facts_text = "\n".join(
            f"[{i}] {str(fact.get('text', '')).strip()}"
            for i, fact in enumerate(facts, start=1)
        )

        rows.append(
            {
                "index": int(index),
                "query": str(row["Query"]).strip(),
                "expected": expected,
                "facts_text": facts_text,
            }
        )

    return rows


def first_correct_rank(hits: list[dict], expected: set[str]) -> int | None:
    for rank, hit in enumerate(hits, start=1):
        source = (hit.get("metadata") or {}).get("source")
        if source in expected:
            return rank
    return None


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def parse_judge_json(raw: str) -> dict:
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\s*", "", text)
        text = re.sub(r"\s*```$", "", text)

    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError("Judge 返回中没有 JSON 对象")

    data = json.loads(text[start : end + 1])
    if not isinstance(data, dict):
        raise ValueError("Judge JSON 顶层不是对象")
    return data


def judge_answer(
    client: LLMClient,
    question: str,
    facts_text: str,
    answer: str,
) -> tuple[int | None, str]:
    """调用 LLM 对答案打 1–5 分。"""
    if not answer.strip():
        return 1, "Agent 未返回答案"

    user_content = (
        f"【用户问题】\n{question}\n\n"
        f"【标准事实】\n{facts_text[:3000]}\n\n"
        f"【Agent 答案】\n{answer[:3000]}\n\n"
        "请按要求输出 JSON 评分。"
    )

    raw = client.chat(
        [
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        temperature=0.0,
        max_tokens=256,
    )

    data = parse_judge_json(raw)
    score = int(round(float(data.get("score"))))
    if not 1 <= score <= 5:
        raise ValueError(f"score 越界: {score}")
    return score, str(data.get("reason", "")).strip()


def format_ms(value: float) -> str:
    return f"{value:.1f} ms"


def main() -> int:
    rows = load_queries()
    sample = random.Random(SEED).sample(rows, SAMPLE_SIZE)

    print("=" * 100)
    print(f"阶段 4 评估 | 题库 {len(rows)} 条 | 随机种子 {SEED} | 抽样 {len(sample)} 条")
    print(f"LLM: {settings.LLM_MODEL}")
    print("抽样明细:")
    for i, row in enumerate(sample, start=1):
        print(f"  {i:2d}. idx={row['index']:3d} expected={sorted(row['expected'])} | {row['query'][:70]}")

    # 预热检索链路
    started = time.perf_counter()
    rebuild_index()
    rag = RAGService()
    rag.retrieve("warmup", top_k=1)
    get_reranker()
    print(f"检索链路预热: {time.perf_counter() - started:.1f} s（不计入指标）")
    print("=" * 100)

    schemes = {
        "A": {"name": "A 基础向量", "hits": [], "rr": [], "times": [], "rows": []},
        "B": {"name": "B 混合+重排", "hits": [], "rr": [], "times": [], "rows": []},
        "C": {"name": "C Agentic RAG", "hits": [], "rr": [], "times": [], "rows": []},
    }
    c_answers: list[dict] = []

    for order, row in enumerate(sample, start=1):
        question = row["query"]
        expected = row["expected"]
        print(f"\n[{order:2d}/{len(sample)}] idx={row['index']} expected={sorted(expected)}")
        print(f"  Q: {question}")

        # A: 基础向量
        started = time.perf_counter()
        try:
            hits_a = rag.retrieve(question, top_k=TOP_K)
        except Exception as exc:
            hits_a = []
            print(f"  A 检索失败: {type(exc).__name__}: {exc}")
        time_a = (time.perf_counter() - started) * 1000
        rank_a = first_correct_rank(hits_a, expected)
        schemes["A"]["hits"].append(1.0 if rank_a else 0.0)
        schemes["A"]["rr"].append(1.0 / rank_a if rank_a else 0.0)
        schemes["A"]["times"].append(time_a)
        schemes["A"]["rows"].append(rank_a)
        print(f"  A 基础向量: {'hit@' + str(rank_a) if rank_a else 'miss'}"
              f" | {format_ms(time_a)}")

        # B: 混合 + 重排
        started = time.perf_counter()
        try:
            hits_b = rag.retrieve_hybrid_rerank(question, top_k=TOP_K)
        except Exception as exc:
            hits_b = []
            print(f"  B 检索失败: {type(exc).__name__}: {exc}")
        time_b = (time.perf_counter() - started) * 1000
        rank_b = first_correct_rank(hits_b, expected)
        schemes["B"]["hits"].append(1.0 if rank_b else 0.0)
        schemes["B"]["rr"].append(1.0 / rank_b if rank_b else 0.0)
        schemes["B"]["times"].append(time_b)
        schemes["B"]["rows"].append(rank_b)
        print(f"  B 混合+重排: {'hit@' + str(rank_b) if rank_b else 'miss'}"
              f" | {format_ms(time_b)}")

        # C: Agentic RAG（含改写、判断、重试、生成，耗时为端到端）
        started = time.perf_counter()
        state: dict = {}
        try:
            state = invoke_workflow(question)
            hits_c = state.get("chunks") or []
            answer_c = str(state.get("answer") or "")
        except Exception as exc:
            hits_c = []
            answer_c = ""
            print(f"  C 工作流失败: {type(exc).__name__}: {exc}")
        time_c = (time.perf_counter() - started) * 1000
        rank_c = first_correct_rank(hits_c, expected)
        schemes["C"]["hits"].append(1.0 if rank_c else 0.0)
        schemes["C"]["rr"].append(1.0 / rank_c if rank_c else 0.0)
        schemes["C"]["times"].append(time_c)
        schemes["C"]["rows"].append(rank_c)
        c_answers.append(
            {
                "question": question,
                "facts_text": row["facts_text"],
                "answer": answer_c,
                "expected": expected,
                "retry_count": int(state.get("retry_count", 0)),
                "is_enough": bool(state.get("is_enough", False)),
            }
        )
        print(
            f"  C Agentic RAG: {'hit@' + str(rank_c) if rank_c else 'miss'}"
            f" | {time_c / 1000:.1f} s"
            f" | enough={state.get('is_enough')} retry={state.get('retry_count')}"
        )

    # ---------- 三种方案汇总 ----------
    print("\n" + "=" * 100)
    print("检索指标对比")
    print(f"{'方案':<18}{'Recall@5':>12}{'MRR':>10}{'平均耗时':>16}")
    for key in ("A", "B", "C"):
        item = schemes[key]
        recall = mean(item["hits"])
        mrr = mean(item["rr"])
        avg_ms = mean(item["times"])
        if key == "C":
            time_text = f"{avg_ms / 1000:.2f} s"
        else:
            time_text = f"{avg_ms:.1f} ms"
        print(f"{item['name']:<18}{recall:>12.3f}{mrr:>10.3f}{time_text:>16}")

    print("\n说明: A/B 为纯检索耗时；C 为包含改写、证据判断、重试和生成的端到端工作流耗时。")

    # ---------- LLM-as-a-Judge（复用前 5 条 C 的答案） ----------
    print("\n" + "=" * 100)
    print(f"LLM-as-a-Judge：抽样前 {JUDGE_COUNT} 题（复用 Agentic RAG 答案）")
    judge_client = LLMClient()
    judge_results: list[tuple[int | None, str, dict]] = []

    for number, item in enumerate(c_answers[:JUDGE_COUNT], start=1):
        try:
            score, reason = judge_answer(
                judge_client,
                item["question"],
                item["facts_text"],
                item["answer"],
            )
        except Exception as exc:
            score, reason = None, f"Judge 失败: {type(exc).__name__}: {exc}"

        judge_results.append((score, reason, item))
        answer_preview = " ".join(item["answer"].split())[:160]
        print(f"\n[{number}] score={score} | expected={sorted(item['expected'])}")
        print(f"  Q: {item['question']}")
        print(f"  A: {answer_preview}{'...' if len(item['answer']) > 160 else ''}")
        print(f"  reason: {reason}")

    valid_scores = [score for score, _, _ in judge_results if score is not None]
    print("\n" + "-" * 100)
    if valid_scores:
        print(
            f"Judge 得分: {valid_scores} | 平均 {mean([float(s) for s in valid_scores]):.2f}/5"
        )
    else:
        print("Judge 没有产生有效分数。")

    print("\n评估完成。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
