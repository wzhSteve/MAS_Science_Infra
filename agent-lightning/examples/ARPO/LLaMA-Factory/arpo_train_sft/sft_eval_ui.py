#!/usr/bin/env python3
"""ARPO Qwen3-1.7B-SFT 格式烟测界面。

用法：
  bash .../arpo_train_sft/sft_eval_ui.sh
  # 或
  /root/autodl-tmp/envs/arpo_sft/bin/python sft_eval_ui.py --port 8789
"""

from __future__ import annotations

import argparse
import html
import re
import threading
import time
from pathlib import Path

import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def _find_repo_root() -> Path:
    cur = Path(__file__).resolve().parent
    for _ in range(10):
        if (cur / "LLM").is_dir() and (cur / "agent-lightning").is_dir():
            return cur
        if cur.parent == cur:
            break
        cur = cur.parent
    raise RuntimeError("找不到仓库根（需要同时存在 LLM/ 与 agent-lightning/）")


REPO_ROOT = _find_repo_root()
DEFAULT_MODEL = REPO_ROOT / "LLM" / "ARPO" / "Qwen3-1.7B-SFT"

ARPO_SYSTEM = (
    "You are a helpful assistant that can solve the given question step by step with the help of the "
    "wikipedia search tool and python interpreter tool. Given a question, you need to first think about "
    "the reasoning process in the mind and then provide the answer. During thinking, you can invoke the "
    "wikipedia search tool to search and python interpreter tool to calculate the math problem for fact "
    "information about specific topics if needed. The reasoning process and answer are enclosed within "
    "<think> </think> and <answer> </answer> tags respectively, and the search query and result are "
    "enclosed within <search> </search> and <result> </result> tags respectively. For example, "
    "<think> This is the reasoning process. </think> <search> search query here </search> "
    "<result> search result here </result> <think> This is the reasoning process. </think> "
    "<python> python code here </python> <result> python interpreter result here </result> "
    "<think> This is the reasoning process. </think> "
    "<answer> The final answer is \\[ \\boxed{answer here} \\] </answer>. "
    "In the last part of the answer, the final exact answer is enclosed within \\boxed{} with latex format."
)

EXAMPLES = [
    "What is 17 * 19? Show the calculation.",
    "If a circle has radius 7, what is its area? Use pi=3.1416.",
    "Who invented the World Wide Web and in which year?",
    "Marine annelids typically have a larval stage during their development. What is it called?",
    "The matrix [[a, 3], [-8, d]] is its own inverse. How many possible pairs (a, d)?",
]

TAG_SPECS = [
    ("think", r"<think>", r"</think>", "推理块"),
    ("search", r"<search>", r"</search>", "维基搜索"),
    ("python", r"<python>", r"</python>", "Python 工具"),
    ("result", r"<result>", r"</result>", "工具结果"),
    ("answer", r"<answer>", r"</answer>", "最终答案"),
    ("boxed", r"\\boxed\{", r"\}", r"\boxed{}"),
]


class ModelBundle:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.model = None
        self.tok = None
        self.path: str | None = None
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.load_secs: float | None = None
        self.error: str | None = None

    def ensure(self, model_path: str) -> str:
        with self.lock:
            if self.model is not None and self.path == model_path:
                return f"已加载 · {self.path} · {self.device} · {self.load_secs:.1f}s"
            self.error = None
            t0 = time.time()
            try:
                self.tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
                dtype = torch.bfloat16 if self.device.startswith("cuda") else torch.float32
                kwargs = {"torch_dtype": dtype, "trust_remote_code": True}
                if self.device.startswith("cuda"):
                    kwargs["device_map"] = self.device
                self.model = AutoModelForCausalLM.from_pretrained(model_path, **kwargs)
                if not self.device.startswith("cuda"):
                    self.model = self.model.to(self.device)
                self.model.eval()
                self.path = model_path
                self.load_secs = time.time() - t0
                return f"已加载 · {self.path} · {self.device} · {self.load_secs:.1f}s"
            except Exception as exc:  # noqa: BLE001
                self.model = None
                self.tok = None
                self.path = None
                self.error = str(exc)
                raise


BUNDLE = ModelBundle()


def analyze(text: str) -> dict:
    scores = {}
    for key, open_t, close_t, _ in TAG_SPECS:
        if key == "boxed":
            scores[key] = bool(re.search(r"\\boxed\s*\{", text))
        else:
            has_open = open_t in text
            has_close = close_t in text
            scores[key] = has_open and has_close
            scores[f"{key}_open"] = has_open
            scores[f"{key}_close"] = has_close
    scores["len"] = len(text)
    # 训练预期：至少有 think；工具题常有 search/python；收敛时有 answer+boxed
    expected_core = scores["think"]
    tool_or_answer = scores["search"] or scores["python"] or scores["answer"]
    scores["pass_core"] = expected_core and tool_or_answer
    scores["pass_strong"] = scores["think"] and scores["answer"] and scores["boxed"]
    return scores


def score_html(scores: dict) -> str:
    chips = []
    for key, _, _, label in TAG_SPECS:
        ok = scores.get(key, False)
        color = "#1b7f4e" if ok else "#9a3412"
        bg = "#dcfce7" if ok else "#ffedd5"
        mark = "✓" if ok else "✗"
        chips.append(
            f'<span style="display:inline-block;margin:4px 6px 4px 0;padding:4px 10px;'
            f'border-radius:999px;background:{bg};color:{color};font:600 13px/1.4 ui-sans-serif,system-ui">'
            f"{mark} {html.escape(label)}</span>"
        )
    if scores.get("pass_strong"):
        verdict = ('<div style="margin-top:10px;font:700 15px ui-sans-serif;color:#166534">'
                   "判定：强匹配（think + answer + boxed）</div>")
    elif scores.get("pass_core"):
        verdict = ('<div style="margin-top:10px;font:700 15px ui-sans-serif;color:#a16207">'
                   "判定：部分匹配（有 think，且出现工具或 answer）</div>")
    else:
        verdict = ('<div style="margin-top:10px;font:700 15px ui-sans-serif;color:#b91c1c">'
                   "判定：未按训练格式（缺少 think / 工具 / answer）</div>")
    meta = f'<div style="margin-top:6px;color:#64748b;font:13px ui-sans-serif">输出长度 {scores["len"]} 字符</div>'
    return "".join(chips) + verdict + meta


def highlight(text: str) -> str:
    esc = html.escape(text)
    colors = {
        "think": ("#1d4ed8", "#dbeafe"),
        "search": ("#7c2d12", "#ffedd5"),
        "python": ("#065f46", "#d1fae5"),
        "result": ("#6b21a8", "#f3e8ff"),
        "answer": ("#9f1239", "#ffe4e6"),
    }

    def wrap(tag: str, body: str) -> str:
        fg, bg = colors[tag]
        return (
            f'<span style="background:{bg};color:{fg};border-radius:4px;padding:0 2px">'
            f"&lt;{tag}&gt;</span>"
            f'<span style="background:{bg}">{body}</span>'
            f'<span style="background:{bg};color:{fg};border-radius:4px;padding:0 2px">'
            f"&lt;/{tag}&gt;</span>"
        )

    for tag in ("think", "search", "python", "result", "answer"):
        pat = re.compile(rf"&lt;{tag}&gt;(.*?)&lt;/{tag}&gt;", re.DOTALL | re.IGNORECASE)

        def repl(m: re.Match, t=tag) -> str:
            return wrap(t, m.group(1))

        esc = pat.sub(repl, esc)

    esc = re.sub(
        r"\\boxed\s*\{([^{}]*)\}",
        r'<span style="background:#fef08a;padding:0 4px;border-radius:4px;font-weight:700">'
        r"\\boxed{\1}</span>",
        esc,
    )
    return (
        '<pre style="white-space:pre-wrap;word-break:break-word;margin:0;'
        'font:13px/1.55 ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;'
        f'background:#f8fafc;border:1px solid #e2e8f0;border-radius:10px;padding:14px">{esc}</pre>'
    )


def generate(
    model_path: str,
    system: str,
    question: str,
    max_new_tokens: int,
    temperature: float,
    do_sample: bool,
) -> tuple[str, str, str, str]:
    question = (question or "").strip()
    if not question:
        return "请先输入问题。", "", "", score_html(analyze(""))

    status = BUNDLE.ensure(model_path.strip() or str(DEFAULT_MODEL))
    messages = [
        {"role": "system", "content": system.strip() or ARPO_SYSTEM},
        {"role": "user", "content": question},
    ]
    prompt = BUNDLE.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = BUNDLE.tok(prompt, return_tensors="pt")
    inputs = {k: v.to(BUNDLE.model.device) for k, v in inputs.items()}

    gen_kwargs = {
        "max_new_tokens": int(max_new_tokens),
        "pad_token_id": BUNDLE.tok.eos_token_id,
    }
    if do_sample and temperature > 0:
        gen_kwargs.update(do_sample=True, temperature=float(temperature), top_p=0.9)
    else:
        gen_kwargs.update(do_sample=False)

    t0 = time.time()
    with torch.no_grad():
        out = BUNDLE.model.generate(**inputs, **gen_kwargs)
    secs = time.time() - t0
    new_tokens = out[0][inputs["input_ids"].shape[1] :]
    text = BUNDLE.tok.decode(new_tokens, skip_special_tokens=True)
    scores = analyze(text)
    meta = f"{status} · 生成 {secs:.1f}s · {len(new_tokens)} tokens"
    return meta, text, highlight(text), score_html(scores)


CSS = """
.gradio-container { max-width: 1100px !important; }
footer { display: none !important; }
"""


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="ARPO SFT 1.7B 格式测试", css=CSS, theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            "## ARPO · Qwen3-1.7B-SFT 格式测试\n"
            "对照冷启动 SFT 的训练约定：输出应含 "
            "`<think>` / 可选 `<search>`·`<python>` / `<answer>` + `\\\\boxed{}`。"
        )
        with gr.Row():
            model_path = gr.Textbox(
                label="模型路径",
                value=str(DEFAULT_MODEL),
                scale=4,
            )
            status = gr.Textbox(label="状态", value="尚未加载（点生成时自动加载）", interactive=False, scale=3)

        with gr.Row():
            with gr.Column(scale=1):
                system = gr.Textbox(label="System（训练同款）", value=ARPO_SYSTEM, lines=8)
                question = gr.Textbox(
                    label="问题",
                    value=EXAMPLES[0],
                    lines=4,
                    placeholder="输入要测的问题…",
                )
                with gr.Row():
                    max_new = gr.Slider(64, 1024, value=384, step=32, label="max_new_tokens")
                    temperature = gr.Slider(0.0, 1.5, value=0.0, step=0.05, label="temperature")
                do_sample = gr.Checkbox(label="采样（关闭则贪心，推荐测格式）", value=False)
                with gr.Row():
                    btn = gr.Button("生成并检查格式", variant="primary")
                    clear = gr.Button("清空输出")
                gr.Examples(examples=[[q] for q in EXAMPLES], inputs=[question], label="示例问题")

            with gr.Column(scale=1):
                score = gr.HTML(label="格式检查")
                highlighted = gr.HTML(label="高亮输出")
                raw = gr.Textbox(label="原始输出", lines=16)

        def _run(mp, sys, q, mn, temp, sample):
            try:
                return generate(mp, sys, q, mn, temp, sample)
            except Exception as exc:  # noqa: BLE001
                err = f"失败：{exc}"
                return err, "", f"<pre>{html.escape(err)}</pre>", score_html(analyze(""))

        btn.click(
            _run,
            inputs=[model_path, system, question, max_new, temperature, do_sample],
            outputs=[status, raw, highlighted, score],
        )
        clear.click(lambda: ("", "", score_html(analyze(""))), outputs=[raw, highlighted, score])
    return demo


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=str(DEFAULT_MODEL))
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8789)
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--preload", action="store_true", help="启动时预加载模型")
    args = parser.parse_args()

    if args.preload:
        print(BUNDLE.ensure(args.model), flush=True)

    demo = build_ui()
    print(f"打开 http://127.0.0.1:{args.port}  （远程可 ssh -L {args.port}:127.0.0.1:{args.port}）", flush=True)
    demo.launch(server_name=args.host, server_port=args.port, share=args.share, show_error=True)


if __name__ == "__main__":
    main()
