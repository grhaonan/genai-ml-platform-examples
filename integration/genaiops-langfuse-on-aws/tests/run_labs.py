#!/usr/bin/env python3
"""End-to-end runner for the GenAIOps Langfuse labs (maintainer test tool).

Runs the lab notebooks headless with papermill inside the Workshop Studio VSCode
environment, the same way a participant would, and writes a summary of every
notebook: pass/fail, duration, and the first failing cell with its error.

Run it from the VSCode terminal, inside this directory:

    cd /genaiops-workshop/genai-ml-platform-examples/integration/genaiops-langfuse-on-aws
    /genaiops-workshop/.venv/bin/python tests/run_labs.py              # pass 1
    # ... create the lab2 Knowledge Base and do the lab5 console steps ...
    /genaiops-workshop/.venv/bin/python tests/run_labs.py --kb-id XXXXXXXXXX --only lab2,lab5

Pass 1 runs lab1, the lab2 data-prep cells (they upload the corpus the Knowledge
Base is built from), lab3.1, lab3.2 and lab4. lab2 and lab5 need console steps
first (Knowledge Base; model invocation logging + Transaction Search), so they run
in pass 2. The Guardrail ID is read from the CloudFormation stack outputs.

By default the kernels run WITHOUT AWS_REGION / AWS_DEFAULT_REGION, because Jupyter
kernels started by the IDE do not see ~/.bashrc; this exercises the region lookup in
config.get_aws_region(). Use --keep-region-env to keep them.

Results: <workshop home>/e2e-results/<timestamp>/ (executed notebooks + summary.md)
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent  # integration/genaiops-langfuse-on-aws
LABS = {
    "lab1": "lab1/lab1-langfuse-basics.ipynb",
    "lab2-prep": "lab2/lab2-rag-langfuse.ipynb",
    "lab3.1": "lab3/lab3.1-model-based-eval.ipynb",
    "lab3.2": "lab3/lab3.2-bedrock-guardrails.ipynb",
    "lab4": "lab4/lab4-langfuse-strands-mcp.ipynb",
    "lab2": "lab2/lab2-rag-langfuse.ipynb",
    "lab5": "lab5/lab5-langfuse-agentcore.ipynb",
}
PASS1 = ["lab1", "lab2-prep", "lab3.1", "lab3.2", "lab4"]
PASS2 = ["lab2", "lab5"]
KB_PLACEHOLDER = '"<TO FILL>"'
GUARDRAIL_PLACEHOLDER = '"<guardrailid>"'


def sh(cmd, **kw):
    print("+", " ".join(cmd), flush=True)
    return subprocess.run(cmd, check=True, **kw)


def ensure_papermill(py):
    try:
        subprocess.run([py, "-c", "import papermill, ipykernel"], check=True, capture_output=True)
        return
    except subprocess.CalledProcessError:
        pass
    uv = shutil.which("uv") or str(Path(py).parent / "uv")
    if Path(uv).exists():
        sh([uv, "pip", "install", "--python", py, "--quiet", "papermill", "ipykernel"])
    else:
        sh([py, "-m", "pip", "install", "--quiet", "papermill", "ipykernel"])


def guardrail_id_from_stacks():
    import boto3

    sys.path.insert(0, str(HERE))
    from config import get_aws_region

    cfn = boto3.client("cloudformation", region_name=get_aws_region())
    for page in cfn.get_paginator("describe_stacks").paginate():
        for st in page["Stacks"]:
            for out in st.get("Outputs", []):
                if out["OutputKey"].lower() == "guardrailid":
                    return out["OutputValue"]
    return None


def fill_guardrail(gid):
    cfg = HERE / "config.py"
    s = cfg.read_text()
    if GUARDRAIL_PLACEHOLDER in s:
        cfg.write_text(s.replace(GUARDRAIL_PLACEHOLDER, json.dumps(gid)))
        print(f"config.py: GUARDRAIL_CONFIG.guardrailIdentifier = {gid}")


def prepare_notebook(name, src, dst, kb_id):
    nb = json.loads(src.read_text())
    if name == "lab2-prep":
        # keep everything up to (not including) the Knowledge Base ID cell
        for i, c in enumerate(nb["cells"]):
            if c["cell_type"] == "code" and KB_PLACEHOLDER in "".join(c["source"]):
                nb["cells"] = nb["cells"][:i]
                break
    elif name == "lab2":
        for c in nb["cells"]:
            if c["cell_type"] == "code":
                s = "".join(c["source"])
                if KB_PLACEHOLDER in s:
                    c["source"] = s.replace(KB_PLACEHOLDER, json.dumps(kb_id))
    dst.write_text(json.dumps(nb, indent=1, ensure_ascii=False))


def first_error(executed):
    nb = json.loads(executed.read_text())
    for i, c in enumerate(nb["cells"]):
        if c["cell_type"] != "code":
            continue
        for o in c.get("outputs", []):
            if o.get("output_type") == "error":
                src = "".join(c["source"]).strip().splitlines()
                return {
                    "cell": i,
                    "ename": o.get("ename"),
                    "evalue": re.sub(r"\x1b\[[0-9;]*m", "", o.get("evalue", ""))[:600],
                    "source_head": "\n".join(src[:6]),
                }
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kb-id", help="lab2 Knowledge Base ID (enables lab2 in pass 2)")
    ap.add_argument("--only", help="comma-separated subset, e.g. lab2,lab5")
    ap.add_argument("--keep-region-env", action="store_true")
    ap.add_argument("--cell-timeout", type=int, default=1800)
    args = ap.parse_args()

    py = sys.executable
    venv_bin = str(Path(py).parent)
    home = Path(os.environ.get("WORKSHOP_HOME", "/genaiops-workshop"))
    if not home.exists():
        home = HERE
    out_dir = home / "e2e-results" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir.mkdir(parents=True, exist_ok=True)

    if not (HERE / ".env").exists():
        sys.exit(f"Missing {HERE / '.env'} (LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY / LANGFUSE_HOST)")
    if "LANGFUSE_BASE_URL" in (HERE / ".env").read_text() and "LANGFUSE_HOST" not in (HERE / ".env").read_text():
        sys.exit(".env uses LANGFUSE_BASE_URL; these labs expect LANGFUSE_HOST")

    ensure_papermill(py)

    selected = args.only.split(",") if args.only else (PASS1 + PASS2 if args.kb_id else PASS1)
    if "lab2" in selected and not args.kb_id:
        sys.exit("lab2 needs --kb-id (create the Knowledge Base in the console first)")

    if any(n in selected for n in ("lab3.2",)):
        gid = guardrail_id_from_stacks()
        if gid:
            fill_guardrail(gid)
        else:
            print("WARNING: no GuardrailId stack output found; lab3.2 will likely fail")

    env = dict(os.environ)
    env["PATH"] = venv_bin + os.pathsep + env.get("PATH", "")
    if not args.keep_region_env:
        env.pop("AWS_REGION", None)
        env.pop("AWS_DEFAULT_REGION", None)

    rows = []
    for name in selected:
        src = HERE / LABS[name]
        work = src.parent / f".e2e-{name}.ipynb"
        executed = out_dir / f"{name}.ipynb"
        prepare_notebook(name, src, work, args.kb_id)
        t0 = time.time()
        print(f"\n===== {name}: {src.relative_to(HERE)}", flush=True)
        r = subprocess.run(
            [py, "-m", "papermill", str(work), str(executed), "-k", "python3",
             "--cwd", str(src.parent), "--execution-timeout", str(args.cell_timeout),
             "--log-output", "--no-progress-bar"],
            env=env,
        )
        work.unlink(missing_ok=True)
        dur = time.time() - t0
        err = first_error(executed) if executed.exists() else {"ename": "PapermillError", "evalue": "no output notebook"}
        rows.append({"lab": name, "ok": r.returncode == 0 and not err, "seconds": round(dur), "error": err})

    lines = [f"# GenAIOps Langfuse labs — E2E run {out_dir.name}", ""]
    try:
        sys.path.insert(0, str(HERE))
        from config import get_aws_region
        lines.append(f"- region (config.get_aws_region): `{get_aws_region()}`")
    except Exception as e:  # noqa: BLE001
        lines.append(f"- region lookup failed: {e}")
    lines += [f"- region env vars passed to kernels: {'yes' if args.keep_region_env else 'no (IDE-like)'}", "",
              "| lab | result | duration |", "|---|---|---|"]
    lines += [f"| {r['lab']} | {'✅ pass' if r['ok'] else '❌ fail'} | {r['seconds']}s |" for r in rows]
    for r in rows:
        if r["error"]:
            e = r["error"]
            lines += ["", f"## {r['lab']} — first error (cell {e.get('cell')})", "",
                      f"`{e.get('ename')}`: {e.get('evalue')}", "", "```python", e.get("source_head", ""), "```"]
    if not args.kb_id and "lab2" not in selected:
        lines += ["", "Next: create the lab2 Knowledge Base in the console (the corpus is now in S3),",
                  "do the lab5 console steps, then re-run with --kb-id <ID> --only lab2,lab5."]
    summary = "\n".join(lines) + "\n"
    (out_dir / "summary.md").write_text(summary)
    print("\n" + summary)
    print(f"Executed notebooks and summary: {out_dir}")
    sys.exit(0 if all(r["ok"] for r in rows) else 1)


if __name__ == "__main__":
    main()
