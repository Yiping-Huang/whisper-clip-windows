import os
import shlex
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def _split_command(raw_command: str) -> list[str]:
    # Windows needs non-POSIX parsing so backslashes in paths are preserved.
    parts = shlex.split(raw_command, posix=(os.name != "nt"))
    cleaned: list[str] = []
    for part in parts:
        if len(part) >= 2 and part[0] == part[-1] and part[0] in ("'", '"'):
            cleaned.append(part[1:-1])
        else:
            cleaned.append(part)
    return cleaned


def _resolve_codex_base_command(raw_command: str) -> list[str]:
    parts = _split_command(raw_command)
    if not parts:
        raise RuntimeError("CODEX_CMD is empty. Set it to a valid Codex command.")

    exe = parts[0]
    if Path(exe).is_file():
        return parts

    resolved = shutil.which(exe)
    if resolved:
        parts[0] = resolved
        return parts

    # Windows fallback: Node CLIs are often installed as .cmd/.exe shims.
    for candidate in (f"{exe}.cmd", f"{exe}.exe", f"{exe}.bat"):
        resolved = shutil.which(candidate)
        if resolved:
            parts[0] = resolved
            return parts

    raise RuntimeError(
        f"Codex CLI executable not found for {exe!r}. "
        "Set CODEX_CMD to a full command, for example: "
        "CODEX_CMD='C:\\Users\\<you>\\AppData\\Roaming\\npm\\codex.cmd'"
    )


DEFAULT_SYSTEM_INSTRUCTION = (
    "Answer the user's query directly and concisely. "
    "Do not ask follow-up questions unless the user explicitly requests them. "
    "If uncertain, state the uncertainty briefly."
)


def _build_prompt(user_query: str, system_instruction: str) -> str:
    return (
        "System instruction:\n"
        f"{system_instruction.strip()}\n\n"
        "User query:\n"
        f"{user_query.strip()}\n\n"
        "Now return only the final answer to the user query."
    )


def ask_codex(
    user_query: str,
    system_instruction: str = DEFAULT_SYSTEM_INSTRUCTION,
    codex_command: str | None = None,
    timeout_seconds: int = 90,
    cwd: str | None = None,
) -> str:
    """Send a query to Codex CLI and return the final answer text."""
    if not user_query.strip():
        raise ValueError("user_query cannot be empty.")

    raw_command = codex_command or os.environ.get("CODEX_CMD", "codex")
    base_command = _resolve_codex_base_command(raw_command)
    prompt = _build_prompt(user_query=user_query, system_instruction=system_instruction)
    single_line_prompt = " ".join(prompt.splitlines())

    fd, output_path = tempfile.mkstemp(prefix="codex_last_message_", suffix=".txt")
    os.close(fd)

    attempts: list[tuple[list[str], str | None]] = [
        (
            base_command + ["exec", "--skip-git-repo-check", "--output-last-message", output_path, "-"],
            prompt + "\n",
        ),
        (
            base_command
            + [
                "exec",
                "--skip-git-repo-check",
                "--output-last-message",
                output_path,
                single_line_prompt,
            ],
            None,
        ),
    ]

    last_error: Exception | None = None

    try:
        for command, stdin_text in attempts:
            try:
                result = subprocess.run(
                    command,
                    input=stdin_text,
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    timeout=timeout_seconds,
                    cwd=cwd,
                    check=False,
                )
            except Exception as exc:  # pragma: no cover - fallback path
                last_error = exc
                continue

            file_output = Path(output_path).read_text(encoding="utf-8", errors="replace").strip()
            output = (result.stdout or "").strip()
            err = (result.stderr or "").strip()

            if result.returncode == 0 and file_output:
                return file_output
            if result.returncode == 0 and output:
                return output
            if result.returncode == 0 and err:
                return err

            last_error = RuntimeError(
                f"Command failed ({' '.join(command)}), return code {result.returncode}, stderr: {err or '<empty>'}"
            )
    finally:
        Path(output_path).unlink(missing_ok=True)

    raise RuntimeError(f"All Codex invocation attempts failed. Last error: {last_error}")


def run_codex_smoke_test(
    prompt: str = "What is your name? Reply in one short sentence only.",
    codex_command: str | None = None,
    timeout_seconds: int = 90,
    cwd: str | None = None,
) -> tuple[str, str]:
    """Backward-compatible smoke test helper."""
    response = ask_codex(
        user_query=prompt,
        codex_command=codex_command,
        timeout_seconds=timeout_seconds,
        cwd=cwd,
    )
    return response, "codex exec --skip-git-repo-check"
