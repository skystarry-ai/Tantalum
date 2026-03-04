"""
Tantalum CLI
Copyright 2026 Skystarry-AI
SPDX-License-Identifier: Apache-2.0

This module provides the Command Line Interface (CLI) for Tantalum.
It manages the instantiation of the local Gateway and Brain processes,
handles secure TCP connections (via HMAC-SHA256), and provides a rich
terminal UI using prompt_toolkit and Rich.

Slash commands are loaded at startup from commands.toml (located in the
same directory as this file). Adding a new [[command]] block to that file
is sufficient to register it in the autocomplete menu; the handler must
also be wired into dispatch_slash_command() inside this file.
"""

import atexit
import hashlib
import hmac
import json
import os
import secrets
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import tomllib
from importlib.resources import files as _res_files

from dotenv import load_dotenv
from prompt_toolkit import PromptSession
from prompt_toolkit.completion import Completer, Completion
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.styles import Style
from rich import box
from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.spinner import Spinner

# Check for Linux-based OS compatibility
if sys.platform != "linux":
    print("Error: Tantalum is designed for Linux-based systems (including WSL2).")
    print("Windows or MacOS native execution is not supported due to security architecture.")
    sys.exit(1)


# --- Configuration & Constants ---
GATEWAY_HOST = "127.0.0.1"
GATEWAY_PORT = 8080
PAYLOAD = b"hello-tantalum"

# Generate a 32-byte (64 char) random hex key for each run (Zero-trust architecture)
SESSION_SECRET = secrets.token_hex(32)

ROOT_DIR = Path(__file__).parent.parent

# Configuration Path Definitions
CONFIG_DIR = Path.home() / ".config" / "tantalum"
CONFIG_FILE = CONFIG_DIR / "config.env"

# Resolve commands.toml via importlib.resources so the path is correct both
# when running from source and after pip install.
COMMANDS_TOML = Path(str(_res_files("frontend").joinpath("commands.toml")))

# Binary Path Definitions
GATEWAY_BIN_DEV = ROOT_DIR / "backend" / "zig-out" / "bin" / "tantalum-gateway"
GATEWAY_BIN_REL = ROOT_DIR / "service" / "tantalum-gateway"

BRAIN_SCRIPT = ROOT_DIR / "frontend" / "__main__.py"
BRAIN_SOCKET_PATH = "/tmp/tantalum-brain.sock"

console = Console()
_processes: list[subprocess.Popen] = []


# ---------------------------------------------------------------------------
# Slash Command Data Model
# ---------------------------------------------------------------------------

@dataclass
class CommandEntry:
    """
    Represents a single slash command loaded from commands.toml.

    Attributes:
        name:        Primary command string including the leading slash
                     (e.g. "/exit").
        description: Short description rendered as dim meta text in the
                     autocomplete dropdown.
        aliases:     When this entry is itself an alias, this list holds the
                     single primary command name it delegates to
                     (e.g. ["/exit"] for "/quit").
                     When empty, this entry is a primary command.
    """

    name: str
    description: str
    aliases: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Slash Command Loader
# ---------------------------------------------------------------------------

def load_commands(toml_path: Path) -> list[CommandEntry]:
    """
    Parse commands.toml and return an ordered list of CommandEntry objects.

    Expected TOML structure::

        [[command]]
        name        = "/exit"
        description = "Exit Tantalum"

        [[command]]
        name        = "/quit"
        description = "Exit Tantalum (alias for /exit)"
        aliases     = ["/exit"]

    If the file is missing or malformed a warning is printed and an empty
    list is returned so the CLI degrades gracefully (no autocomplete).

    Args:
        toml_path: Absolute path to commands.toml.

    Returns:
        List of CommandEntry objects in file-definition order.
    """
    if not toml_path.exists():
        console.print(
            f"[yellow]warning[/yellow]  {toml_path} not found — "
            "slash-command autocomplete disabled."
        )
        return []

    try:
        with open(toml_path, "rb") as fh:
            data = tomllib.load(fh)
    except Exception as exc:
        console.print(
            f"[yellow]warning[/yellow]  Failed to parse {toml_path}: {exc} — "
            "slash-command autocomplete disabled."
        )
        return []

    entries: list[CommandEntry] = []
    for raw in data.get("command", []):
        name = raw.get("name", "").strip()
        description = raw.get("description", "").strip()
        aliases: list[str] = raw.get("aliases", [])

        if not name or not description:
            console.print(
                f"[yellow]warning[/yellow]  Skipping malformed command entry: {raw}"
            )
            continue

        entries.append(CommandEntry(name=name, description=description, aliases=aliases))

    return entries


def build_command_index(
    entries: list[CommandEntry],
) -> tuple[list[tuple[str, str]], set[str], dict[str, str]]:
    """
    Derive display data, the validity set, and the alias resolution map from
    a list of CommandEntry objects.

    Args:
        entries: Output of load_commands().

    Returns:
        display_pairs:  Ordered list of (command_name, description) tuples
                        used to populate the autocomplete dropdown.
        valid_set:      Flat set of every valid command string (names and
                        aliases) for O(1) lookup in the main loop.
        canonical_map:  Maps every command string to the canonical handler
                        key.  Primary commands map to themselves; alias
                        entries map to the first element of their aliases
                        list (the primary command they delegate to).
    """
    display_pairs: list[tuple[str, str]] = []
    valid_set: set[str] = set()
    canonical_map: dict[str, str] = {}

    for entry in entries:
        display_pairs.append((entry.name, entry.description))
        valid_set.add(entry.name)

        if entry.aliases:
            # This entry is an alias; resolve it to the primary command.
            canonical_map[entry.name] = entry.aliases[0]
        else:
            # Primary command; resolves to itself.
            canonical_map[entry.name] = entry.name

    return display_pairs, valid_set, canonical_map


# ---------------------------------------------------------------------------
# Slash Command Completer
# ---------------------------------------------------------------------------

class SlashCommandCompleter(Completer):
    """
    prompt_toolkit Completer that activates only when the input starts with '/'.

    The command list is injected at construction time from TOML-loaded data
    so the completer stays decoupled from the loading and dispatch logic.

    Attributes:
        _pairs: Ordered list of (command_name, description) tuples used to
                build Completion objects.
    """

    def __init__(self, display_pairs: list[tuple[str, str]]) -> None:
        super().__init__()
        self._pairs = display_pairs

    def get_completions(self, document, complete_event):
        text = document.text_before_cursor

        # Only activate completer while the buffer starts with a slash
        if not text.startswith("/"):
            return

        typed = text.strip()

        for cmd, description in self._pairs:
            if cmd.startswith(typed):
                # display_meta appears as dim text to the right of the command
                yield Completion(
                    cmd,
                    start_position=-len(typed),
                    display=cmd,
                    display_meta=description,
                )


# ---------------------------------------------------------------------------
# Process Management
# ---------------------------------------------------------------------------

def start_process(name: str, cmd: list[str], env: dict | None = None) -> subprocess.Popen:
    """
    Start a background subprocess and stream its output to a log file.

    Args:
        name: Name of the process (used for log filename).
        cmd:  Command and arguments to execute.
        env:  Optional extra environment variables (merged with os.environ).

    Returns:
        The Popen instance of the spawned process.
    """
    proc = subprocess.Popen(
        cmd,
        env={**os.environ, **(env or {})},
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    _processes.append(proc)

    log_path = f"/tmp/tantalum-{name}.log"
    log_file = open(log_path, "w", encoding="utf-8")

    def _log() -> None:
        if proc.stdout:
            for line in proc.stdout:
                log_file.write(line)
                log_file.flush()

    threading.Thread(target=_log, daemon=True).start()
    return proc


def cleanup() -> None:
    """Terminate and kill all spawned background processes."""
    for proc in _processes:
        try:
            proc.terminate()
            proc.wait(timeout=3)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass


def wait_for_gateway(timeout: float = 10.0) -> bool:
    """
    Wait until the Gateway opens the target TCP port.
    Returns True if successful, False if timeout reached.
    """
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            s = socket.create_connection((GATEWAY_HOST, GATEWAY_PORT), timeout=0.5)
            s.close()
            return True
        except OSError:
            time.sleep(0.2)
    return False


def wait_for_brain(timeout: float = 15.0) -> bool:
    """
    Wait until the Brain Unix Domain Socket (UDS) is connectable.
    Returns True if successful, False if timeout reached.
    """
    deadline = time.time() + timeout
    while time.time() < deadline:
        if os.path.exists(BRAIN_SOCKET_PATH):
            try:
                s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                s.connect(BRAIN_SOCKET_PATH)
                s.close()
                return True
            except OSError:
                pass
        time.sleep(0.2)
    return False


def restart_brain_process(sock: socket.socket) -> socket.socket:
    """
    Restart the brain process to apply new environment variables
    and invalidate the KV cache entirely.
    """
    sock.close()

    global _processes
    brain_procs = [p for p in _processes if p.args and str(BRAIN_SCRIPT) in p.args]
    for p in brain_procs:
        p.terminate()
        try:
            p.wait(timeout=3)
        except subprocess.TimeoutExpired:
            p.kill()
        if p in _processes:
            _processes.remove(p)

    load_dotenv(CONFIG_FILE, override=True)
    start_process("brain", [sys.executable, str(BRAIN_SCRIPT)])

    if not wait_for_brain():
        raise RuntimeError("Brain failed to restart.")

    return connect()


# ---------------------------------------------------------------------------
# Networking & Authentication
# ---------------------------------------------------------------------------

def make_spa_packet(payload: bytes, secret: str) -> bytes:
    """Create a secure HMAC-SHA256 packet for Gateway authentication."""
    mac = hmac.new(secret.encode(), payload, hashlib.sha256).digest()
    return mac + payload


def connect() -> socket.socket:
    """Establish a socket connection to the Gateway and authenticate."""
    sock = socket.create_connection((GATEWAY_HOST, GATEWAY_PORT))
    sock.sendall(make_spa_packet(PAYLOAD, SESSION_SECRET))
    time.sleep(0.1)
    return sock



def send_message(sock: socket.socket, message: str) -> dict:
    """
    Send a message over the socket and receive the JSON response stream.

    Args:
        sock:    The connected socket instance.
        message: The user's input string.

    Returns:
        The parsed JSON dictionary of the final response.
    """
    obj = json.dumps({"role": "user", "content": message}, ensure_ascii=False)
    sock.sendall(obj.encode("utf-8") + b"\n")

    buf = b""
    while True:
        chunk = sock.recv(4096)
        if not chunk:
            break
        buf += chunk

        while b"\n" in buf:
            line, buf = buf.split(b"\n", 1)
            if not line:
                continue
            msg = json.loads(line.decode("utf-8"))
            if msg.get("done"):
                return msg
    return {}


# ---------------------------------------------------------------------------
# Terminal UI Components
# ---------------------------------------------------------------------------

def print_header() -> None:
    """Print the application header banner."""
    console.print()
    model = os.environ.get("TANTALUM_MODEL", "gemini/gemini-2.5-flash")
    console.print(Panel.fit(
        f"[bold cyan]Tantalum[/bold cyan] [dim]zero-trust AI agent[/dim]\n"
        f"[dim]Model: {model}  •  Gateway: {GATEWAY_HOST}:{GATEWAY_PORT}  •  / for commands[/dim]",
        border_style="cyan",
        box=box.ROUNDED,
    ))
    console.print()


def print_user(message: str) -> None:
    """Format and print user input."""
    console.print(f"[bold green]you[/bold green]  {message}")
    console.print()


def print_assistant(content: str | None) -> None:
    """Format and print the AI assistant's response (supports Markdown)."""
    console.print("[bold cyan]tantalum[/bold cyan]")
    if content:
        console.print(Markdown(content))
    else:
        console.print("[dim](no response)[/dim]")
    console.print()


def print_error(message: str) -> None:
    """Format and print error messages."""
    console.print(f"[bold red]error[/bold red]  {message}")
    console.print()


# ---------------------------------------------------------------------------
# Prompt Session Builder
# ---------------------------------------------------------------------------

def build_prompt_session(display_pairs: list[tuple[str, str]]) -> PromptSession:
    """
    Construct a PromptSession with slash-command autocomplete.

    - SlashCommandCompleter is initialised with the TOML-derived display_pairs.
    - complete_while_typing=True makes the dropdown appear the moment '/' is typed.
    - Tab confirms the highlighted completion; outside slash-mode it inserts spaces.

    Args:
        display_pairs: Ordered list of (command_name, description) tuples
                       produced by build_command_index().

    Returns:
        A configured PromptSession instance.
    """
    bindings = KeyBindings()

    @bindings.add("tab")
    def _tab(event) -> None:
        """Confirm the current completion when in slash-command mode."""
        buf = event.app.current_buffer
        state = buf.complete_state
        completion = state.current_completion if state else None
        if buf.text.startswith("/") and completion is not None:
            buf.apply_completion(completion)
        else:
            buf.insert_text("    ")

    style = Style.from_dict({
        "prompt":                                  "ansicyan bold",
        "completion-menu":                         "bg:#1e1e2e fg:#cdd6f4",
        "completion-menu.completion":              "bg:#1e1e2e fg:#cdd6f4",
        "completion-menu.completion.current":      "bg:#313244 fg:#89dceb bold",
        "completion-menu.meta.completion":         "bg:#1e1e2e fg:#6c7086",
        "completion-menu.meta.completion.current": "bg:#313244 fg:#6c7086",
    })

    return PromptSession(
        completer=SlashCommandCompleter(display_pairs),
        complete_while_typing=True,
        style=style,
        key_bindings=bindings,
    )


# ---------------------------------------------------------------------------
# Command Dispatcher
# ---------------------------------------------------------------------------

def dispatch_slash_command(
    canonical_cmd: str,
    sock: socket.socket,
) -> tuple[bool, socket.socket | None]:
    """
    Execute the handler for a recognised slash command.

    Handlers that need to replace the active socket (e.g. /reload) return
    the new socket as the second element of the tuple.  All others return
    None so the caller keeps the existing socket unchanged.

    To add a new handler:
      1. Add a [[command]] block to commands.toml.
      2. Add a matching ``if canonical_cmd == "/your-command":`` branch here.

    Args:
        canonical_cmd: The resolved primary command key (aliases already
                       normalised by the caller via canonical_map).
        sock:          The currently active gateway socket.

    Returns:
        (should_break, new_sock) where should_break signals the main loop to
        exit and new_sock is a replacement socket or None.
    """
    # --- /exit ---
    if canonical_cmd in ("/exit", "/quit"):
        console.print("[dim]Goodbye.[/dim]")
        return True, None

    # --- /clean ---
    if canonical_cmd in ("/clean", "/clear"):
        console.clear()
        print_header()
        return False, None

    # --- /config ---
    if canonical_cmd == "/config":
        editor = os.environ.get("EDITOR", "nano")
        if not CONFIG_FILE.exists():
            CONFIG_DIR.mkdir(parents=True, exist_ok=True)
            CONFIG_FILE.touch()
        subprocess.run([editor, str(CONFIG_FILE)])
        console.print(
            f"[dim]Closed {editor}. Use /reload to apply changes and clear cache.[/dim]"
        )
        return False, None

    # --- /reload ---
    if canonical_cmd == "/reload":
        console.clear()
        with Live(
            Spinner("dots", text="[cyan]Reloading config and resetting brain...[/cyan]"),
            refresh_per_second=10,
            transient=True,
        ):
            try:
                new_sock = restart_brain_process(sock)
                console.clear()
                print_header()
                console.print(
                    "\n\n[green]●[/green] [dim]Configuration reloaded and "
                    "KV cache cleared. Session restarted.[/dim]\n"
                )
                return False, new_sock
            except Exception as exc:
                print_error(f"Failed to reload session: {exc}")
                return False, None

    # --- /newchat ---
    if canonical_cmd == "/newchat":
        console.clear()
        print_header()
        with Live(
            Spinner("dots", text="[cyan]Resetting session...[/cyan]"),
            refresh_per_second=10,
            transient=True,
        ):
            try:
                send_message(sock, "/newchat")
                console.print("[green]●[/green] [dim]New chat session started[/dim]\n")
            except Exception as exc:
                print_error(f"Failed to reset session: {exc}")
        return False, None

    # Defensive fallthrough — command is valid but has no registered handler
    console.print(f"[dim]No handler registered for '{canonical_cmd}'.[/dim]")
    return False, None


# ---------------------------------------------------------------------------
# Main Execution Loop
# ---------------------------------------------------------------------------

def main() -> None:
    """Entry point for the Tantalum CLI."""
    load_dotenv(dotenv_path=CONFIG_FILE, override=True)

    atexit.register(cleanup)
    signal.signal(signal.SIGINT, lambda s, f: sys.exit(0))
    signal.signal(signal.SIGTERM, lambda s, f: sys.exit(0))

    # Load slash commands from TOML before printing the header so any parse
    # warnings appear above the UI chrome rather than inside it.
    command_entries = load_commands(COMMANDS_TOML)
    display_pairs, valid_commands, canonical_map = build_command_index(command_entries)

    print_header()

    # --- 1. Start Gateway Process ---
    with Live(Spinner("dots", text="[cyan]Starting gateway...[/cyan]"), refresh_per_second=10, transient=True):
        actual_bin = GATEWAY_BIN_REL if GATEWAY_BIN_REL.exists() else GATEWAY_BIN_DEV

        if not actual_bin.exists():
            if shutil.which("zig"):
                console.print("[yellow]warning[/yellow]  Gateway binary not found, building from source...")
                subprocess.run(["zig", "build", "-Doptimize=ReleaseSafe"], cwd=ROOT_DIR / "backend", check=True)
                actual_bin = GATEWAY_BIN_DEV
            else:
                print_error("Tantalum Gateway binary not found and Zig is not installed.")
                sys.exit(1)

        start_process("gateway", [str(actual_bin)], env={"TANTALUM_SECRET": SESSION_SECRET})

        if not wait_for_gateway():
            print_error("Gateway failed to start.")
            sys.exit(1)

    console.print("[green]●[/green] [dim]Gateway ready[/dim]")

    # --- 2. Start Brain Process ---
    with Live(Spinner("dots", text="[cyan]Starting brain...[/cyan]"), refresh_per_second=10, transient=True):
        start_process("brain", [sys.executable, str(BRAIN_SCRIPT)])

        if not wait_for_brain():
            print_error("Brain failed to start.")
            sys.exit(1)

    console.print("[green]●[/green] [dim]Brain ready[/dim]")

    # --- 3. Connect to Backend ---
    with Live(Spinner("dots", text="[cyan]Connecting...[/cyan]"), refresh_per_second=10, transient=True):
        try:
            sock = connect()
        except Exception as exc:
            print_error(f"Connection failed: {exc}")
            sys.exit(1)

    console.print("[green]●[/green] [dim]Connected[/dim]\n")

    # --- 4. Chat Interface Loop ---
    session = build_prompt_session(display_pairs)

    try:
        while True:
            try:
                user_input = session.prompt(HTML("<ansicyan><b>> </b></ansicyan>")).strip()
            except (EOFError, KeyboardInterrupt):
                console.print("\n[dim]Goodbye.[/dim]")
                break

            if not user_input:
                continue

            # ---------------------------------------------------------------
            # Slash-command dispatch
            # ---------------------------------------------------------------
            if user_input.startswith("/"):
                if user_input not in valid_commands:
                    # Unrecognised — do not forward to the backend.
                    console.print(
                        f"[dim]Unknown command '{user_input}'. "
                        "Type / and select from the menu.[/dim]"
                    )
                    continue

                # Resolve alias -> canonical handler key, then dispatch
                canonical_cmd = canonical_map.get(user_input, user_input)
                should_break, new_sock = dispatch_slash_command(canonical_cmd, sock)

                if new_sock is not None:
                    sock = new_sock
                if should_break:
                    break
                continue

            # ---------------------------------------------------------------
            # Standard message transmission
            # ---------------------------------------------------------------
            print_user(user_input)

            with Live(
                Spinner("dots", text="[cyan]thinking...[/cyan]"),
                refresh_per_second=10,
                transient=True,
            ):
                try:
                    reply = send_message(sock, user_input)
                except Exception as exc:
                    print_error(f"Send failed: {exc}")
                    try:
                        sock = connect()
                        reply = send_message(sock, user_input)
                    except Exception as exc2:
                        print_error(f"Reconnect failed: {exc2}")
                        continue

            if reply.get("role") == "error":
                print_error(reply.get("content", "unknown error"))
            else:
                print_assistant(reply.get("content", ""))

    finally:
        sock.close()


if __name__ == "__main__":
    main()