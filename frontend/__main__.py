"""
Tantalum-Brain: Intelligence Layer
Copyright 2026 Skystarry-AI
SPDX-License-Identifier: Apache-2.0

This module acts as the backend "brain" for the Tantalum AI agent.
It handles Docker-based isolated tool execution, vector database
retrieval (ChromaDB) for Tool-RAG, and communication with the frontend
via Unix Domain Sockets (UDS).
"""

import array
import json
import os
import shutil
import socket
import subprocess
import tempfile
import threading
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field

import chromadb
import litellm
from chromadb.utils import embedding_functions
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Runtime Detection
# ---------------------------------------------------------------------------

def _detect_runtime() -> str:
    """
    Auto-detect the available container runtime.
    Prefers Podman (rootless, no daemon required) over Docker.
    Raises RuntimeError if neither is found.
    """
    for runtime in ("podman", "docker"):
        if shutil.which(runtime):
            return runtime
    raise RuntimeError(
        "No container runtime found. Install Podman (recommended) or Docker."
    )


# ---------------------------------------------------------------------------
# Configuration Dataclass
# ---------------------------------------------------------------------------

_DEFAULT_SYSTEM_PROMPT = (
    "You are Tantalum, a zero-trust AI agent assistant. Be concise and helpful. "
    "Your first philosophy/principle is \"A good agent should want boundaries.\""
    "You have access to these tools ONLY for specific tasks: run_python, run_shell, read_file, write_file. "
    "Your boundaries are defined by the tools you are given; use them to their full potential within those limits."
    "You have to retrieve the information when you use the tools. "
    "You must hide the thinking process. "
    "When you proceed with multi-level reasoning, you should decompose the given query and infer what the user wants through the previous context."
    "After reaching a conclusion, verify your reasoning by working backwards: does each step logically support the final answer?"
    "NEVER call tools that are not in this list. "
    "Never return an empty response after tool use."
)


@dataclass
class BrainConfig:
    """
    All runtime-configurable settings for the Brain process.

    Every field maps to a TANTALUM_* environment variable and is populated
    by BrainConfig.from_env() after load_dotenv() has run.  Keeping config
    in a single dataclass makes it easy to pass around, inspect, and test
    without relying on global state or os.environ at call sites.
    """

    # LLM settings
    model: str = "gemini/gemini-2.5-flash"
    system_prompt: str = field(default=_DEFAULT_SYSTEM_PROMPT)
    gemini_cache_ttl: str = "3600s"

    # Conversation settings
    lookback_block_limit: int = 18
    tool_rag_top_k: int = 3

    # Container settings
    runtime: str = field(default_factory=_detect_runtime)
    volume: str = "tantalum-storage"
    image: str = "python:3.12-slim"
    memory: str = "256m"
    cpus: str = "0.5"
    timeout: int = 30
    persistent_container: str = "tantalum-persistent"

    # Socket path
    socket_path: str = "/tmp/tantalum-brain.sock"

    @classmethod
    def from_env(cls) -> "BrainConfig":
        """
        Construct a BrainConfig from environment variables.

        Must be called after load_dotenv() so that values from config.env
        are already present in os.environ.
        """
        return cls(
            model=os.environ.get("TANTALUM_MODEL", "gemini/gemini-2.5-flash"),
            system_prompt=os.environ.get("TANTALUM_SYSTEM_PROMPT", _DEFAULT_SYSTEM_PROMPT),
            gemini_cache_ttl=os.environ.get("TANTALUM_GEMINI_CACHE_TTL", "3600s"),
            lookback_block_limit=int(os.environ.get("TANTALUM_LOOKBACK_LIMIT", "18")),
            tool_rag_top_k=int(os.environ.get("TANTALUM_TOOL_RAG_TOP_K", "3")),
            runtime=os.environ.get("TANTALUM_RUNTIME", _detect_runtime()),
            volume=os.environ.get("TANTALUM_VOLUME", "tantalum-storage"),
            image=os.environ.get("TANTALUM_DOCKER_IMAGE", "python:3.12-slim"),
            memory=os.environ.get("TANTALUM_DOCKER_MEMORY", "256m"),
            cpus=os.environ.get("TANTALUM_DOCKER_CPUS", "0.5"),
            timeout=int(os.environ.get("TANTALUM_DOCKER_TIMEOUT", "30")),
            persistent_container=os.environ.get("TANTALUM_PERSISTENT_CONTAINER", "tantalum-persistent"),
            socket_path=os.environ.get("TANTALUM_SOCKET_PATH", "/tmp/tantalum-brain.sock"),
        )


# ---------------------------------------------------------------------------
# Persistent Container Management
# ---------------------------------------------------------------------------

def ensure_persistent_container(cfg: BrainConfig) -> None:
    """
    Ensure the persistent container is running for storage operations.
    If it is not running, initializes a new volume and starts the container.
    """
    result = subprocess.run(
        [cfg.runtime, "inspect", "--format", "{{.State.Running}}", cfg.persistent_container],
        capture_output=True,
        text=True,
    )

    if result.returncode == 0 and result.stdout.strip() == "true":
        print(f"[Brain] Persistent container '{cfg.persistent_container}' is already running.")
        return

    # Remove existing container if it's in a stopped/broken state
    subprocess.run([cfg.runtime, "rm", "-f", cfg.persistent_container], capture_output=True)

    # Create volume if it doesn't exist
    subprocess.run([cfg.runtime, "volume", "create", cfg.volume], capture_output=True)

    subprocess.run(
        [
            cfg.runtime, "run", "-d",
            "--name", cfg.persistent_container,
            "--memory", cfg.memory,
            "--cpus", cfg.cpus,
            "-v", f"{cfg.volume}:/storage:rw",
            cfg.image,
            "sleep", "infinity",
        ],
        check=True,
    )
    print(f"[Brain] Persistent container '{cfg.persistent_container}' started.")


# ---------------------------------------------------------------------------
# Session Container Execution
# ---------------------------------------------------------------------------

def run_in_session_container(
    command: list[str],
    workspace: str,
    cfg: BrainConfig,
) -> str:
    """
    Run a command inside an ephemeral, network-isolated container.

    Args:
        command:   The executable and its arguments.
        workspace: Local directory mounted as /workspace inside the container.
        cfg:       Active BrainConfig instance.

    Returns:
        Combined stdout/stderr of the execution, or "(no output)".
    """
    container_name = f"tantalum-session-{uuid.uuid4().hex[:8]}"

    docker_cmd = [
        cfg.runtime, "run", "--rm",
        "--name", container_name,
        "--network", "none",
        "--memory", cfg.memory,
        "--cpus", cfg.cpus,
        "--workdir", "/workspace",
        "-v", f"{workspace}:/workspace:rw",
        "-v", f"{cfg.volume}:/storage:ro",
        cfg.image,
    ] + command

    result = subprocess.run(
        docker_cmd,
        capture_output=True,
        text=True,
        timeout=cfg.timeout,
    )
    return result.stdout or result.stderr or "(no output)"


# ---------------------------------------------------------------------------
# Tool Registry
# ---------------------------------------------------------------------------

TOOL_REGISTRY: list[dict] = [
    {
        "name": "run_python",
        "description": "Execute a Python code snippet in an isolated Docker container and return stdout/stderr.",
        "parameters": {
            "type": "object",
            "properties": {
                "code": {"type": "string", "description": "Python code to execute"}
            },
            "required": ["code"],
        },
    },
    {
        "name": "run_shell",
        "description": (
            "Execute a shell command in an isolated Docker container. "
            "Set loop=true when the command needs to be repeated until it "
            "succeeds or produces no more output — e.g. iterative grep/sed "
            "passes over a file. The agent will keep calling this tool in a "
            "loop as long as you return loop=true."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "Shell command to execute.",
                },
                "loop": {
                    "type": "boolean",
                    "description": (
                        "When true, signals the agent runtime to keep invoking "
                        "this tool call in a loop until the model sets loop=false "
                        "or omits it. Use for iterative operations such as "
                        "grep/sed passes that must run multiple times."
                    ),
                },
            },
            "required": ["command"],
        },
    },
    {
        "name": "read_file",
        "description": "Read file from /storage.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path relative to /storage"}
            },
            "required": ["path"],
        },
    },
    {
        "name": "write_file",
        "description": "Write to file in /storage.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path relative to /storage"},
                "content": {"type": "string", "description": "Content to write"},
            },
            "required": ["path", "content"],
        },
    },
]


# ---------------------------------------------------------------------------
# Model Family Detection & Tool Formatting
# ---------------------------------------------------------------------------

def _get_model_family(model: str) -> str:
    """
    Detect the model provider family from the model string prefix.

    Returns one of: "anthropic", "gemini", "openai", "ollama", "unknown".

    Ollama models are exposed via an OpenAI-compatible endpoint through
    litellm (ollama/ or ollama_chat/ prefix).  They do not support explicit
    cache_control markup, but the serving runtime (llama.cpp / ollama)
    handles KV caching internally as long as the system prompt prefix
    remains stable across turns — identical behaviour to the OpenAI path.
    """
    m = model.lower()
    if m.startswith("anthropic/") or m.startswith("claude"):
        return "anthropic"
    if m.startswith("gemini/") or m.startswith("vertex_ai/gemini"):
        return "gemini"
    if m.startswith("openai/") or m.startswith("gpt") or m.startswith("o1") or m.startswith("o3"):
        return "openai"
    if m.startswith("ollama/") or m.startswith("ollama_chat/"):
        return "ollama"
    return "unknown"


def to_litellm_tool(tool: dict, family: str) -> dict:
    """
    Convert a tool registry entry into the provider-appropriate format for LiteLLM.

    Args:
        tool:   A dict from TOOL_REGISTRY.
        family: Provider family string from _get_model_family().

    Returns:
        A provider-formatted tool dict.
    """
    if family == "openai":
        # OpenAI function calling with strict JSON schema enforcement
        return {
            "type": "function",
            "function": {
                "name": tool["name"],
                "description": tool["description"],
                "parameters": tool["parameters"],
                "strict": True,
            },
        }

    # Anthropic and Gemini both accept the OpenAI-style spec through litellm.
    # Anthropic cache_control is injected separately in build_tools_for_request().
    return {
        "type": "function",
        "function": {
            "name": tool["name"],
            "description": tool["description"],
            "parameters": tool["parameters"],
        },
    }


def build_tools_for_request(tools: list[dict], cfg: BrainConfig) -> list[dict]:
    """
    Convert a list of tool registry entries for the active provider.

    For Anthropic models, attaches cache_control to the last tool definition
    so that the entire tool list is eligible for prompt caching.

    Args:
        tools: List of dicts from TOOL_REGISTRY.
        cfg:   Active BrainConfig instance.

    Returns:
        A list of provider-formatted tool dicts ready for the litellm call.
    """
    family = _get_model_family(cfg.model)
    formatted = [to_litellm_tool(t, family) for t in tools]

    if family == "anthropic" and formatted:
        # Mark the last tool so the full tool block is cached by Anthropic
        formatted[-1]["cache_control"] = {"type": "ephemeral"}

    return formatted


# ---------------------------------------------------------------------------
# Tool RAG (Retrieval-Augmented Generation)
# ---------------------------------------------------------------------------

def init_tool_db() -> chromadb.Collection:
    """
    Initialize the ChromaDB collection containing descriptions of available tools.
    """
    ef = embedding_functions.DefaultEmbeddingFunction()
    client = chromadb.Client()
    collection = client.get_or_create_collection("tools", embedding_function=ef)

    existing = collection.get()["ids"]
    for tool in TOOL_REGISTRY:
        if tool["name"] not in existing:
            collection.add(
                ids=[tool["name"]],
                documents=[f"{tool['name']}: {tool['description']}"],
                metadatas=[{"name": tool["name"]}],
            )
    return collection


def search_tools(
    collection: chromadb.Collection,
    query: str,
    cfg: BrainConfig,
) -> list[dict]:
    """
    Retrieve the top K most relevant tools for a given user query.
    """
    results = collection.query(
        query_texts=[query],
        n_results=min(cfg.tool_rag_top_k, len(TOOL_REGISTRY)),
    )
    matched_names = {m["name"] for m in results["metadatas"][0]}
    return [t for t in TOOL_REGISTRY if t["name"] in matched_names]


# ---------------------------------------------------------------------------
# Tool Executor
# ---------------------------------------------------------------------------

def validate_storage_path(raw_path: str) -> str:
    """
    Validate that the path is within /storage and return the absolute path inside the container.
    Prevents Path Traversal attacks.

    Args:
        raw_path: The user-provided path string.

    Returns:
        The normalized absolute path starting with /storage.

    Raises:
        ValueError: If the path attempts to traverse outside /storage.
    """
    # 1. Normalize path (resolve .. and .)
    clean_path = os.path.normpath(f"/storage/{raw_path.lstrip('/')}")

    # 2. Ensure path starts with /storage
    if not clean_path.startswith("/storage"):
        raise ValueError("Access denied: Path must be within /storage")

    return clean_path


def execute_tool(name: str, arguments: str, cfg: BrainConfig) -> str:
    """
    Parse arguments and execute the specified tool inside a container.
    """
    try:
        args = json.loads(arguments)
    except json.JSONDecodeError:
        return "Error: Invalid JSON arguments provided."

    print(f"[Brain] Executing tool: {name}")

    workspace = tempfile.mkdtemp(prefix="tantalum-")
    try:
        if name == "run_python":
            code = args.get("code", "")
            code_file = os.path.join(workspace, "run.py")
            with open(code_file, "w") as f:
                f.write(code)
            output = run_in_session_container(["python3", "/workspace/run.py"], workspace, cfg)

        elif name == "run_shell":
            command = args.get("command", "")
            output = run_in_session_container(["sh", "-c", command], workspace, cfg)

        elif name == "read_file":
            raw_path = args.get("path", "")
            try:
                safe_path = validate_storage_path(raw_path)
                # Use list form for subprocess.run to prevent shell injection
                result = subprocess.run(
                    [cfg.runtime, "exec", cfg.persistent_container, "cat", safe_path],
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                output = result.stdout or result.stderr or "(empty file)"
            except ValueError as e:
                output = str(e)

        elif name == "write_file":
            raw_path = args.get("path", "")
            file_content = args.get("content", "")
            try:
                safe_path = validate_storage_path(raw_path)
                dir_path = os.path.dirname(safe_path)

                # 1. Create directory (mkdir -p)
                subprocess.run(
                    [cfg.runtime, "exec", cfg.persistent_container, "mkdir", "-p", dir_path],
                    check=True,
                    timeout=5
                )

                # 2. Write file using tee (avoids shell injection in content redirection)
                result = subprocess.run(
                    [cfg.runtime, "exec", "-i", cfg.persistent_container, "tee", safe_path],
                    input=file_content,
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                output = f"Written to {safe_path}" if result.returncode == 0 else result.stderr
            except (ValueError, subprocess.CalledProcessError) as e:
                output = f"Error: {str(e)}"

        else:
            output = f"Error: Unknown tool '{name}'"

    except Exception as e:
        output = f"Error during execution: {e}"
    finally:
        shutil.rmtree(workspace, ignore_errors=True)

    return output


# ---------------------------------------------------------------------------
# System Message Builder (per-provider prompt caching)
# ---------------------------------------------------------------------------

def build_system_message(cfg: BrainConfig) -> dict:
    """
    Construct the system message with provider-appropriate prompt caching markup.

    Caching strategy per provider:
      - Anthropic: cache_control block on the text part (charges for cache
        writes; subsequent reads are discounted).
      - Gemini: cache_control with TTL (Google AI Studio ephemeral caching
        via litellm).
      - OpenAI: automatic server-side caching for prompts >= 1024 tokens;
        no annotation required.
      - Ollama: OpenAI-compatible endpoint; KV caching is handled internally
        by the serving runtime as long as the system prompt prefix is stable.
      - Unknown: plain string fallback.

    Args:
        cfg: Active BrainConfig instance.

    Returns:
        A system message dict ready for the messages list.
    """
    family = _get_model_family(cfg.model)

    if family == "anthropic":
        content = [
            {
                "type": "text",
                "text": cfg.system_prompt,
                "cache_control": {"type": "ephemeral"},
            }
        ]

    elif family == "gemini":
        content = [
            {
                "type": "text",
                "text": cfg.system_prompt,
                "cache_control": {
                    "type": "ephemeral",
                    "ttl": cfg.gemini_cache_ttl,
                },
            }
        ]

    else:
        content = cfg.system_prompt

    return {"role": "system", "content": content}


# ---------------------------------------------------------------------------
# LLM Integration
# ---------------------------------------------------------------------------

_MAX_AGENT_ITERATIONS = 32  # Hard cap to prevent runaway loops


def call_llm_with_tools(
    messages: list[dict],
    tools: list[dict],
    cfg: BrainConfig,
    conn: socket.socket | None = None,
) -> str:
    """
    Run the agentic tool-use loop via LiteLLM and return the final response.

    The loop continues as long as the model emits tool calls.  For the
    ``run_shell`` tool, the model may include a ``loop`` boolean argument.
    When ``loop=true``, the runtime re-executes the *same* tool call (with
    the same arguments) and feeds the new output back, repeating until the
    model either stops requesting it or _MAX_AGENT_ITERATIONS is reached.

    If *conn* is provided, a lightweight JSON event is written to the socket
    before each tool execution so the CLI can display which tool is active::

        {"event": "tool_start", "tool": "<tool_name>"}

    Args:
        messages: Full message list (system message + conversation history).
        tools:    Relevant tool registry entries for this turn.
        cfg:      Active BrainConfig instance.
        conn:     Optional connected client socket for streaming tool events.

    Returns:
        The final assistant response string.
    """
    litellm_tools = build_tools_for_request(tools, cfg) if tools else None
    valid_tool_names = {t["name"] for t in tools}

    for iteration in range(_MAX_AGENT_ITERATIONS):
        response = litellm.completion(
            model=cfg.model,
            messages=messages,
            tools=litellm_tools,
        )
        msg = response.choices[0].message

        # No tool calls — model produced a final answer.
        if not msg.tool_calls:
            return msg.content or "(no response)"

        # Guard against hallucinated tool names before appending to history.
        if any(tc.function.name not in valid_tool_names for tc in msg.tool_calls):
            fallback = litellm.completion(model=cfg.model, messages=messages)
            return fallback.choices[0].message.content or "(no response)"

        messages.append(msg)

        # Execute every tool the model requested in this turn.
        for tool_call in msg.tool_calls:
            tool_name = tool_call.function.name

            # Notify the CLI which tool is running so it can update its UI.
            if conn is not None:
                try:
                    write_jsonl(conn, {"event": "tool_start", "tool": tool_name})
                except OSError:
                    pass  # Tolerate a broken pipe; the reply will still arrive.

            # Parse arguments once; we may need them for loop repetition.
            try:
                args = json.loads(tool_call.function.arguments)
            except json.JSONDecodeError:
                args = {}

            result = execute_tool(tool_name, tool_call.function.arguments, cfg)
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": result,
            })

            # --- Loop extension for run_shell ---
            # When the model sets loop=true we repeat the identical command,
            # appending successive outputs, until the flag is absent/false or
            # the iteration cap is reached.
            if tool_name == "run_shell" and args.get("loop", False):
                loop_iter = 0
                max_loop = _MAX_AGENT_ITERATIONS

                while loop_iter < max_loop:
                    loop_iter += 1
                    print(
                        f"[Brain] run_shell loop iteration {loop_iter} "
                        f"(command: {args.get('command', '')!r})"
                    )

                    if conn is not None:
                        try:
                            write_jsonl(
                                conn,
                                {
                                    "event": "tool_start",
                                    "tool": tool_name,
                                    "loop_iter": loop_iter,
                                },
                            )
                        except OSError:
                            pass

                    loop_result = execute_tool(
                        tool_name, tool_call.function.arguments, cfg
                    )

                    # Feed the loop output back as a synthetic tool-result so
                    # the model can decide whether to continue or stop.
                    loop_check_messages = messages + [
                        {
                            "role": "tool",
                            "tool_call_id": tool_call.id + f"_loop{loop_iter}",
                            "content": loop_result,
                        }
                    ]

                    # Ask the model whether to keep looping.
                    loop_response = litellm.completion(
                        model=cfg.model,
                        messages=loop_check_messages,
                        tools=litellm_tools,
                    )
                    loop_msg = loop_response.choices[0].message

                    # If the model returns plain text it has decided to stop.
                    if not loop_msg.tool_calls:
                        messages.extend(loop_check_messages[len(messages):])
                        messages.append(loop_msg)
                        return loop_msg.content or "(no response)"

                    # Check if the continued tool call still requests looping.
                    next_args_raw = loop_msg.tool_calls[0].function.arguments
                    try:
                        next_args = json.loads(next_args_raw)
                    except json.JSONDecodeError:
                        next_args = {}

                    messages.extend(loop_check_messages[len(messages):])
                    messages.append(loop_msg)

                    # Persist the loop result as a proper tool message.
                    messages.append({
                        "role": "tool",
                        "tool_call_id": loop_msg.tool_calls[0].id,
                        "content": loop_result,
                    })

                    if not next_args.get("loop", False):
                        break

                    # Update args for next iteration.
                    args = next_args
                    tool_call = loop_msg.tool_calls[0]

    # Iteration cap reached — perform a final synthesis pass.
    print(f"[Brain] Agent loop reached iteration cap ({_MAX_AGENT_ITERATIONS}).")
    final = litellm.completion(model=cfg.model, messages=messages)
    return final.choices[0].message.content or "(no response)"


# ---------------------------------------------------------------------------
# Lookback Monitor
# ---------------------------------------------------------------------------

def apply_lookback_monitor(history: list[dict], cfg: BrainConfig) -> list[dict]:
    """
    Inject a cache breakpoint marker into long conversation histories to keep
    the active context window within cfg.lookback_block_limit turns.
    """
    if len(history) <= cfg.lookback_block_limit:
        return history

    inject_idx = len(history) - cfg.lookback_block_limit
    marker = {"role": "system", "content": "[CACHE_BREAKPOINT]"}

    if history[inject_idx] == marker:
        return history

    return history[:inject_idx] + [marker] + history[inject_idx:]


# ---------------------------------------------------------------------------
# File Descriptor / Socket Utilities
# ---------------------------------------------------------------------------

@contextmanager
def fd_as_socket(fd: int):
    """Context manager to convert a raw file descriptor to a socket object."""
    conn = socket.fromfd(fd, socket.AF_INET, socket.SOCK_STREAM)
    os.close(fd)
    try:
        yield conn
    finally:
        conn.close()


def recv_fd(sock: socket.socket) -> int | None:
    """Receive a file descriptor over a Unix domain socket."""
    msg, ancdata, _, _ = sock.recvmsg(1, socket.CMSG_SPACE(array.array("i").itemsize))
    if not msg:
        return None

    for cmsg_level, cmsg_type, cmsg_data in ancdata:
        if cmsg_level == socket.SOL_SOCKET and cmsg_type == socket.SCM_RIGHTS:
            fd_array = array.array("i")
            fd_array.frombytes(cmsg_data[:fd_array.itemsize])
            return fd_array[0]

    return None


def write_jsonl(conn: socket.socket, obj: dict) -> None:
    """Write a dictionary as a JSON Line to the socket."""
    conn.sendall((json.dumps(obj, ensure_ascii=False) + "\n").encode("utf-8"))


# ---------------------------------------------------------------------------
# Session Handler
# ---------------------------------------------------------------------------

def handle_session(
    conn: socket.socket,
    tool_collection: chromadb.Collection,
    cfg: BrainConfig,
) -> None:
    """
    Handle an active communication session with the frontend.
    Processes user messages, manages conversation history, and invokes the LLM.

    Args:
        conn:            Connected client socket.
        tool_collection: Initialised ChromaDB tool collection.
        cfg:             Active BrainConfig instance.
    """
    history: list[dict] = []
    rfile = conn.makefile("rb")

    try:
        for line in rfile:
            line = line.strip()
            if not line:
                continue

            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                continue

            content = msg.get("content", "")
            if msg.get("role") != "user" or not content:
                continue

            # Handle reset command
            if content == "/newchat":
                history.clear()
                write_jsonl(conn, {
                    "role": "assistant",
                    "content": "New chat session started. Context has been cleared.",
                    "done": True,
                })
                continue

            history.append({"role": "user", "content": content})
            history = apply_lookback_monitor(history, cfg)

            system_msg = build_system_message(cfg)
            relevant_tools = search_tools(tool_collection, content, cfg)
            messages = [system_msg] + history

            response_text = call_llm_with_tools(messages, relevant_tools, cfg, conn=conn)
            history.append({"role": "assistant", "content": response_text})

            write_jsonl(conn, {"role": "assistant", "content": response_text, "done": True})

    except Exception as e:
        print(f"[Brain] Session error: {e}")
    finally:
        rfile.close()


def handle_connection(
    client_sock: socket.socket,
    tool_collection: chromadb.Collection,
    cfg: BrainConfig,
) -> None:
    """Handle an incoming UDS connection and extract the forwarded file descriptor."""
    try:
        fd = recv_fd(client_sock)
        if fd is not None:
            with fd_as_socket(fd) as conn:
                handle_session(conn, tool_collection, cfg)
    finally:
        client_sock.close()


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------

def main() -> None:
    """Main execution loop for Tantalum Brain."""
    # Resolve config.env path, then load it before building BrainConfig so
    # that all os.environ.get() calls inside from_env() see the user's settings.
    config_file = (
        os.environ.get("TANTALUM_CONFIG")
        or os.path.expanduser("~/.config/tantalum/config.env")
    )
    load_dotenv(config_file, override=True)

    cfg = BrainConfig.from_env()

    if os.path.exists(cfg.socket_path):
        os.unlink(cfg.socket_path)

    ensure_persistent_container(cfg)
    tool_collection = init_tool_db()

    server_sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server_sock.bind(cfg.socket_path)
    server_sock.listen(16)

    family = _get_model_family(cfg.model)
    print(f"[Brain] Running with prompt caching enabled (provider: {family}).")
    print(f"[Brain] Target Model: {cfg.model}")
    print(f"[Brain] Container Runtime: {cfg.runtime}")
    print(f"[Brain] Listening on {cfg.socket_path}...")

    while True:
        client_sock, _ = server_sock.accept()
        threading.Thread(
            target=handle_connection,
            args=(client_sock, tool_collection, cfg),
            daemon=True,
        ).start()


if __name__ == "__main__":
    main()