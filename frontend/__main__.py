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

import chromadb
import litellm
from chromadb.utils import embedding_functions

# --- Runtime Detection ---

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

# --- Configuration & Constants ---
BRAIN_SOCKET_PATH = "/tmp/tantalum-brain.sock"
MODEL = os.environ.get("TANTALUM_MODEL", "gemini/gemini-2.5-flash")
SYSTEM_PROMPT = os.environ.get(
    "TANTALUM_SYSTEM_PROMPT",
    "You are Tantalum, a zero-trust AI agent assistant. Be concise and helpful. "
    "Your first philosophy/principle is \"A good agent should want boundaries.\""
    "You have access to these tools ONLY for specific tasks: run_python, run_shell, read_file, write_file. "
    "Your boundaries are defined by the tools you are given; use them to their full potential within those limits."
    "You have to retrieve the information when you use the tools. "
    "You must hide the thinking process. "
    "When you proceed with multi-level reasoning, you should decompose the given query and infer what the user wants through the previous context."
    "After reaching a conclusion, verify your reasoning by working backwards: does each step logically support the final answer?"
    "NEVER call tools that are not in this list. "
    "Never return an empty response after tool use.",
)
LOOKBACK_BLOCK_LIMIT = 18
TOOL_RAG_TOP_K = 3

DOCKER_VOLUME = os.environ.get("TANTALUM_VOLUME", "tantalum-storage")
DOCKER_IMAGE = os.environ.get("TANTALUM_DOCKER_IMAGE", "python:3.12-slim")
DOCKER_MEMORY = os.environ.get("TANTALUM_DOCKER_MEMORY", "256m")
DOCKER_CPUS = os.environ.get("TANTALUM_DOCKER_CPUS", "0.5")
DOCKER_TIMEOUT = int(os.environ.get("TANTALUM_DOCKER_TIMEOUT", "30"))
PERSISTENT_CONTAINER = "tantalum-persistent"
CONTAINER_RUNTIME = os.environ.get("TANTALUM_RUNTIME", _detect_runtime())

# Gemini prompt cache TTL in seconds (1 hour default)
GEMINI_CACHE_TTL = os.environ.get("TANTALUM_GEMINI_CACHE_TTL", "3600s")


# ---------------------------------------------------------------------------
# Persistent Container Management
# ---------------------------------------------------------------------------

def ensure_persistent_container() -> None:
    """
    Ensure the persistent Docker container is running for storage operations.
    If it is not running, it initializes a new volume and starts the container.
    """
    result = subprocess.run(
        [CONTAINER_RUNTIME, "inspect", "--format", "{{.State.Running}}", PERSISTENT_CONTAINER],
        capture_output=True,
        text=True,
    )

    if result.returncode == 0 and result.stdout.strip() == "true":
        print(f"[Brain] Persistent container '{PERSISTENT_CONTAINER}' is already running.")
        return

    # Remove existing container if it's in a stopped/broken state
    subprocess.run([CONTAINER_RUNTIME, "rm", "-f", PERSISTENT_CONTAINER], capture_output=True)

    # Create volume if it doesn't exist
    subprocess.run([CONTAINER_RUNTIME, "volume", "create", DOCKER_VOLUME], capture_output=True)

    subprocess.run(
        [
            CONTAINER_RUNTIME, "run", "-d",
            "--name", PERSISTENT_CONTAINER,
            "--memory", DOCKER_MEMORY,
            "--cpus", DOCKER_CPUS,
            "-v", f"{DOCKER_VOLUME}:/storage:rw",
            DOCKER_IMAGE,
            "sleep", "infinity",
        ],
        check=True,
    )
    print(f"[Brain] Persistent container '{PERSISTENT_CONTAINER}' started.")


# ---------------------------------------------------------------------------
# Session Container Execution
# ---------------------------------------------------------------------------

def run_in_session_container(command: list[str], workspace: str) -> str:
    """
    Run a command inside an ephemeral, network-isolated Docker container.

    Args:
        command: The shell command/executable and its arguments.
        workspace: Path to the local workspace directory to mount.

    Returns:
        Standard output or standard error of the execution.
    """
    session_id = uuid.uuid4().hex[:8]
    container_name = f"tantalum-session-{session_id}"

    docker_cmd = [
        CONTAINER_RUNTIME, "run", "--rm",
        "--name", container_name,
        "--network", "none",
        "--memory", DOCKER_MEMORY,
        "--cpus", DOCKER_CPUS,
        "--workdir", "/workspace",
        "-v", f"{workspace}:/workspace:rw",
        "-v", f"{DOCKER_VOLUME}:/storage:ro",
        DOCKER_IMAGE,
    ] + command

    result = subprocess.run(
        docker_cmd,
        capture_output=True,
        text=True,
        timeout=DOCKER_TIMEOUT,
    )
    return result.stdout or result.stderr or "(no output)"


# ---------------------------------------------------------------------------
# Tool Registry
# ---------------------------------------------------------------------------

TOOL_REGISTRY = [
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
        "description": "Execute a shell command in an isolated Docker container.",
        "parameters": {
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "Shell command to execute"}
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
                "content": {"type": "string", "description": "Content to write"}
            },
            "required": ["path", "content"],
        },
    },
]


def _get_model_family(model: str) -> str:
    """
    Detect the model provider family from the model string prefix.

    Returns one of: "anthropic", "gemini", "openai", "unknown".
    """
    m = model.lower()
    if m.startswith("anthropic/") or m.startswith("claude"):
        return "anthropic"
    if m.startswith("gemini/") or m.startswith("vertex_ai/gemini"):
        return "gemini"
    if m.startswith("openai/") or m.startswith("gpt") or m.startswith("o1") or m.startswith("o3"):
        return "openai"
    return "unknown"


def to_litellm_tool(tool: dict, model: str = MODEL) -> dict:
    """
    Convert a tool registry entry into the provider-appropriate format for LiteLLM.

    - Anthropic: uses {"type": "function", "function": {...}} with optional
      cache_control on the last tool to enable prompt caching for the tool list.
    - Gemini: same OpenAI-style function spec; litellm handles translation internally.
    - OpenAI: standard OpenAI function calling spec with strict mode enabled.
    - Unknown: falls back to the OpenAI spec.

    Args:
        tool: A dict from TOOL_REGISTRY.
        model: The active model string (used to select the correct format).

    Returns:
        A dict formatted for the target provider via litellm.
    """
    family = _get_model_family(model)

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


def build_tools_for_request(tools: list[dict], model: str = MODEL) -> list[dict]:
    """
    Convert a list of tool registry entries for the active provider.

    For Anthropic models, attach cache_control to the last tool definition so
    that the entire tool list is eligible for prompt caching.

    Args:
        tools: List of dicts from TOOL_REGISTRY.
        model: The active model string.

    Returns:
        A list of provider-formatted tool dicts ready for the litellm call.
    """
    formatted = [to_litellm_tool(t, model) for t in tools]

    if _get_model_family(model) == "anthropic" and formatted:
        # Mark the last tool so the full tool block is cached by Anthropic.
        last = formatted[-1]
        last["cache_control"] = {"type": "ephemeral"}

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


def search_tools(collection: chromadb.Collection, query: str, k: int = TOOL_RAG_TOP_K) -> list[dict]:
    """
    Retrieve the top K most relevant tools for a given user query.
    """
    results = collection.query(query_texts=[query], n_results=min(k, len(TOOL_REGISTRY)))
    matched_names = [m["name"] for m in results["metadatas"][0]]

    # Filter original registry to return full tool definitions
    matched = [t for t in TOOL_REGISTRY if t["name"] in matched_names]
    return matched


# ---------------------------------------------------------------------------
# Tool Executor
# ---------------------------------------------------------------------------

def execute_tool(name: str, arguments: str) -> str:
    """
    Parse arguments and execute the specified tool, safely running it in Docker.
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
            output = run_in_session_container(["python3", "/workspace/run.py"], workspace)

        elif name == "run_shell":
            command = args.get("command", "")
            output = run_in_session_container(["sh", "-c", command], workspace)

        elif name == "read_file":
            path = args.get("path", "").lstrip("/")
            result = subprocess.run(
                [CONTAINER_RUNTIME, "exec", PERSISTENT_CONTAINER, "cat", f"/storage/{path}"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            output = result.stdout or result.stderr or "(empty file)"

        elif name == "write_file":
            path = args.get("path", "").lstrip("/")
            file_content = args.get("content", "")
            result = subprocess.run(
                [
                    CONTAINER_RUNTIME, "exec", "-i", PERSISTENT_CONTAINER, "sh", "-c",
                    f"mkdir -p $(dirname /storage/{path}) && cat > /storage/{path}"
                ],
                input=file_content,
                capture_output=True,
                text=True,
                timeout=10,
            )
            output = f"Written to /storage/{path}" if result.returncode == 0 else result.stderr
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

def build_system_message(model: str = MODEL) -> dict:
    """
    Construct the system message with provider-appropriate prompt caching markup.

    Caching strategy per provider:
      - Anthropic: cache_control block attached to the text part (charges for
        cache writes, then reads are discounted).
      - Gemini: cache_control with TTL in the content part (Google AI Studio
        supports ephemeral context caching via litellm).
      - OpenAI: automatic server-side caching for prompts >= 1024 tokens;
        no annotation required.
      - Unknown: plain string content, no caching markup.

    Args:
        model: The active model string.

    Returns:
        A dict representing the system message ready for the messages list.
    """
    family = _get_model_family(model)

    if family == "anthropic":
        # Anthropic explicit cache_control on the system prompt text block.
        content = [
            {
                "type": "text",
                "text": SYSTEM_PROMPT,
                "cache_control": {"type": "ephemeral"},
            }
        ]

    elif family == "gemini":
        # Gemini supports cache_control with an optional TTL via litellm.
        content = [
            {
                "type": "text",
                "text": SYSTEM_PROMPT,
                "cache_control": {
                    "type": "ephemeral",
                    "ttl": GEMINI_CACHE_TTL,
                },
            }
        ]

    else:
        # OpenAI: caching is automatic for large prompts; plain string is fine.
        # Unknown providers: safe fallback with no markup.
        content = SYSTEM_PROMPT

    return {"role": "system", "content": content}


# ---------------------------------------------------------------------------
# LLM Integration
# ---------------------------------------------------------------------------


def call_llm_with_tools(messages: list[dict], tools: list[dict]) -> str:
    """
    Call the LLM using LiteLLM, passing the conversation history and available tools.
    Handles tool calls internally and returns the final response.

    Tool format is selected per-provider via build_tools_for_request().
    """
    litellm_tools = build_tools_for_request(tools, MODEL) if tools else None

    response = litellm.completion(model=MODEL, messages=messages, tools=litellm_tools)
    msg = response.choices[0].message

    if not msg.tool_calls:
        return msg.content or "(no response)"

    valid_tool_names = {t["name"] for t in tools}

    # Fallback to standard completion if model hallucinates an invalid tool name
    if any(tc.function.name not in valid_tool_names for tc in msg.tool_calls):
        fallback = litellm.completion(model=MODEL, messages=messages)
        return fallback.choices[0].message.content or "(no response)"

    messages.append(msg)

    # Execute all tools requested by the LLM
    for tool_call in msg.tool_calls:
        result = execute_tool(tool_call.function.name, tool_call.function.arguments)
        messages.append({
            "role": "tool",
            "tool_call_id": tool_call.id,
            "content": result,
        })

    # Get the final synthesis after tool execution
    final_response = litellm.completion(model=MODEL, messages=messages, tools=litellm_tools)
    return final_response.choices[0].message.content or "(no response)"


# ---------------------------------------------------------------------------
# Lookback Monitor
# ---------------------------------------------------------------------------

def apply_lookback_monitor(history: list[dict]) -> list[dict]:
    """
    Injects a cache breakpoint into the conversation history to optimize
    context window handling for long conversations.
    """
    if len(history) <= LOOKBACK_BLOCK_LIMIT:
        return history

    inject_idx = len(history) - LOOKBACK_BLOCK_LIMIT
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
    line = json.dumps(obj, ensure_ascii=False) + "\n"
    conn.sendall(line.encode("utf-8"))


# ---------------------------------------------------------------------------
# Session Handler
# ---------------------------------------------------------------------------

def handle_session(conn: socket.socket, tool_collection: chromadb.Collection) -> None:
    """
    Handle an active communication session with the frontend.
    Processes user messages, manages conversation history, and invokes the LLM.
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
                write_jsonl(
                    conn,
                    {
                        "role": "assistant",
                        "content": "New chat session started. Context has been cleared.",
                        "done": True,
                    }
                )
                continue

            history.append({"role": "user", "content": content})
            history = apply_lookback_monitor(history)

            # Build provider-aware system message with prompt caching markup
            system_msg = build_system_message(MODEL)

            relevant_tools = search_tools(tool_collection, content)
            messages = [system_msg] + history

            response_text = call_llm_with_tools(messages, relevant_tools)
            history.append({"role": "assistant", "content": response_text})

            write_jsonl(conn, {"role": "assistant", "content": response_text, "done": True})

    except Exception as e:
        print(f"[Brain] Session error: {e}")
    finally:
        rfile.close()


def handle_connection(client_sock: socket.socket, tool_collection: chromadb.Collection) -> None:
    """Handle incoming connection and extract file descriptor."""
    try:
        fd = recv_fd(client_sock)
        if fd is not None:
            with fd_as_socket(fd) as conn:
                handle_session(conn, tool_collection)
    finally:
        client_sock.close()


def main() -> None:
    """Main execution loop for Tantalum Brain."""
    if os.path.exists(BRAIN_SOCKET_PATH):
        os.unlink(BRAIN_SOCKET_PATH)

    ensure_persistent_container()
    tool_collection = init_tool_db()

    server_sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server_sock.bind(BRAIN_SOCKET_PATH)
    server_sock.listen(16)

    family = _get_model_family(MODEL)
    print(f"[Brain] Running with prompt caching enabled (provider: {family}).")
    print(f"[Brain] Target Model: {MODEL}")
    print(f"[Brain] Container Runtime: {CONTAINER_RUNTIME}")
    print(f"[Brain] Listening on {BRAIN_SOCKET_PATH}...")

    while True:
        client_sock, _ = server_sock.accept()
        threading.Thread(
            target=handle_connection,
            args=(client_sock, tool_collection),
            daemon=True
        ).start()


if __name__ == "__main__":
    main()