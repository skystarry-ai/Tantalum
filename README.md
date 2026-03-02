# Tantalum

**Security-First AI Agent Runtime**

> *"A good agent should want boundaries."*

Tantalum is an open-source AI agent runtime designed for security-sensitive environments. Rather than exposing network ports or granting the model direct execution authority, Tantalum enforces a strict layered isolation model: every action is mediated through explicitly allowlisted tools, every tool invocation runs inside an ephemeral container, and the intelligence layer never touches the network directly.

The system is built on three principles:

- **Bounded execution** — the model can only act through the tools it is given, and nothing else.
- **Explicit tool mediation** — no action is taken without passing through the tool registry.
- **Minimal attack surface** — each layer knows as little as necessary about the others.

<p style="text-align: center;">
    <img src="https://github.com/user-attachments/assets/70004a9b-1e7e-46b8-a2c5-84e5b9ad2e8d" style="max-width: 100%; height: auto;">
</p>

---

## Architecture

```
CLI (frontend/cli.py)
    └─ SPA Packet (HMAC-SHA256)
        └─ Zig Gateway (port 8080)
            └─ FD passing via SCM_RIGHTS (Unix Domain Socket)
                └─ Python Brain (frontend/__main__.py)
                    ├─ LiteLLM (Ollama / Gemini / Anthropic / OpenAI)
                    ├─ Tool RAG (ChromaDB)
                    └─ Docker Sandbox
                        ├─ Persistent Container (tantalum-persistent)
                        │   └─ Docker Volume (tantalum-storage) :rw
                        └─ Session Container (tantalum-session-{uuid})
                            └─ Docker Volume (tantalum-storage) :ro
```

Execution is session-bounded. Each tool invocation runs in a freshly created, network-isolated container and is destroyed immediately after completion.

---

## Key Features

### Zero-Trust Execution Gateway (Zig)

* **Single Packet Authorization (SPA)** using HMAC-SHA256 — the service remains inaccessible until a valid packet arrives.
* **File descriptor passing via `SCM_RIGHTS`** over Unix Domain Sockets — the Python brain never binds a network port.
* Memory-safe implementation in Zig 0.15.2.

### Intelligence Layer (Python)

* **LiteLLM integration** — switch between Ollama, Gemini, Anthropic, OpenAI, or any compatible provider via a single environment variable.
* **Tool RAG with ChromaDB** — only the tools most relevant to the current query are injected into the context window.
* **Provider-aware prompt caching** — system prompt and tool definitions are cached per provider (Anthropic `cache_control`, Gemini TTL, OpenAI automatic server-side caching).
* **Lookback Monitor** — automatically inserts KV cache breakpoints when conversation history grows beyond a configurable threshold.
* **JSONL protocol** over a passed file descriptor — clean, explicit multi-turn message framing.
* **Hallucination guard** — if the model attempts to call a tool not in the allowlist, the request falls back to a standard completion rather than executing.

### Sandboxed Execution (Docker)

* **Persistent container (`tantalum-persistent`)** — acts as a volume anchor and long-term storage layer; does not execute user code.
* **Ephemeral session containers (`tantalum-session-{uuid}`)** — one per tool invocation, `--network none`, memory- and CPU-capped, removed immediately after use. Persistent storage is mounted read-only.

This enforces:

* No cross-session write access from execution containers.
* No network egress from tool execution.
* Bounded resource usage per invocation.
* Clear blast radius — a compromised session container cannot affect the host or other sessions.

---

## Prerequisites

* Linux or WSL2 (Windows Subsystem for Linux)
* **x86_64 Architecture** (ARM64/Apple Silicon is not supported in v0.1.0)
* Python 3.11+
* Docker or Podman
* Ollama (for local inference) or an API key for a cloud provider
* Zig 0.15.2 (development builds only — `install_dev.sh` handles this automatically)

---

## Installation

```bash
git clone https://github.com/skystarry-ai/tantalum.git
cd tantalum
bash install.sh
pipx install --include-deps .
```

`install.sh` handles:

* Docker installation and group setup
* Python dependencies (`litellm`, `chromadb`, `rich`, `prompt_toolkit`, `python-dotenv`)
* Docker volume creation (`tantalum-storage`)
* Docker image pull (`python:3.12-slim`)
* Configuration file creation at `~/.config/tantalum/config.env`

After installation, edit the configuration file and add your API key, then run `tantalum` to start.

### Development Install

Builds the Zig gateway from source and installs in editable mode:

```bash
bash install_dev.sh
pipx install -e --include-deps .
```

`install_dev.sh` additionally handles:

* Zig 0.15.2 installation (`x86_64` and `aarch64` supported)
* Gateway build: `backend/zig-out/bin/tantalum-gateway`

---

## Configuration

The installer creates a configuration file at `~/.config/tantalum/config.env`. Edit it to set your provider:

```env
# LLM provider — set one API key
GEMINI_API_KEY=your_key
ANTHROPIC_API_KEY=your_key
OPENAI_API_KEY=your_key

# Model selection (default: gemini/gemini-2.5-flash)
# TANTALUM_MODEL=gemini/gemini-2.5-flash
# TANTALUM_MODEL=claude-haiku-4-5
# TANTALUM_MODEL=ollama/llama3

# Optional overrides
# TANTALUM_DOCKER_MEMORY=256m
# TANTALUM_DOCKER_CPUS=0.5
# TANTALUM_DOCKER_TIMEOUT=30
# TANTALUM_VOLUME=tantalum-storage
# TANTALUM_DOCKER_IMAGE=python:3.12-slim
# TANTALUM_SYSTEM_PROMPT=You are a helpful assistant.
```

You can also edit this file at any time from within the CLI using the `/config` slash command.

For local inference:
```bash
ollama pull llama3.2:3b
# then set: TANTALUM_MODEL=ollama/llama3.2:3b
```

> [!NOTE]  
> Local model support depends on LiteLLM's built-in compatibility. Tool calling quality may vary significantly depending on the model.

---

## Usage

```bash
tantalum
```

This launches the Zig gateway, the Python brain, and the interactive CLI. All subprocesses are cleaned up automatically on exit.

### Slash Commands

Slash commands are available via autocomplete (type `/` to see the list). Commands are defined in `frontend/commands.toml` — adding a new `[[command]]` block there is sufficient to register it in the autocomplete menu.

| Command | Description |
|---|---|
| `/config` | Open `~/.config/tantalum/config.env` to edit provider and model settings |
| `/newchat` | Clear conversation history and start a new session |

---

## Available Tools

| Tool | Description | Execution Context |
|---|---|---|
| `run_python` | Execute a Python code snippet | Session container |
| `run_shell` | Execute a shell command | Session container |
| `read_file` | Read a file from `/storage` | Persistent container |
| `write_file` | Write a file to `/storage` | Persistent container |

All tools are explicitly allowlisted. The model cannot invoke any tool outside this registry — attempting to do so triggers an automatic fallback to a plain completion.

Tool injection is query-scoped: ChromaDB selects the top-K most semantically relevant tools per message rather than including the full list every time.

---

## Security Model

Tantalum assumes authenticated users (SPA-verified) and enforces isolation at the runtime layer.

**Network**
* The gateway is port-knocking protected via HMAC-SHA256 SPA.
* Session containers run with `--network none` — no outbound connections are possible from tool execution.
* The Python brain communicates exclusively over a passed file descriptor; it never listens on a port.

**Filesystem**
* Session containers mount the persistent volume read-only (`/storage:ro`).
* Write access to the persistent volume is restricted to the persistent container.

**Resources**
* Per-container memory cap: `256m` (configurable).
* Per-container CPU cap: `0.5` cores (configurable).
* Execution timeout: `30s` (configurable).

**Isolation**
* Fresh container per tool call — no shared process state between invocations.
* No autonomous background execution.
* No model-initiated network activity.

---

## Project Structure

```
.
├── LICENSE
├── README.md
├── SECURITY.md
├── backend/
│   ├── build.zig
│   ├── build.zig.zon
│   └── src/
│       └── main.zig
├── frontend/
│   ├── __main__.py
│   ├── cli.py
│   └── commands.toml
├── install.sh
├── install_dev.sh
├── pyproject.toml
├── service/
│   └── tantalum-gateway
└── setup.py
```

---

## License

Apache-2.0 — Copyright 2026 Skystarry-AI
