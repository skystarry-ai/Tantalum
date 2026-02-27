# Security Policy

## Overview

Tantalum is designed as a **controlled execution runtime**, not an autonomous agent framework.
Its security model is based on:

* Explicit human authorization
* Network isolation by default
* Docker-based execution sandboxing
* Clear trust boundaries between components

Security is a first-class architectural concern in this project.

---

## Supported Versions

Only the latest tagged release is considered supported for security updates.

Development branches, forks, or modified builds are not covered by this policy.

---

## Reporting a Vulnerability

If you discover a security vulnerability in the canonical Tantalum implementation:

1. **Do not open a public GitHub issue.**
2. Contact the maintainer privately via:

   * Email: security@skystarry.xyz
   * Or GitHub security advisory feature (if enabled)

Please include:

* A clear description of the issue
* Steps to reproduce
* A minimal proof of concept (if possible)
* A description of potential impact

You will receive acknowledgment within a reasonable timeframe.

---

## Security Scope

This policy covers vulnerabilities related to:

* Zig SPA gateway implementation
* File descriptor passing via Unix Domain Sockets
* Authentication boundary enforcement
* Docker isolation logic
* Tool execution boundary enforcement
* Permission separation and usergroup handling

---

## Out of Scope

The following are **not considered vulnerabilities in Tantalum itself**:

* Issues arising from user-modified source code
* Behavior changes in forks or derivative projects
* Security risks introduced by:

  * Enabling network access in containers
  * Relaxing Docker permissions
  * Running as privileged/root containers
  * Modifying default isolation settings
* Vulnerabilities in:

  * Docker daemon
  * The operating system
  * External model providers
* Prompt injection or model misinterpretation where isolation boundaries remain intact

Tantalum assumes that users understand and accept responsibility when altering default security configurations.

---

## Security Design Principles

Tantalum follows these principles:

1. **Secure by default**

   * No external network access in session containers
   * No exposed brain ports
   * Local socket-based communication

2. **Human-in-the-loop authorization**

   * Destructive or sensitive actions require explicit user approval

3. **Clear trust boundaries**

   * Zig gateway handles external interaction
   * Brain handles reasoning only
   * Docker handles execution isolation

4. **Fail closed**

   * If boundaries are uncertain, execution should not proceed silently

---

## Responsible Disclosure

Security research is welcome and appreciated.

Please allow reasonable time for patching before public disclosure.

---

## Final Note

Tantalum is intentionally minimal and conservative in its autonomy model.

Security guarantees apply to the canonical, unmodified implementation operating under its documented default configuration.

Once modified, redistributed, or reconfigured, responsibility shifts to the operator of that derivative system.