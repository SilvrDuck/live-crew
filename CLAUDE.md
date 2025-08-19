# CLAUDE.md

This file provides guidance to Claude Code when working with code in this repository.

## Project Overview

**live-crew** is a low-latency, slice-based orchestration layer for running multiple CrewAI crews concurrently over real-time event streams. It adds deterministic timing, global shared context, replay capabilities, and NATS transport while preserving CrewAI's agent/task abstractions.

## Core Principles

**Build vertical slices** that deliver complete user value quickly, then iterate based on real testing feedback.

### Decision Making Priority
1. **User input** (highest priority)
2. **Spec guidance** from `.vibes/live_crew_spec.md`
3. **Current slice goals** from `.vibes/scrum.md`
4. **Real testing feedback**

## Development Protocol

### Before Any Implementation
1. **Ask clarification questions** until requirements are crystal clear
2. **Get explicit user approval** for the specific task
3. **Never make assumptions** - ask for confirmation when uncertain

### Test-Driven Development
1. **Write tests first** - define expected behavior before implementation
2. **Present tests to user** for validation and approval
3. **Verify tests fail** (red phase)
4. **Write minimal code** to make tests pass (green phase)
5. **Refactor** while keeping tests green
6. **Always validate** with `uv run -- ruff check` and `uv run -- ty check`

### Implementation Rules
- **Code analysis is always permitted** without asking
- **Focus on current slice** - avoid building for imagined future needs
- **Minimal viable implementation** that can be extended later
- **Modern Python patterns** (3.13+): `T | None`, `list[Any]`
- **Descriptive variable names** (no single letters except `i,j,k` for loops, `x,y,z` for math, `e` for exceptions)

## External Libraries

🚨 **CRITICAL**: Research ALL libraries before use!

1. **Check `.vibes/references/<library>.md` first**
2. **If missing/incomplete**: Research online and create/update reference file
3. **Never implement without documentation**

## Package Management

**ALWAYS use `uv`** - never pip, pipenv, or poetry:
- `uv sync --dev` - Install dependencies
- `uv run <command>` - Run commands
- `uv add <package>` - Add dependencies
- `uv run -- pytest` - Run tests
- `uv run -- ruff check` - Linting
- `uv run -- ty check` - Type checking

## Documentation

- **Meaningful docstrings** explaining purpose and behavior
- **No sprint/slice references** in code comments
- **Document limitations** when implementations are simplified
- **Keep development state in scrum.md** only

## Architecture

- **Vertical slices over horizontal layers**
- **Functional/domain-driven** rather than OOP when possible
- **Extensible design** without premature optimization
- **No relative imports** - use complete paths
- **No backward compatibility** during initial development
