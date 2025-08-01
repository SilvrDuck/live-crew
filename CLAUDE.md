# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**live-crew** is a low-latency, slice-based orchestration layer for running multiple CrewAI crews concurrently over real-time event streams. It adds deterministic timing, global shared context, replay capabilities, and NATS transport while preserving CrewAI's agent/task abstractions.

## Development Guidelines

**In case of doubt or conflicts**: Use user input as truth, then refer to `.vibes/live_crew_spec.md`. Never make assumptions or invent approaches. If multiple valid solutions exist, ask the user for their preferred approach.

**ALWAYS refer back to `.vibes/live_crew_spec.md` when making any decisions on architecture, structure, models, etc. The specifications are well-defined and must be followed.**

**Spec Evolution**: Respect the spec file, but if something seems not to make sense anymore or conflicts arise, raise it to the user so they can decide if the spec should be updated.

**Continuous Improvement**: The user will critique your approach often. When their feedback seems relevant for future work, update this CLAUDE.md file to incorporate the lessons learned. This helps improve development practices over time.

**Coding Rules**:
- Never start coding without explicit user approval
- Always ask clarification questions first until requirements are extremely clear and defined
- Code analysis is always permitted without asking
- When user requirements are unclear, especially regarding architecture, ask for clarification rather than making decisions
- **Use `.vibes/scrum.md` to track progress and avoid doing unplanned work** - all development should align with defined milestones and tasks
- **Never use single-letter variable names** except in very specific cases:
  - `i`, `j`, `k` for simple loop counters
  - `x`, `y`, `z` for mathematical coordinates/operations
  - `e` for caught exceptions in short exception handlers
  - All other variables must have descriptive names that explain their purpose

**Preferred Development Approach**:
- Prefer a functional data/domain driven approach rather than OOP
- Use DRY principles, except for config and testing where DAMP is preferred

**Test-Driven Development Protocol**:
- **Always write tests first**: Before implementing any functionality, create comprehensive tests that define the expected behavior
- **Validate tests before presenting**: Always run linting (ruff) and type checking (ty) on test files before presenting to user
- **Handle expected type errors**: Add `# type: ignore` comments with explanation for intentionally invalid test cases (e.g., testing ValidationError with wrong types)
- **Present tests for validation**: Show the test cases to the user and get approval before writing the implementation code
- **Iterative TDD process**:
  1. Start with simple failing tests and minimal implementation
  2. Make tests pass with simplest possible code
  3. Add more complex/challenging tests that fail
  4. Enhance implementation to make new tests pass
  5. Repeat: progressively add harder tests, improve implementation
  6. Refactor when needed while keeping all tests green
- **Test-first workflow**:
  1. Analyze requirements and create initial simple test cases
  2. Run `uvx ruff check` and `uvx ty check` on test files
  3. Fix any linting/type issues (add `# type: ignore` for expected test errors)
  4. Present tests to user for validation
  5. Get user approval
  6. **Verify tests fail first**: Run new tests immediately after writing them to confirm they fail as expected (red phase)
  7. Implement minimal code to make tests pass
  8. **Always validate before responding**: Run `uvx ruff check` and `uvx ty check` on all code before talking back to user
  9. Run full test suite to ensure tests pass (green phase)
  10. **Iterate**: Add more challenging tests, enhance implementation, repeat steps 6-9
  11. Refactor if needed while keeping tests green
- **Comprehensive test coverage**: Include happy path, edge cases, error conditions, and validation scenarios

## External Library Documentation Protocol
1. **ALWAYS check .vibes/references folder first**: Look for `.vibes/references/<library_name>.md` file before making any library-related decisions
2. **If file exists**: Read it thoroughly and use the documented patterns and best practices
3. **If information is sufficient**: Use the existing documentation - don't reinvent patterns
4. **If file missing or incomplete**:
   - Research the library online using WebFetch/WebSearch
   - Create new `.vibes/references/<library_name>.md` file OR update existing file with the new information
   - Structure the file with clear sections (installation, key concepts, code examples, common patterns, best practices)
   - Then use the information for your implementation
5. **Always maintain .vibes/references**: Keep library documentation files updated and comprehensive for future use
6. **CRITICAL**: Never make library usage decisions (like ConfigDict vs Config) without first checking or creating reference documentation

## Lessons Learned
- Always use the `.vibes/references` folder for storing and referencing configuration and utility modules
- Utilize existing utility classes like `ConfigDict` when appropriate
- Carefully review and leverage existing project infrastructure and reference materials before implementing new solutions
- Performance matters: Avoid creating unnecessary string objects in high-frequency validation
- Error messages should include context like file paths for better debugging
- TypeVar usage should match actual usage patterns (invariant vs covariant)

## Package Management
- **ALWAYS use `uv` for all Python operations** - This project uses uv as the package manager and virtual environment tool
- **Never use pip, pipenv, poetry, or other tools** - Only use uv commands:
  - `uv sync --dev` - Install dependencies and dev dependencies
  - `uv run <command>` - Run commands in the uv environment
  - `uv add <package>` - Add new dependencies
  - `uv remove <package>` - Remove dependencies
- **Examples**:
  - Running tests: `uv run pytest`
  - Running linting: `uv run ruff check`
  - Running pre-commit: `uv run pre-commit run --all-files`
  - Installing new packages: `uv add requests` or `uv add --dev pytest-mock`

## Memory Notes
- make sure that you always udpate .vibes/references with fresh info you just got from the internet, do not make up everything, even if you are sure
- pydantic has a shit ton of specific types and validators. look up online for the ones you need rather than trying to implement yours
- aggressively google stuff up (and update the corresponding .vibes/references afterwards)
- **ALWAYS use uv commands** - never forget this project uses uv for package management
- don't forget to update the milestones as well when you update your scrum file
