# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**live-crew** is a low-latency, slice-based orchestration layer for running multiple CrewAI crews concurrently over real-time event streams. It adds deterministic timing, global shared context, replay capabilities, and NATS transport while preserving CrewAI's agent/task abstractions.

## Development Guidelines

## Development Philosophy: Vertical Slices & Rapid Testing

**Core Principle**: Build **vertical slices** that deliver complete user value quickly, then iterate based on real testing feedback.

### Spec File Evolution Strategy

- **`.vibes/live_crew_spec.md` should be RESPECTED** - follow the spec unless user explicitly tells you to change it
- **BE PROACTIVE with spec suggestions** - if you see something that makes more sense, suggest changes to user
- **ALWAYS ask user to confirm spec changes** - never modify the spec file without explicit user approval
- **Spec provides guidance and boundaries** - use it to understand the intended architecture and design

### Vertical Slice Approach

- **Deliver working software every 2-4 weeks** via complete vertical slices
- **Test the app as soon as possible** - get user feedback on real working software
- **Implement only what's needed right now** - avoid over-engineering for imagined future needs
- **Keep extensibility in mind** but don't pre-build flexibility until proven necessary
- **Each slice should work end-to-end** from user input to visible output

### Decision Making Priority

1. **User input as truth** (highest priority)
2. **Spec guidance** from `.vibes/live_crew_spec.md` (respect unless told otherwise)
3. **Current vertical slice goals** from `.vibes/scrum.md`
4. **Real testing feedback** for suggesting improvements

### Requirements Clarification Protocol

- **ALWAYS ask clarification questions first** until requirements are extremely clear and defined
- **Never make assumptions** about what the user wants - get explicit confirmation
- **Be proactive with suggestions** but always ask for user approval before proceeding
- **In case of conflicts or ambiguity**: Stop and ask user for clarification rather than guessing

### Coding Rules

- **Never start coding without explicit user approval** for the specific task
- **Always ask clarification questions first** until requirements are extremely clear and defined
- **Test the app early and often** - working software over comprehensive documentation
- **Add features incrementally** - start minimal, extend based on actual needs
- **Focus on current slice** - avoid building for future slices until needed
- **Code analysis is always permitted** without asking
- **Use `.vibes/scrum.md` to track progress** and stay focused on current vertical slice
- **Never use single-letter variable names** except in very specific cases:
    - `i`, `j`, `k` for simple loop counters
    - `x`, `y`, `z` for mathematical coordinates/operations
    - `e` for caught exceptions in short exception handlers
    - All other variables must have descriptive names that explain their purpose
- Use modern python patterns (3.13+), e.g. `T | None` for optional types and `list[Any]` instead of `List[Any]`.

### Documentation Rules

- **Write meaningful docstrings** - help developers understand code without scrum/sprint context
- **Avoid sprint/slice references** in docstrings (e.g., "Slice 1", "Sprint 2") - meaningless to future developers
- **Explain purpose and behavior** - focus on what the code does, why it exists, and how to use it
- **Document limitations clearly** - explain when simple implementations might need enhancement
- **Use concrete examples** in docstrings when helpful
- **Comments should be evergreen** - avoid references to development state, sprints, or temporary implementation phases
- **Keep development state in scrum.md** - development progress, sprint goals, and temporary decisions belong in scrum.md, not in code comments

### Preferred Development Approach

- **Vertical slices over horizontal layers** - complete features over perfect architecture
- **Functional data/domain driven approach** rather than OOP
- **Minimal viable implementation** that can be extended later
- **DRY principles**, except for config and testing where DAMP is preferred
- **Extensible design without premature optimization** - design for change, but don't build what isn't needed

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

🚨 **ABSOLUTELY CRITICAL**: Research ALL libraries before use - no exceptions!

1. **ALWAYS check .vibes/references folder first**: Look for `.vibes/references/<library_name>.md` file before making any library-related decisions
2. **If file exists**: Read it thoroughly and use the documented patterns and best practices
3. **If information is sufficient**: Use the existing documentation - don't reinvent patterns
4. **If file missing or incomplete**:

   - **MANDATORY**: Research the library online using WebFetch/WebSearch for latest best practices
   - Create new `.vibes/references/<library_name>.md` file OR update existing file with the new information
   - Structure the file with clear sections (installation, key concepts, code examples, common patterns, best practices, current limitations)
   - Include 2024-2025 specific patterns and recent updates. Adjust snippets to modern python patterns (3.13+).
   - Then use the information for your implementation
5. **Always maintain .vibes/references**: Keep library documentation files updated and comprehensive for future use
6. **CRITICAL**: Never make library usage decisions without first checking or creating reference documentation
7. **RESEARCH REQUIREMENT**: For ANY library usage (even well-known ones), you MUST research current best practices online and document them in .vibes/references BEFORE implementing
8. **UPDATE MANDATE**: If using a library that has existing documentation but seems outdated, research and update the file with latest patterns

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
- don't forget to update the slice section as well as the sprint section when you update your scrum file
- use your agents very proactively
- For scrum.md management:
    - Each slice in `scrum.md` should represent a complete, end-to-end functionality milestone
    - Within each slice, individual items can be considered sprints - these are testable, specific features
    - Consistently maintain and use `scrum.md` as a living document to track project progress
    - Ensure that `scrum.md` reflects the current development focus and provides clear, actionable goals for each vertical slice
- We are still in initial dev, do not maintain backward compatibility. Do not use relative imports. Do not declare imports in __init__ unless they are for the final user (for internal stuff, use complete path)
