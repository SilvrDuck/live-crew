# .vibes Folder

## What is this?

The `.vibes` folder is **Claude Code's scratchbook** - a structured workspace where the AI assistant takes notes, stores research, and learns project-specific patterns. Think of it as the AI's persistent memory and notebook for this codebase.

## Why does this exist?

AI assistants like Claude Code are stateless - they forget everything between conversations. The `.vibes` folder solves this by creating a persistent knowledge base where the AI can:

- **Take notes** on what it learns about the project
- **Store research** from googling libraries and best practices
- **Remember decisions** and architectural patterns
- **Track progress** with sprint planning and todo management
- **Follow user preferences** for coding style and approach
- **Stay focused** on planned features instead of wandering off

It's essentially giving the AI a brain that persists between conversations, plus a project manager to keep it on track.

## What's inside?

```
.vibes/
├── README.md                    # This explanation
├── live_crew_spec.md           # Project requirements (AI's assignment sheet)
├── scrum.md                    # Sprint planning (AI's project management)
└── references/                 # AI's research notes
    ├── pydantic.md            # Notes from googling Pydantic best practices
    ├── pydantic-settings.md   # Research on configuration management
    ├── pytest.md             # Testing patterns discovered
    ├── pyyaml.md             # YAML handling notes
    └── typing.md              # Python typing reference
```

## How Claude Code uses this

### Learning & Memory
- **Researches online** when encountering new libraries or patterns
- **Takes detailed notes** in `references/` folder after googling
- **Remembers user preferences** like "prefer functional over OOP"
- **Tracks what works** and what doesn't for this specific project

### Staying Focused
- **Follows the spec** in `live_crew_spec.md` to avoid feature creep
- **Updates progress** in `scrum.md` to track sprint goals
- **Maintains consistency** by referencing previous decisions

### User Customization
The user can **bias the AI's behavior** by:
- Adding preferences to `CLAUDE.md` (coding style, patterns to avoid)
- Updating specs when requirements change
- Providing feedback that gets incorporated into the scratchbook

## Why this matters

Without this folder, Claude Code would:
- ❌ Forget everything between sessions
- ❌ Make inconsistent decisions
- ❌ Reinvent solutions already researched
- ❌ Ignore user preferences

With `.vibes`, Claude Code:
- ✅ Maintains project context
- ✅ Builds on previous work
- ✅ Follows established patterns
- ✅ Gets smarter over time

## The Experiment

This is testing whether giving AI assistants a **persistent scratchbook** makes them:
- More consistent across sessions
- Better at learning project patterns
- More aligned with user preferences
- Capable of building institutional knowledge

Think of it as **AI pair programming** with memory.

## For Human Developers

While designed for AI, this pattern also helps humans:
- **Onboarding**: New devs can read the AI's notes to understand decisions
- **Knowledge capture**: Research and patterns don't get lost
- **Consistency**: Established patterns stay documented
- **Progress tracking**: Clear visibility into development status

## Try This Yourself

Want to experiment with AI-assisted development? Create your own `.vibes` folder:

1. **Add project specs** - Give the AI clear requirements
2. **Document preferences** - Tell the AI how you like to code
3. **Let it research** - Allow it to google and take notes
4. **Track progress** - Use sprint planning for focus
5. **Iterate** - Update the scratchbook as you learn

The goal is making AI assistants **project-aware** instead of just **code-aware**.

---

**Learn more**: [Claude Code](https://claude.ai/code) | **Fork this pattern**: Copy `.vibes/` to your repo
