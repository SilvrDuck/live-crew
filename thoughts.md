# The Vibes Report

_The state of vibe coding in August 2025_

## Agenda

- Present my latest attempt at feeling the vibes
- Discuss what it does well and what it does terribly
- My thoughts on professionnal use
- Open discussion

## Vibe coding, how to?

### Showcase of the project

-> Demo

### Planning phase

- Plan, plan, plan. LLMs code super fast, but are super dumb. Spend 90% of your discussion planning.
- Use a thinking, web connected model, to plan. E.g. o3 (or now gpt5).
- Treat it like a design discussion. Insist on it doing research, do your reaserch, use different models -> specs.md
- Make it create a snippet.md bible
- Make it build a scrum.md plan.

### The coding phase

- Make a strong CLAUDE.md / copilot-instructions.md (use LLMs to help you).
    - use internet, build up doc
    - TDD
    - specify tools
    - make it ask you a lot of things, wait for crystal clear
    - RTFM
- Make a .vibes folder (my idea, proud)
    - running doc
    - scrum.md
    - specs.md
    - snippets.md (tell it to refer to it in the other docs)
    - ...
- Really thought of them as your junior engineer. Can do great things with precise tasks.
- Use your modes, use `#`
- Do the smallest thing yourself: if you know exactly how to do it, it will be faster and less frustrating
- Refer a lot to your .vibes folder
- Agents are goated -> see my list
- Stay close to the code. Basically a fancy multicursor tool.
- drop pics in the chat
- Ask it to repeat your favorite instruction at the end of each message #hack
- Use the power of PEP 582, or rather of a local .venv and tell it to actively browse
- You are the scrum master
- Enjoy your work, do the stuff you like
- use the hard tools, like ty or ruff

### The reviews

- Use external LLMs to validate the work
- Use drastic agents (like radical simplicity engineer)
- Make it use the `gh` command! Can create PR and then read your comments.
- Ask it first what a good _ _ _ is, then ask it to review as a _ _ _

## How did it do?

### The bad

- Many public cases about security. You are the senior engineer, remember that.
- Extremely abstract protocol based infra. Nifty, but overkill
- YOU ARE ABSOLUTELY RIGHT. Make it ask to a critical agent
- You very quickly lose track, even if you «read». Then you’re stuck in prompting hell. My rule: if remprompting once fails, look at it yourself.

### The good

- Three days of work, and the project looks like it is real (and sort of is)
- You can do something else at the same time, even vibe code another thing
- Able to «code» on my phone, in the car
- A usable prototype at least

## Should we do it at work?

### Excellent use cases

- Document code. Caveat: it will explain what it sees, _unless_ you tell it first to explore unclear patterns and give your explanations to it
- Maintain coherence between hard and squishy text -> can even be a quality gate
- Do advanced git stuff (amend, squash, rebase, write up proper messages, etc)
- Gather wider context for a new feature. Use it as a brainstorm tool once you have a plan.
- Boiler plate and pattern matching (route creation, pydantic models, etc). Warning: it over does it a lot.
- Backend work
- Quickly mock missing functionality. Tell it to mark the spot for easy removal. (e.g. frontend functionality while working on backend).
- Build test cases. Warning, it will quickly test if pydantic actually checks its inputs. Ask it for a list of weird edge cases and add your own. -> showcase tests

### Nope.

- Full vibe-coding -> too much weird stuff for now
- Architecture is terrible
- This empty feeling inside your chest

## What do you guys think?

## Next time:

- faire une démo
