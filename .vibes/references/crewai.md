# CrewAI Framework Reference

## Overview

CrewAI is a framework for orchestrating role-playing, autonomous AI agents. It enables collaborative intelligence where agents work together seamlessly to tackle complex tasks. CrewAI offers two primary paradigms: Crews (for autonomy and collaborative intelligence) and Flows (for granular, event-driven control).

## Architecture Philosophy

### Core Components
1. **Crew**: Top-level organization that coordinates agents and tasks
2. **AI Agents**: Specialized team members with defined roles, goals, and tools
3. **Process**: Workflow management (sequential, hierarchical, etc.)
4. **Tasks**: Individual assignments with specific outputs and agent assignments

### Two Primary Paradigms

#### Crews (Autonomy-Focused)
- Optimize for autonomy and collaborative intelligence
- Ideal for creative, exploratory tasks
- Agents have specific roles, tools, and goals
- Mimics organizational team dynamics
- Best for autonomous problem-solving

#### Flows (Control-Focused)
- Enable granular, event-driven control
- Provides structured, precise workflow orchestration
- Supports conditional logic and state management
- Allows fine-grained execution control
- Best for deterministic outcomes and auditable processes

## Configuration Approaches

### 1. YAML-Based Configuration (Recommended)

CrewAI strongly recommends YAML configuration for better maintainability, separation of concerns, and team collaboration.

#### Project Structure
```
my_project/
├── pyproject.toml
├── README.md
└── src/
    └── my_project/
        ├── __init__.py
        ├── main.py
        ├── crew.py
        ├── tools/
        │   ├── custom_tool.py
        │   └── __init__.py
        └── config/
            ├── agents.yaml
            └── tasks.yaml
```

#### CrewBase Class Pattern
```python
from crewai import Agent, Crew, Task, Process
from crewai.project import CrewBase, agent, task, crew, before_kickoff, after_kickoff

@CrewBase
class YourCrewName:
    """Description of your crew"""

    # Paths to YAML configuration files
    agents_config = 'config/agents.yaml'
    tasks_config = 'config/tasks.yaml'

    @agent
    def researcher(self) -> Agent:
        return Agent(
            config=self.agents_config['researcher'],
            verbose=True,
            tools=[SerperDevTool()]  # Tools must be defined in Python
        )

    @task
    def research_task(self) -> Task:
        return Task(
            config=self.tasks_config['research_task']
        )

    @crew
    def crew(self) -> Crew:
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True
        )
```

#### Agent YAML Configuration
```yaml
researcher:
  role: >
    {topic} Senior Data Researcher
  goal: >
    Uncover cutting-edge developments in {topic}
  backstory: >
    You're a seasoned researcher with a knack for uncovering the latest
    developments in {topic}. You have a keen eye for detail and a passion
    for uncovering hidden gems.
  tools:
    - serper_tool  # Must match @tool decorated function name
  llm: openai_llm  # Must match @llm decorated function name

reporting_analyst:
  role: >
    {topic} Reporting Analyst
  goal: >
    Create detailed reports based on {topic} research
  backstory: >
    You're a meticulous analyst with a gift for turning complex data into
    compelling narratives.
  allow_delegation: false
```

#### Task YAML Configuration
```yaml
research_task:
  description: >
    Conduct a thorough research about {topic}
    Make sure you find any interesting and relevant information.
  expected_output: >
    A list with 10 bullet points of the most relevant information about {topic}
  agent: researcher

reporting_task:
  description: >
    Review the context you got and expand each topic into a full section for a report.
    Make sure the report is detailed and contains any and all relevant information.
  expected_output: >
    A fully fledged report with the main topics, each with a full section of information.
    Formatted as markdown without '```'
  agent: reporting_analyst
  output_file: report.md
  create_directory: true
```

### 2. Direct Code Definition (Alternative)

For simpler projects or when you need more programmatic control:

```python
from crewai import Agent, Crew, Task, Process

# Direct agent definition
researcher = Agent(
    role="Senior Data Researcher",
    goal="Uncover cutting-edge developments in AI",
    backstory="You're a seasoned researcher...",
    verbose=True,
    allow_delegation=False,
    tools=[SerperDevTool()]
)

# Direct task definition
research_task = Task(
    description="Conduct thorough research about AI developments",
    expected_output="A list with 10 bullet points",
    agent=researcher
)

# Direct crew definition
crew = Crew(
    agents=[researcher],
    tasks=[research_task],
    process=Process.sequential,
    verbose=True
)
```

## Key Decorators

### @CrewBase
- Marks the class as a crew base class
- Enables automatic collection of agents and tasks
- Supports configuration file paths

### @agent
- Defines methods that return Agent objects
- Automatically collected by CrewBase
- Can reference YAML configuration via `config` parameter

### @task
- Defines methods that return Task objects
- Automatically collected by CrewBase
- Can reference YAML configuration via `config` parameter

### @crew
- Defines the method that creates the Crew object
- Assembles all agents and tasks
- Configures crew-level settings (process, verbose, etc.)

### Additional Decorators
- `@tool`: Defines tool objects for agents (must be defined in Python)
- `@llm`: Initializes Language Model objects (must be defined in Python)
- `@before_kickoff`: Preprocessing before crew execution
- `@after_kickoff`: Postprocessing after crew execution
- `@callback`: Event handling during execution
- `@output_json`: Structured JSON outputs
- `@output_pydantic`: Pydantic model outputs

## YAML Configuration Limitations

### What CAN be configured in YAML:
- Agent role, goal, backstory
- Task description, expected_output, agent assignment
- Output files and directory creation
- Basic agent settings (allow_delegation, verbose)
- References to tools and LLMs (by decorated function name)

### What MUST be defined in Python:
- Tool configurations and implementations
- LLM configurations and API settings
- Complex logic and conditional operations
- Dynamic data processing
- Custom callbacks and event handlers

### Tools and LLMs Pattern
```python
@CrewBase
class MyCrew:
    @tool
    def serper_tool(self):
        return SerperDevTool(api_key=os.getenv("SERPER_API_KEY"))

    @llm
    def openai_llm(self):
        return ChatOpenAI(
            model="gpt-4",
            temperature=0.1,
            api_key=os.getenv("OPENAI_API_KEY")
        )

    @agent
    def researcher(self) -> Agent:
        return Agent(
            config=self.agents_config['researcher'],  # References serper_tool and openai_llm
            tools=[self.serper_tool()],  # Also explicit assignment
            llm=self.openai_llm()
        )
```

## Dynamic Input Parameters

YAML configurations support variable interpolation:

```python
# main.py
inputs = {
    'topic': 'Open source AI agent frameworks',
    'current_year': str(datetime.now().year)
}

crew.kickoff(inputs=inputs)
```

Variables in YAML files (like `{topic}`) are automatically replaced with input values.

## Best Practices (2025)

### 1. Project Organization
- Use YAML configuration for larger, production projects
- Keep tools and LLMs in separate Python modules
- Use consistent naming conventions across YAML and Python

### 2. Performance Optimization
- Set `allow_delegation=False` for agents that don't need delegation
- Use structured outputs with Pydantic models for type safety
- Implement conditional task execution to reduce unnecessary work

### 3. Security and Maintainability
- Use environment variables for sensitive information (API keys)
- Separate configuration from implementation code
- Version control YAML files for easy change tracking

### 4. Team Collaboration
- YAML files can be edited by non-Python team members
- Clear separation allows domain experts to modify agent behavior
- Standardized format improves consistency across projects

### 5. Testing and Development
- Use `verbose=True` during development
- Implement proper error handling and logging
- Create reusable tool and LLM configurations

## When to Use Each Approach

### Choose YAML Configuration When:
- Building production applications
- Working with non-technical team members
- Need to maintain multiple similar crews
- Configuration changes are frequent
- Project complexity is medium to high

### Choose Direct Code Definition When:
- Building prototypes or simple scripts
- All configuration is static and simple
- Working solo or with only technical team members
- Need maximum programmatic control
- Project is small and unlikely to grow

### Hybrid Approach
Many projects benefit from combining both approaches:
- Use YAML for agent roles, goals, and task descriptions
- Use Python for complex tools, LLM configurations, and business logic
- This provides the best balance of maintainability and flexibility

## CrewAI vs Traditional Orchestration

CrewAI differentiates itself by:
- **Autonomous Intelligence**: Agents make decisions rather than just following scripts
- **Role-Playing**: Agents have personalities, goals, and backstories
- **Collaborative**: Agents can communicate and delegate to each other
- **Flexible Processes**: Support for sequential, hierarchical, and custom workflows
- **Tool Integration**: Easy integration with external APIs and services

This makes CrewAI ideal for creative, exploratory tasks where traditional workflow engines might be too rigid.
