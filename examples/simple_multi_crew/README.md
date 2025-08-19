# Live-Crew Content Moderation Examples

Real-world social media content moderation using live-crew's CrewAI integration with meaningful crew dependencies and context sharing.

## What This Example Demonstrates

- **🤖 Real CrewAI Integration**: Native @CrewBase crews with agents, tasks, and processes
- **🔄 Multi-Crew Dependencies**: Action crew waits for Analysis crew completion
- **📊 Event-Driven Processing**: Messages trigger moderation pipeline
- **🗃️ Context Sharing**: Analysis results shared via KV backend between crews
- **⚙️ YAML Configuration**: Declarative orchestration with proper dependencies
- **🎯 Real Business Logic**: Actual content moderation with spam detection and sentiment analysis

## Content Moderation Pipeline

This example implements a realistic social media moderation system:

**Flow**: `Message Posted → Analysis Crew → Context Storage → Action Crew → Moderation Decision`

### 1. Analysis Crew (Runs First)
- **Trigger**: `message_posted` events
- **Role**: Content analysis and risk assessment
- **Output**: Spam score, sentiment analysis, guideline compliance
- **Context**: Stores analysis results using `shared_` prefix for Action crew

### 2. Action Crew (Depends on Analysis)
- **Trigger**: `message_posted` events (same trigger, different execution order)
- **Dependencies**: Waits for Analysis crew completion
- **Role**: Takes moderation actions based on analysis results
- **Context**: Reads analysis results from shared context
- **Output**: Moderation decision (approve/flag/remove) with user message

## Examples Included

### 1. Python API Example (`python_api_example.py`)
Direct Python configuration:

```bash
# From the project root
python examples/hello_world/python_api_example.py
```

### 2. YAML Configuration Example (`yaml_config_example.py`)
YAML-based orchestration showing dependency configuration:

```bash
# From the project root
python examples/hello_world/yaml_config_example.py
```

## Expected Output

Both examples will:
1. **Load CrewAI Crews**: Real content analysis and moderation agents
2. **Process Messages**: 3 social media messages with varying risk levels
3. **Multi-Crew Pipeline**:
   - Analysis crew evaluates each message for spam, sentiment, compliance
   - Action crew waits for analysis, then makes moderation decisions
4. **Context Sharing**: Analysis results passed through KV backend to Action crew
5. **Generate Actions**: Structured moderation decisions with reasoning

## Files Structure

```
examples/hello_world/
├── python_api_example.py     # 🚀 Python API example
├── yaml_config_example.py    # 📄 YAML configuration example
├── events.json               # 📊 Social media messages (3 messages with varying risk)
├── orchestration.yaml        # ⚙️ Master orchestration config
├── crews/
│   ├── analysis/
│   │   ├── crew.py           # @CrewBase content analysis crew
│   │   └── runtime.yaml      # Triggers on message_posted, no dependencies
│   └── action/
│       ├── crew.py           # @CrewBase moderation action crew
│       └── runtime.yaml      # Depends on analysis crew completion
└── README.md                 # This documentation
```

## Key Concepts Demonstrated

### Content Analysis Crew
```python
@CrewBase
class AnalysisCrew:
    @agent
    def content_analyst(self) -> Agent:
        return Agent(
            role="Content Analyst",
            goal="Analyze social media messages for spam, sentiment, and guideline compliance",
            backstory="You're an experienced content moderator who can quickly identify problematic content while being fair to users.",
            verbose=True,
            allow_delegation=False
        )

    @task
    def analyze_message_task(self) -> Task:
        return Task(
            description="Analyze message for spam, sentiment, and compliance",
            expected_output="Analysis report with spam_score, sentiment, and violations",
            agent=self.content_analyst()
        )
```

### Context Sharing Between Crews
```python
# Analysis crew stores results with shared_ prefix
context["shared_analysis_result"] = {
    "spam_score": 8,
    "sentiment": "promotional",
    "guideline_violation": True,
    "reasoning": "Contains promotional link and urgency language"
}

# Action crew reads shared context
analysis = context.get("shared_analysis_result", {})
if analysis.get("spam_score", 0) > 7:
    return "remove"
```

### YAML Dependency Configuration
```yaml
# crews/action/runtime.yaml
crew: action_crew
triggers: [message_posted]
needs:
  - type: crew
    crew: analysis_crew
    offset: 0  # Wait for analysis crew in same slice
wait_policy: all  # Must wait for all dependencies
```

## Architecture Highlights

- **🎯 Thin CrewAI Wrapper**: Zero modifications to standard CrewAI crews
- **🔄 Real Dependencies**: Action crew truly depends on Analysis crew results
- **🗃️ KV Context Backend**: Shared state management with `shared_` prefix pattern
- **📋 Declarative Config**: YAML-first orchestration with meaningful dependencies
- **⚡ Native Performance**: Direct CrewAI execution with live-crew orchestration
- **🔗 Framework Portable**: CrewAI crews work independently of live-crew

## Sample Messages Processed

The demo includes three messages with different moderation scenarios:

1. **Alice's Message**: Promotional spam with external link - should be flagged/removed
2. **Bob's Message**: Positive welcoming message - should be approved
3. **Charlie's Message**: Community appreciation - should be approved

## Using as a Project Starter

1. **Copy the directory**: `cp -r examples/hello_world/ my-moderation-system/`
2. **Customize analysis**: Modify `crews/analysis/crew.py` with your content policies
3. **Customize actions**: Update `crews/action/crew.py` with your moderation rules
4. **Update events**: Replace `events.json` with your message feed
5. **Configure dependencies**: Adjust `runtime.yaml` files for your workflow
6. **Run**: `python my-moderation-system/python_api_example.py`

## Real-World Applications

This pattern can be extended for:
- **E-commerce**: Product review moderation
- **Social Media**: Community content filtering
- **Customer Support**: Message classification and routing
- **Content Publishing**: Editorial review workflows
- **Gaming**: Chat moderation and community management
