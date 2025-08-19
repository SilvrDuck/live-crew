"""Content Analysis CrewAI crew for social media moderation."""

from crewai import Agent, Crew, Task, Process
from crewai.project import CrewBase, agent, task, crew


@CrewBase
class AnalysisCrew:
    """CrewAI crew for content analysis and moderation."""

    @agent
    def content_analyst(self) -> Agent:
        return Agent(
            role="Content Analyst",
            goal="Analyze social media messages for spam, sentiment, and guideline compliance",
            backstory="You're an experienced content moderator who can quickly identify problematic content while being fair to users. You analyze messages for spam detection, sentiment analysis, and community guideline violations.",
            verbose=True,
            allow_delegation=False,
        )

    @task
    def analyze_message_task(self) -> Task:
        return Task(
            description="Analyze the message content for:\n1. Spam detection (promotional links, repetitive content)\n2. Sentiment analysis (positive/negative/neutral)\n3. Community guideline compliance\n4. Store analysis results in context for action crew",
            expected_output="Analysis report with spam_score (0-10), sentiment, guideline_violation (true/false), and reasoning",
            agent=self.content_analyst(),
        )

    @crew
    def crew(self) -> Crew:
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
        )
