"""Action CrewAI crew for content moderation actions."""

from crewai import Agent, Crew, Task, Process
from crewai.project import CrewBase, agent, task, crew


@CrewBase
class ActionCrew:
    """CrewAI crew for taking moderation actions based on analysis."""

    @agent
    def moderator_agent(self) -> Agent:
        return Agent(
            role="Content Moderator",
            goal="Take appropriate moderation actions based on content analysis results",
            backstory="You're a fair but firm community moderator who takes action based on analysis results. You can approve content, flag for review, or remove violations while being transparent with users.",
            verbose=True,
            allow_delegation=False,
        )

    @task
    def moderation_action_task(self) -> Task:
        return Task(
            description="Based on the analysis crew's results stored in context:\n1. Read analysis results (spam_score, sentiment, guideline_violation)\n2. Decide on appropriate action (approve/flag/remove)\n3. Generate user-facing response with reasoning\n4. Log moderation decision",
            expected_output="Moderation decision with action taken (approve/flag/remove) and user message",
            agent=self.moderator_agent(),
        )

    @crew
    def crew(self) -> Crew:
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
        )
