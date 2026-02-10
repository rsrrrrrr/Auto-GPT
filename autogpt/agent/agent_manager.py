"""Agent manager for managing GPT agents"""
from __future__ import annotations

import re
from typing import List

from autogpt.config.config import Config
from autogpt.llm import Message, create_chat_completion
from autogpt.singleton import Singleton


class AgentManager(metaclass=Singleton):
    """Agent manager for managing GPT agents"""

    def __init__(self):
        self.next_key = 0
        self.agents = {}  # key, (task, full_message_history, model)
        self.cfg = Config()

    # Create new GPT agent
    # TODO: Centralise use of create_chat_completion() to globally enforce token limit

    def create_agent(self, task: str, prompt: str, model: str) -> tuple[int, str]:
        """Create a new agent and return its key

        Args:
            task: The task to perform
            prompt: The prompt to use
            model: The model to use

        Returns:
            The key of the new agent
        """
        messages: List[Message] = [
            {"role": "user", "content": prompt},
        ]
        for plugin in self.cfg.plugins:
            if not plugin.can_handle_pre_instruction():
                continue
            if plugin_messages := plugin.pre_instruction(messages):
                messages.extend(iter(plugin_messages))
        # Start GPT instance
        agent_reply = create_chat_completion(
            model=model,
            messages=messages,
        )

        messages.append({"role": "assistant", "content": agent_reply})

        plugins_reply = ""
        for i, plugin in enumerate(self.cfg.plugins):
            if not plugin.can_handle_on_instruction():
                continue
            if plugin_result := plugin.on_instruction(messages):
                sep = "\n" if i else ""
                plugins_reply = f"{plugins_reply}{sep}{plugin_result}"

        if plugins_reply and plugins_reply != "":
            messages.append({"role": "assistant", "content": plugins_reply})
        key = self.next_key
        # This is done instead of len(agents) to make keys unique even if agents
        # are deleted
        self.next_key += 1

        self.agents[key] = (task, messages, model)

        for plugin in self.cfg.plugins:
            if not plugin.can_handle_post_instruction():
                continue
            agent_reply = plugin.post_instruction(agent_reply)

        return key, agent_reply

    def message_agent(self, key: str | int, message: str) -> str:
        """Send a message to an agent and return its response

        Args:
            key: The key of the agent to message
            message: The message to send to the agent

        Returns:
            The agent's response
        """
        task, messages, model = self.agents[int(key)]

        # Add user message to message history before sending to agent
        messages.append({"role": "user", "content": message})

        for plugin in self.cfg.plugins:
            if not plugin.can_handle_pre_instruction():
                continue
            if plugin_messages := plugin.pre_instruction(messages):
                for plugin_message in plugin_messages:
                    messages.append(plugin_message)

        # Start GPT instance
        agent_reply = create_chat_completion(
            model=model,
            messages=messages,
        )

        messages.append({"role": "assistant", "content": agent_reply})

        plugins_reply = agent_reply
        for i, plugin in enumerate(self.cfg.plugins):
            if not plugin.can_handle_on_instruction():
                continue
            if plugin_result := plugin.on_instruction(messages):
                sep = "\n" if i else ""
                plugins_reply = f"{plugins_reply}{sep}{plugin_result}"
        # Update full message history
        if plugins_reply and plugins_reply != "":
            messages.append({"role": "assistant", "content": plugins_reply})

        for plugin in self.cfg.plugins:
            if not plugin.can_handle_post_instruction():
                continue
            agent_reply = plugin.post_instruction(agent_reply)

        return agent_reply

    def list_agents(self) -> list[tuple[str | int, str]]:
        """Return a list of all agents

        Returns:
            A list of tuples of the form (key, task)
        """

        # Return a list of agent keys and their tasks
        return [(key, task) for key, (task, _, _) in self.agents.items()]

    def delete_agent(self, key: str | int) -> bool:
        """Delete an agent from the agent manager

        Args:
            key: The key of the agent to delete

        Returns:
            True if successful, False otherwise
        """

        try:
            del self.agents[int(key)]
            return True
        except KeyError:
            return False

    @staticmethod
    def _clean_output(text: str) -> str:
        """Normalize model output so it can be reused as stable context."""
        collapsed_whitespace = " ".join(text.split())
        return re.sub(r"[^\w\s.,!?\-:;()]", "", collapsed_whitespace).strip()

    def run_three_agent_cycle(self, task: str, model: str) -> dict[str, str | int]:
        """Run an end-to-end loop with 3 specialized agents.

        The workflow is:
        1) Planner agent creates an execution plan for the task.
        2) Executor agent produces a first-pass result from the plan.
        3) Reviewer agent cleans and critiques the first-pass result.
        4) Executor agent performs a second inference pass using clean output.
        """

        planner_prompt = (
            "You are the planning agent. Produce a concise, numbered plan to "
            f"complete this mission: {task}"
        )
        planner_key, plan = self.create_agent(
            task=f"planner:{task}",
            prompt=planner_prompt,
            model=model,
        )

        executor_prompt = (
            "You are the execution agent. Use this plan to complete the task and "
            f"return the mission result:\n{plan}"
        )
        executor_key, first_result = self.create_agent(
            task=f"executor:{task}",
            prompt=executor_prompt,
            model=model,
        )

        reviewer_prompt = (
            "You are the review agent. Clean the output, remove noise, and point out "
            "critical improvements.\n"
            f"Task: {task}\n"
            f"First output: {first_result}"
        )
        reviewer_key, review = self.create_agent(
            task=f"reviewer:{task}",
            prompt=reviewer_prompt,
            model=model,
        )

        clean_output = self._clean_output(review)
        refined_result = self.message_agent(
            executor_key,
            (
                "Use this cleaned review to perform a second inference pass and "
                f"return the final result only:\n{clean_output}"
            ),
        )

        return {
            "planner_agent_id": planner_key,
            "executor_agent_id": executor_key,
            "reviewer_agent_id": reviewer_key,
            "plan": plan,
            "first_result": first_result,
            "clean_output": clean_output,
            "refined_result": refined_result,
        }
