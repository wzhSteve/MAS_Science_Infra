#!/usr/bin/env python
# coding=utf-8

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Modifications Copyright (c) 2025 Fulin Lin and the SupervisorAgent authors
# Prominent notice: This file has been modified from the original Hugging Face
# implementation to add runtime supervision features.

import importlib
import json
import os
import re
import tempfile
import textwrap
import time
import warnings
from abc import ABC, abstractmethod
from collections.abc import Callable, Generator
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from logging import getLogger
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Type, TypeAlias, TypedDict, Union, List, Dict, Optional, Tuple
import yaml
from huggingface_hub import create_repo, metadata_update, snapshot_download, upload_folder
from jinja2 import StrictUndefined, Template
from rich.console import Group
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.rule import Rule
from rich.text import Text


if TYPE_CHECKING:
    import PIL.Image

from .agent_types import AgentAudio, AgentImage, handle_agent_output_types
from .default_tools import TOOL_MAPPING, FinalAnswerTool
from .local_python_executor import BASE_BUILTIN_MODULES, LocalPythonExecutor, PythonExecutor, fix_final_answer_code
from .memory import (
    ActionStep,
    AgentMemory,
    CallbackRegistry,
    FinalAnswerStep,
    MemoryStep,
    PlanningStep,
    SystemPromptStep,
    TaskStep,
    Timing,
    ToolCall,
)
from smolagents.monitoring import TokenUsage
from .models import (
    CODEAGENT_RESPONSE_FORMAT,
    ChatMessage,
    ChatMessageStreamDelta,
    ChatMessageToolCall,
    MessageRole,
    Model,
    agglomerate_stream_deltas,
    parse_json_if_needed,
)
from .monitoring import (
    YELLOW_HEX,
    AgentLogger,
    LogLevel,
    Monitor,
)
from .remote_executors import DockerExecutor, E2BExecutor, WasmExecutor
from .tools import BaseTool, Tool, validate_tool_arguments
from .utils import (
    AgentError,
    AgentExecutionError,
    AgentGenerationError,
    AgentMaxStepsError,
    AgentParsingError,
    AgentToolCallError,
    AgentToolExecutionError,
    create_agent_gradio_app_template,
    extract_code_from_text,
    is_valid_name,
    make_init_file,
    parse_code_blobs,
    truncate_content,
)

from types import SimpleNamespace

logger = getLogger(__name__)

# Define a global dynamic memory data structure
from typing import TypedDict, List, Optional

# Record Agent single-step operation
class StepRecord(TypedDict):
    step_number: int
    step_task: str
    tool_call: str
    observation_summary: str # Only record a brief summary of the observation, not the full text
    timestamp: float # Use time.time()

# Record the entire session of an Agent from invocation to task completion, containing multiple StepRecords.
class Session(TypedDict):
    session_id: int
    agent_name: str
    parent_session_id: Optional[int]
    task: str
    steps: List[StepRecord]
    status: str # 'running', 'completed', 'failed'
    start_time: float # Session start time
    end_time: Optional[float] # Session end time

def get_variable_names(self, template: str) -> set[str]:
    pattern = re.compile(r"\{\{([^{}]+)\}\}")
    return {match.group(1).strip() for match in pattern.finditer(template)}


def populate_template(template: str, variables: dict[str, Any]) -> str:
    compiled_template = Template(template, undefined=StrictUndefined)
    try:
        return compiled_template.render(**variables)
    except Exception as e:
        raise Exception(f"Error during jinja template rendering: {type(e).__name__}: {e}")


@dataclass
class ActionOutput:
    output: Any
    is_final_answer: bool


@dataclass
class ToolOutput:
    id: str
    output: Any
    is_final_answer: bool
    observation: str
    tool_call: ToolCall


class PlanningPromptTemplate(TypedDict):
    """
    Prompt templates for the planning step.

    Args:
        plan (`str`): Initial plan prompt.
        update_plan_pre_messages (`str`): Update plan pre-messages prompt.
        update_plan_post_messages (`str`): Update plan post-messages prompt.
    """

    initial_plan: str
    update_plan_pre_messages: str
    update_plan_post_messages: str


class ManagedAgentPromptTemplate(TypedDict):
    """
    Prompt templates for the managed agent.

    Args:
        task (`str`): Task prompt.
        report (`str`): Report prompt.
    """

    task: str
    report: str


class FinalAnswerPromptTemplate(TypedDict):
    """
    Prompt templates for the final answer.

    Args:
        pre_messages (`str`): Pre-messages prompt.
        post_messages (`str`): Post-messages prompt.
    """

    pre_messages: str
    post_messages: str


class PromptTemplates(TypedDict):
    """
    Prompt templates for the agent.

    Args:
        system_prompt (`str`): System prompt.
        planning ([`~agents.PlanningPromptTemplate`]): Planning prompt templates.
        managed_agent ([`~agents.ManagedAgentPromptTemplate`]): Managed agent prompt templates.
        final_answer ([`~agents.FinalAnswerPromptTemplate`]): Final answer prompt templates.
    """

    system_prompt: str
    planning: PlanningPromptTemplate
    managed_agent: ManagedAgentPromptTemplate
    final_answer: FinalAnswerPromptTemplate


EMPTY_PROMPT_TEMPLATES = PromptTemplates(
    system_prompt="",
    planning=PlanningPromptTemplate(
        initial_plan="",
        update_plan_pre_messages="",
        update_plan_post_messages="",
    ),
    managed_agent=ManagedAgentPromptTemplate(task="", report=""),
    final_answer=FinalAnswerPromptTemplate(pre_messages="", post_messages=""),
)


@dataclass
class RunResult:
    """Holds extended information about an agent run.

    Attributes:
        output (Any | None): The final output of the agent run, if available.
        state (Literal["success", "max_steps_error"]): The final state of the agent after the run.
        steps (list[dict]): The agent's memory, as a list of steps.
        token_usage (TokenUsage | None): Count of tokens used during the run.
        timing (Timing): Timing details of the agent run: start time, end time, duration.
        messages (list[dict]): The agent's memory, as a list of messages.
            <Deprecated version="1.22.0">
            Parameter 'messages' is deprecated and will be removed in version 1.25. Please use 'steps' instead.
            </Deprecated>
    """

    output: Any | None
    state: Literal["success", "max_steps_error"]
    steps: list[dict]
    token_usage: TokenUsage | None
    timing: Timing

    def __init__(self, output=None, state=None, steps=None, token_usage=None, timing=None, messages=None):
        # Handle deprecated 'messages' parameter
        if messages is not None:
            if steps is not None:
                raise ValueError("Cannot specify both 'messages' and 'steps' parameters. Use 'steps' instead.")
            warnings.warn(
                "Parameter 'messages' is deprecated and will be removed in version 1.25. Please use 'steps' instead.",
                FutureWarning,
                stacklevel=2,
            )
            steps = messages

        # Initialize with dataclass fields
        self.output = output
        self.state = state
        self.steps = steps
        self.token_usage = token_usage
        self.timing = timing

    @property
    def messages(self):
        """Backward compatibility property that returns steps."""
        warnings.warn(
            "Parameter 'messages' is deprecated and will be removed in version 1.25. Please use 'steps' instead.",
            FutureWarning,
            stacklevel=2,
        )
        return self.steps

    def dict(self):
        return {
            "output": self.output,
            "state": self.state,
            "steps": self.steps,
            "token_usage": self.token_usage.dict() if self.token_usage is not None else None,
            "timing": self.timing.dict(),
        }


StreamEvent: TypeAlias = Union[
    ChatMessageStreamDelta,
    ChatMessageToolCall,
    ActionOutput,
    ToolCall,
    ToolOutput,
    PlanningStep,
    ActionStep,
    FinalAnswerStep,
]


class MultiStepAgent(ABC):
    """
    Agent class that solves the given task step by step, using the ReAct framework:
    While the objective is not reached, the agent will perform a cycle of action (given by the LLM) and observation (obtained from the environment).

    Args:
        tools (`list[Tool]`): [`Tool`]s that the agent can use.
        model (`Callable[[list[dict[str, str]]], ChatMessage]`): Model that will generate the agent's actions.
        prompt_templates ([`~agents.PromptTemplates`], *optional*): Prompt templates.
        instructions (`str`, *optional*): Custom instructions for the agent, will be inserted in the system prompt.
        max_steps (`int`, default `20`): Maximum number of steps the agent can take to solve the task.
        add_base_tools (`bool`, default `False`): Whether to add the base tools to the agent's tools.
        verbosity_level (`LogLevel`, default `LogLevel.INFO`): Level of verbosity of the agent's logs.
        managed_agents (`list`, *optional*): Managed agents that the agent can call.
        step_callbacks (`list[Callable]` | `dict[Type[MemoryStep], Callable | list[Callable]]`, *optional*): Callbacks that will be called at each step.
        planning_interval (`int`, *optional*): Interval at which the agent will run a planning step.
        name (`str`, *optional*): Necessary for a managed agent only - the name by which this agent can be called.
        description (`str`, *optional*): Necessary for a managed agent only - the description of this agent.
        provide_run_summary (`bool`, *optional*): Whether to provide a run summary when called as a managed agent.
        final_answer_checks (`list[Callable]`, *optional*): List of validation functions to run before accepting a final answer.
            Each function should:
            - Take the final answer and the agent's memory as arguments.
            - Return a boolean indicating whether the final answer is valid.
        return_full_result (`bool`, default `False`): Whether to return the full [`RunResult`] object or just the final answer output from the agent run.
    """

    def __init__(
        self,
        tools: list[Tool],
        model: Model,
        prompt_templates: PromptTemplates | None = None,
        instructions: str | None = None,
        max_steps: int = 20,
        add_base_tools: bool = False,
        verbosity_level: LogLevel = LogLevel.INFO,
        managed_agents: list | None = None,
        step_callbacks: list[Callable] | dict[Type[MemoryStep], Callable | list[Callable]] | None = None,
        planning_interval: int | None = None,
        name: str | None = None,
        description: str | None = None,
        provide_run_summary: bool = False,
        final_answer_checks: list[Callable] | None = None,
        return_full_result: bool = False,
        logger: AgentLogger | None = None,
    ):
        self.agent_name = self.__class__.__name__
        self.model = model
        self.prompt_templates = prompt_templates or EMPTY_PROMPT_TEMPLATES
        if prompt_templates is not None:
            missing_keys = set(EMPTY_PROMPT_TEMPLATES.keys()) - set(prompt_templates.keys())
            assert not missing_keys, (
                f"Some prompt templates are missing from your custom `prompt_templates`: {missing_keys}"
            )
            for key, value in EMPTY_PROMPT_TEMPLATES.items():
                if isinstance(value, dict):
                    for subkey in value.keys():
                        assert key in prompt_templates.keys() and (subkey in prompt_templates[key].keys()), (
                            f"Some prompt templates are missing from your custom `prompt_templates`: {subkey} under {key}"
                        )

        self.max_steps = max_steps
        self.step_number = 0
        self.planning_interval = planning_interval
        self.state: dict[str, Any] = {}
        self.name = self._validate_name(name)
        self.description = description
        self.provide_run_summary = provide_run_summary
        self.final_answer_checks = final_answer_checks if final_answer_checks is not None else []
        self.return_full_result = return_full_result
        self.instructions = instructions
        self._setup_managed_agents(managed_agents)
        self._setup_tools(tools, add_base_tools)
        self._validate_tools_and_managed_agents(tools, managed_agents)

        self.task: str | None = None
        self.memory = AgentMemory(self.system_prompt)

        if logger is None:
            self.logger = AgentLogger(level=verbosity_level)
        else:
            self.logger = logger

        self.monitor = Monitor(self.model, self.logger)
        self._setup_step_callbacks(step_callbacks)
        self.stream_outputs = False

    @property
    def system_prompt(self) -> str:
        return self.initialize_system_prompt()

    @system_prompt.setter
    def system_prompt(self, value: str):
        raise AttributeError(
            """The 'system_prompt' property is read-only. Use 'self.prompt_templates["system_prompt"]' instead."""
        )

    def _validate_name(self, name: str | None) -> str | None:
        if name is not None and not is_valid_name(name):
            raise ValueError(f"Agent name '{name}' must be a valid Python identifier and not a reserved keyword.")
        return name

    def _setup_managed_agents(self, managed_agents: list | None = None) -> None:
        """Setup managed agents with proper logging."""
        self.managed_agents = {}
        if managed_agents:
            assert all(agent.name and agent.description for agent in managed_agents), (
                "All managed agents need both a name and a description!"
            )
            self.managed_agents = {agent.name: agent for agent in managed_agents}
            # Ensure managed agents can be called as tools by the model: set their inputs and output_type
            for agent in self.managed_agents.values():
                agent.inputs = {
                    "task": {"type": "string", "description": "Long detailed description of the task."},
                    "additional_args": {
                        "type": "object",
                        "description": "Dictionary of extra inputs to pass to the managed agent, e.g. images, dataframes, or any other contextual data it may need.",
                    },
                }
                agent.output_type = "string"

    def _setup_tools(self, tools, add_base_tools):
        assert all(isinstance(tool, BaseTool) for tool in tools), (
            "All elements must be instance of BaseTool (or a subclass)"
        )
        self.tools = {tool.name: tool for tool in tools}
        if add_base_tools:
            self.tools.update(
                {
                    name: cls()
                    for name, cls in TOOL_MAPPING.items()
                    if name != "python_interpreter" or self.__class__.__name__ == "ToolCallingAgent"
                }
            )
        self.tools.setdefault("final_answer", FinalAnswerTool())

    def _validate_tools_and_managed_agents(self, tools, managed_agents):
        tool_and_managed_agent_names = [tool.name for tool in tools]
        if managed_agents is not None:
            tool_and_managed_agent_names += [agent.name for agent in managed_agents]
        if self.name:
            tool_and_managed_agent_names.append(self.name)
        if len(tool_and_managed_agent_names) != len(set(tool_and_managed_agent_names)):
            raise ValueError(
                "Each tool or managed_agent should have a unique name! You passed these duplicate names: "
                f"{[name for name in tool_and_managed_agent_names if tool_and_managed_agent_names.count(name) > 1]}"
            )

    def _setup_step_callbacks(self, step_callbacks):
        # Initialize step callbacks registry
        self.step_callbacks = CallbackRegistry()
        if step_callbacks:
            # Register callbacks list only for ActionStep for backward compatibility
            if isinstance(step_callbacks, list):
                for callback in step_callbacks:
                    self.step_callbacks.register(ActionStep, callback)
            # Register callbacks dict for specific step classes
            elif isinstance(step_callbacks, dict):
                for step_cls, callbacks in step_callbacks.items():
                    if not isinstance(callbacks, list):
                        callbacks = [callbacks]
                    for callback in callbacks:
                        self.step_callbacks.register(step_cls, callback)
            else:
                raise ValueError("step_callbacks must be a list or a dict")
        # Register monitor update_metrics only for ActionStep for backward compatibility
        self.step_callbacks.register(ActionStep, self.monitor.update_metrics)

    def run(
        self,
        task: str,
        stream: bool = False,
        reset: bool = True,
        images: list["PIL.Image.Image"] | None = None,
        additional_args: dict | None = None,
        max_steps: int | None = None,
        return_full_result: bool | None = None,
    ) -> Any | RunResult:
        """
        Run the agent for the given task.

        Args:
            task (`str`): Task to perform.
            stream (`bool`): Whether to run in streaming mode.
                If `True`, returns a generator that yields each step as it is executed. You must iterate over this generator to process the individual steps (e.g., using a for loop or `next()`).
                If `False`, executes all steps internally and returns only the final answer after completion.
            reset (`bool`): Whether to reset the conversation or keep it going from previous run.
            images (`list[PIL.Image.Image]`, *optional*): Image(s) objects.
            additional_args (`dict`, *optional*): Any other variables that you want to pass to the agent run, for instance images or dataframes. Give them clear names!
            max_steps (`int`, *optional*): Maximum number of steps the agent can take to solve the task. if not provided, will use the agent's default value.
            return_full_result (`bool`, *optional*): Whether to return the full [`RunResult`] object or just the final answer output.
                If `None` (default), the agent's `self.return_full_result` setting is used.

        Example:
        ```py
        from smolagents import CodeAgent
        agent = CodeAgent(tools=[])
        agent.run("What is the result of 2 power 3.7384?")
        ```
        """
        max_steps = max_steps or self.max_steps
        self.task = task
        self.interrupt_switch = False
        if additional_args:
            self.state.update(additional_args)
            self.task += f"""
You have been provided with these additional arguments, that you can access directly using the keys as variables:
{str(additional_args)}."""

        self.memory.system_prompt = SystemPromptStep(system_prompt=self.system_prompt)
        if reset:
            self.memory.reset()
            self.monitor.reset()

        self.logger.log_task(
            content=self.task.strip(),
            subtitle=f"{type(self.model).__name__} - {(self.model.model_id if hasattr(self.model, 'model_id') else '')}",
            level=LogLevel.INFO,
            title=self.name if hasattr(self, "name") else None,
        )
        self.memory.steps.append(TaskStep(task=self.task, task_images=images))

        if getattr(self, "python_executor", None):
            self.python_executor.send_variables(variables=self.state)
            self.python_executor.send_tools({**self.tools, **self.managed_agents})

        if stream:
            # The steps are returned as they are executed through a generator to iterate on.
            return self._run_stream(task=self.task, max_steps=max_steps, images=images)
        run_start_time = time.time()
        # Outputs are returned only at the end. We only look at the last step.

        steps = list(self._run_stream(task=self.task, max_steps=max_steps, images=images))
        assert isinstance(steps[-1], FinalAnswerStep)
        output = steps[-1].output

        return_full_result = return_full_result if return_full_result is not None else self.return_full_result
        if return_full_result:
            total_input_tokens = 0
            total_output_tokens = 0
            correct_token_usage = True
            for step in self.memory.steps:
                if isinstance(step, (ActionStep, PlanningStep)):
                    if step.token_usage is None:
                        correct_token_usage = False
                        break
                    else:
                        total_input_tokens += step.token_usage.input_tokens
                        total_output_tokens += step.token_usage.output_tokens
            if correct_token_usage:
                token_usage = TokenUsage(input_tokens=total_input_tokens, output_tokens=total_output_tokens)
            else:
                token_usage = None

            if self.memory.steps and isinstance(getattr(self.memory.steps[-1], "error", None), AgentMaxStepsError):
                state = "max_steps_error"
            else:
                state = "success"

            step_dicts = self.memory.get_full_steps()

            return RunResult(
                output=output,
                token_usage=token_usage,
                steps=step_dicts,
                timing=Timing(start_time=run_start_time, end_time=time.time()),
                state=state,
            )

        return output

    def _run_stream(
        self, task: str, max_steps: int, images: list["PIL.Image.Image"] | None = None
    ) -> Generator[ActionStep | PlanningStep | FinalAnswerStep | ChatMessageStreamDelta]:
        self.step_number = 1
        returned_final_answer = False
        while not returned_final_answer and self.step_number <= max_steps:
            if self.interrupt_switch:
                raise AgentError("Agent interrupted.", self.logger)

            # Run a planning step if scheduled
            if self.planning_interval is not None and (
                self.step_number == 1 or (self.step_number - 1) % self.planning_interval == 0
            ):
                planning_start_time = time.time()
                planning_step = None
                for element in self._generate_planning_step(
                    task, is_first_step=len(self.memory.steps) == 1, step=self.step_number
                ):  # Don't use the attribute step_number here, because there can be steps from previous runs
                    yield element
                    planning_step = element
                assert isinstance(planning_step, PlanningStep)  # Last yielded element should be a PlanningStep
                planning_end_time = time.time()
                planning_step.timing = Timing(
                    start_time=planning_start_time,
                    end_time=planning_end_time,
                )
                self._finalize_step(planning_step)
                self.memory.steps.append(planning_step)

            # Start action step!
            action_step_start_time = time.time()
            action_step = ActionStep(
                step_number=self.step_number,
                timing=Timing(start_time=action_step_start_time),
                observations_images=images,
            )
            self.logger.log_rule(f"Step {self.step_number}", level=LogLevel.INFO)
            try:
                for output in self._step_stream(action_step):
                    # Yield all
                    yield output

                    if isinstance(output, ActionOutput) and output.is_final_answer:
                        final_answer = output.output
                        self.logger.log(
                            Text(f"Final answer: {final_answer}", style=f"bold {YELLOW_HEX}"),
                            level=LogLevel.INFO,
                        )

                        if self.final_answer_checks:
                            self._validate_final_answer(final_answer)
                        returned_final_answer = True
                        action_step.is_final_answer = True

            except AgentGenerationError as e:
                # Agent generation errors are not caused by a Model error but an implementation error: so we should raise them and exit.
                raise e
            except AgentError as e:
                # Other AgentError types are caused by the Model, so we should log them and iterate.
                action_step.error = e
            finally:
                self._finalize_step(action_step)
                self.memory.steps.append(action_step)
                yield action_step
                self.step_number += 1

        if not returned_final_answer and self.step_number == max_steps + 1:
            final_answer = self._handle_max_steps_reached(task, images)
            yield action_step
        yield FinalAnswerStep(handle_agent_output_types(final_answer))

    def _validate_final_answer(self, final_answer: Any):
        for check_function in self.final_answer_checks:
            try:
                assert check_function(final_answer, self.memory)
            except Exception as e:
                raise AgentError(f"Check {check_function.__name__} failed with error: {e}", self.logger)

    def _finalize_step(self, memory_step: ActionStep | PlanningStep):
        memory_step.timing.end_time = time.time()
        self.step_callbacks.callback(memory_step, agent=self)

    def _handle_max_steps_reached(self, task: str, images: list["PIL.Image.Image"]) -> Any:
        action_step_start_time = time.time()
        final_answer = self.provide_final_answer(task, images)
        final_memory_step = ActionStep(
            step_number=self.step_number,
            error=AgentMaxStepsError("Reached max steps.", self.logger),
            timing=Timing(start_time=action_step_start_time, end_time=time.time()),
            token_usage=final_answer.token_usage,
        )
        final_memory_step.action_output = final_answer.content
        self._finalize_step(final_memory_step)
        self.memory.steps.append(final_memory_step)
        return final_answer.content

    def _generate_planning_step(
        self, task, is_first_step: bool, step: int
    ) -> Generator[ChatMessageStreamDelta | PlanningStep]:
        start_time = time.time()
        if is_first_step:
            input_messages = [
                ChatMessage(
                    role=MessageRole.USER,
                    content=[
                        {
                            "type": "text",
                            "text": populate_template(
                                self.prompt_templates["planning"]["initial_plan"],
                                variables={"task": task, "tools": self.tools, "managed_agents": self.managed_agents},
                            ),
                        }
                    ],
                )
            ]
            if self.stream_outputs and hasattr(self.model, "generate_stream"):
                plan_message_content = ""
                output_stream = self.model.generate_stream(input_messages, stop_sequences=["<end_plan>"])  # type: ignore
                input_tokens, output_tokens = 0, 0
                with Live("", console=self.logger.console, vertical_overflow="visible") as live:
                    for event in output_stream:
                        if event.content is not None:
                            plan_message_content += event.content
                            live.update(Markdown(plan_message_content))
                            if event.token_usage:
                                output_tokens += event.token_usage.output_tokens
                                input_tokens = event.token_usage.input_tokens
                        yield event
            else:
                plan_message = self.model.generate(input_messages, stop_sequences=["<end_plan>"])
                plan_message_content = plan_message.content
                input_tokens, output_tokens = (
                    (
                        plan_message.token_usage.input_tokens,
                        plan_message.token_usage.output_tokens,
                    )
                    if plan_message.token_usage
                    else (None, None)
                )
            plan = textwrap.dedent(
                f"""Here are the facts I know and the plan of action that I will follow to solve the task:\n```\n{plan_message_content}\n```"""
            )
        else:
            # Summary mode removes the system prompt and previous planning messages output by the model.
            # Removing previous planning messages avoids influencing too much the new plan.
            memory_messages = self.write_memory_to_messages(summary_mode=True)
            plan_update_pre = ChatMessage(
                role=MessageRole.SYSTEM,
                content=[
                    {
                        "type": "text",
                        "text": populate_template(
                            self.prompt_templates["planning"]["update_plan_pre_messages"], variables={"task": task}
                        ),
                    }
                ],
            )
            plan_update_post = ChatMessage(
                role=MessageRole.USER,
                content=[
                    {
                        "type": "text",
                        "text": populate_template(
                            self.prompt_templates["planning"]["update_plan_post_messages"],
                            variables={
                                "task": task,
                                "tools": self.tools,
                                "managed_agents": self.managed_agents,
                                "remaining_steps": (self.max_steps - step),
                            },
                        ),
                    }
                ],
            )
            input_messages = [plan_update_pre] + memory_messages + [plan_update_post]
            if self.stream_outputs and hasattr(self.model, "generate_stream"):
                plan_message_content = ""
                input_tokens, output_tokens = 0, 0
                with Live("", console=self.logger.console, vertical_overflow="visible") as live:
                    for event in self.model.generate_stream(
                        input_messages,
                        stop_sequences=["<end_plan>"],
                    ):  # type: ignore
                        if event.content is not None:
                            plan_message_content += event.content
                            live.update(Markdown(plan_message_content))
                            if event.token_usage:
                                output_tokens += event.token_usage.output_tokens
                                input_tokens = event.token_usage.input_tokens
                        yield event
            else:
                plan_message = self.model.generate(input_messages, stop_sequences=["<end_plan>"])
                plan_message_content = plan_message.content
                if plan_message.token_usage is not None:
                    input_tokens, output_tokens = (
                        plan_message.token_usage.input_tokens,
                        plan_message.token_usage.output_tokens,
                    )
            plan = textwrap.dedent(
                f"""I still need to solve the task I was given:\n```\n{self.task}\n```\n\nHere are the facts I know and my new/updated plan of action to solve the task:\n```\n{plan_message_content}\n```"""
            )
        log_headline = "Initial plan" if is_first_step else "Updated plan"
        self.logger.log(Rule(f"[bold]{log_headline}", style="orange"), Text(plan), level=LogLevel.INFO)
        yield PlanningStep(
            model_input_messages=input_messages,
            plan=plan,
            model_output_message=ChatMessage(role=MessageRole.ASSISTANT, content=plan_message_content),
            token_usage=TokenUsage(input_tokens=input_tokens, output_tokens=output_tokens),
            timing=Timing(start_time=start_time, end_time=time.time()),
        )

    @abstractmethod
    def initialize_system_prompt(self) -> str:
        """To be implemented in child classes"""
        ...

    def interrupt(self):
        """Interrupts the agent execution."""
        self.interrupt_switch = True

    def write_memory_to_messages(
        self,
        summary_mode: bool = False,
    ) -> list[ChatMessage]:
        """
        Reads past llm_outputs, actions, and observations or errors from the memory into a series of messages
        that can be used as input to the LLM. Adds a number of keywords (such as PLAN, error, etc) to help
        the LLM.
        """
        messages = self.memory.system_prompt.to_messages(summary_mode=summary_mode)
        for memory_step in self.memory.steps:
            messages.extend(memory_step.to_messages(summary_mode=summary_mode))
        return messages

    def _step_stream(
        self, memory_step: ActionStep
    ) -> Generator[ChatMessageStreamDelta | ToolCall | ToolOutput | ActionOutput]:
        """
        Perform one step in the ReAct framework: the agent thinks, acts, and observes the result.
        Yields ChatMessageStreamDelta during the run if streaming is enabled.
        At the end, yields either None if the step is not final, or the final answer.
        """
        raise NotImplementedError("This method should be implemented in child classes")

    def step(self, memory_step: ActionStep) -> Any:
        """
        Perform one step in the ReAct framework: the agent thinks, acts, and observes the result.
        Returns either None if the step is not final, or the final answer.
        """
        return list(self._step_stream(memory_step))[-1]

    def extract_action(self, model_output: str, split_token: str) -> tuple[str, str]:
        """
        Parse action from the LLM output

        Args:
            model_output (`str`): Output of the LLM
            split_token (`str`): Separator for the action. Should match the example in the system prompt.
        """
        try:
            split = model_output.split(split_token)
            rationale, action = (
                split[-2],
                split[-1],
            )  # NOTE: using indexes starting from the end solves for when you have more than one split_token in the output
        except Exception:
            raise AgentParsingError(
                f"No '{split_token}' token provided in your output.\nYour output:\n{model_output}\n. Be sure to include an action, prefaced with '{split_token}'!",
                self.logger,
            )
        return rationale.strip(), action.strip()

    def provide_final_answer(self, task: str, images: list["PIL.Image.Image"] | None = None) -> ChatMessage:
        """
        Provide the final answer to the task, based on the logs of the agent's interactions.

        Args:
            task (`str`): Task to perform.
            images (`list[PIL.Image.Image]`, *optional*): Image(s) objects.

        Returns:
            `str`: Final answer to the task.
        """
        messages = [
            ChatMessage(
                role=MessageRole.SYSTEM,
                content=[
                    {
                        "type": "text",
                        "text": self.prompt_templates["final_answer"]["pre_messages"],
                    }
                ],
            )
        ]
        if images:
            messages[0].content += [{"type": "image", "image": image} for image in images]
        messages += self.write_memory_to_messages()[1:]
        messages.append(
            ChatMessage(
                role=MessageRole.USER,
                content=[
                    {
                        "type": "text",
                        "text": populate_template(
                            self.prompt_templates["final_answer"]["post_messages"], variables={"task": task}
                        ),
                    }
                ],
            )
        )
        try:
            chat_message: ChatMessage = self.model.generate(messages)
            return chat_message
        except Exception as e:
            return ChatMessage(
                role=MessageRole.ASSISTANT,
                content=[{"type": "text", "text": f"Error in generating final LLM output: {e}"}],
            )

    def visualize(self):
        """Creates a rich tree visualization of the agent's structure."""
        self.logger.visualize_agent_tree(self)

    def replay(self, detailed: bool = False):
        """Prints a pretty replay of the agent's steps.

        Args:
            detailed (bool, optional): If True, also displays the memory at each step. Defaults to False.
                Careful: will increase log length exponentially. Use only for debugging.
        """
        self.memory.replay(self.logger, detailed=detailed)

    def __call__(self, task: str, **kwargs):
        """Adds additional prompting for the managed agent, runs it, and wraps the output.
        This method is called only by a managed agent.
        """
        full_task = populate_template(
            self.prompt_templates["managed_agent"]["task"],
            variables=dict(name=self.name, task=task),
        )
        result = self.run(full_task, **kwargs)
        if isinstance(result, RunResult):
            report = result.output
        else:
            report = result
        answer = populate_template(
            self.prompt_templates["managed_agent"]["report"], variables=dict(name=self.name, final_answer=report)
        )
        if self.provide_run_summary:
            answer += "\n\nFor more detail, find below a summary of this agent's work:\n<summary_of_work>\n"
            for message in self.write_memory_to_messages(summary_mode=True):
                content = message.content
                answer += "\n" + truncate_content(str(content)) + "\n---"
            answer += "\n</summary_of_work>"
        return answer

    def save(self, output_dir: str | Path, relative_path: str | None = None):
        """
        Saves the relevant code files for your agent. This will copy the code of your agent in `output_dir` as well as autogenerate:

        - a `tools` folder containing the logic for each of the tools under `tools/{tool_name}.py`.
        - a `managed_agents` folder containing the logic for each of the managed agents.
        - an `agent.json` file containing a dictionary representing your agent.
        - a `prompt.yaml` file containing the prompt templates used by your agent.
        - an `app.py` file providing a UI for your agent when it is exported to a Space with `agent.push_to_hub()`
        - a `requirements.txt` containing the names of the modules used by your tool (as detected when inspecting its
          code)

        Args:
            output_dir (`str` or `Path`): The folder in which you want to save your agent.
        """
        make_init_file(output_dir)

        # Recursively save managed agents
        if self.managed_agents:
            make_init_file(os.path.join(output_dir, "managed_agents"))
            for agent_name, agent in self.managed_agents.items():
                agent_suffix = f"managed_agents.{agent_name}"
                if relative_path:
                    agent_suffix = relative_path + "." + agent_suffix
                agent.save(os.path.join(output_dir, "managed_agents", agent_name), relative_path=agent_suffix)

        class_name = self.__class__.__name__

        # Save tools to different .py files
        for tool in self.tools.values():
            make_init_file(os.path.join(output_dir, "tools"))
            tool.save(os.path.join(output_dir, "tools"), tool_file_name=tool.name, make_gradio_app=False)

        # Save prompts to yaml
        yaml_prompts = yaml.safe_dump(
            self.prompt_templates,
            default_style="|",  # This forces block literals for all strings
            default_flow_style=False,
            width=float("inf"),
            sort_keys=False,
            allow_unicode=True,
            indent=2,
        )

        with open(os.path.join(output_dir, "prompts.yaml"), "w", encoding="utf-8") as f:
            f.write(yaml_prompts)

        # Save agent dictionary to json
        agent_dict = self.to_dict()
        agent_dict["tools"] = [tool.name for tool in self.tools.values()]
        agent_dict["managed_agents"] = {agent.name: agent.__class__.__name__ for agent in self.managed_agents.values()}
        with open(os.path.join(output_dir, "agent.json"), "w", encoding="utf-8") as f:
            json.dump(agent_dict, f, indent=4)

        # Save requirements
        with open(os.path.join(output_dir, "requirements.txt"), "w", encoding="utf-8") as f:
            f.writelines(f"{r}\n" for r in agent_dict["requirements"])

        # Make agent.py file with Gradio UI
        agent_name = f"agent_{self.name}" if getattr(self, "name", None) else "agent"
        managed_agent_relative_path = relative_path + "." if relative_path is not None else ""
        app_template = create_agent_gradio_app_template()

        # Render the app.py file from Jinja2 template
        app_text = app_template.render(
            {
                "agent_name": agent_name,
                "class_name": class_name,
                "agent_dict": agent_dict,
                "tools": self.tools,
                "managed_agents": self.managed_agents,
                "managed_agent_relative_path": managed_agent_relative_path,
            }
        )

        with open(os.path.join(output_dir, "app.py"), "w", encoding="utf-8") as f:
            f.write(app_text + "\n")  # Append newline at the end

    def to_dict(self) -> dict[str, Any]:
        """Convert the agent to a dictionary representation.

        Returns:
            `dict`: Dictionary representation of the agent.
        """
        # TODO: handle serializing step_callbacks and final_answer_checks
        for attr in ["final_answer_checks", "step_callbacks"]:
            if getattr(self, attr, None):
                self.logger.log(f"This agent has {attr}: they will be ignored by this method.", LogLevel.INFO)

        tool_dicts = [tool.to_dict() for tool in self.tools.values()]
        tool_requirements = {req for tool in self.tools.values() for req in tool.to_dict()["requirements"]}
        managed_agents_requirements = {
            req for managed_agent in self.managed_agents.values() for req in managed_agent.to_dict()["requirements"]
        }
        requirements = tool_requirements | managed_agents_requirements
        if hasattr(self, "authorized_imports"):
            requirements.update(
                {package.split(".")[0] for package in self.authorized_imports if package not in BASE_BUILTIN_MODULES}
            )

        agent_dict = {
            "class": self.__class__.__name__,
            "tools": tool_dicts,
            "model": {
                "class": self.model.__class__.__name__,
                "data": self.model.to_dict(),
            },
            "managed_agents": [managed_agent.to_dict() for managed_agent in self.managed_agents.values()],
            "prompt_templates": self.prompt_templates,
            "max_steps": self.max_steps,
            "verbosity_level": int(self.logger.level),
            "planning_interval": self.planning_interval,
            "name": self.name,
            "description": self.description,
            "requirements": sorted(requirements),
        }
        return agent_dict

    @classmethod
    def from_dict(cls, agent_dict: dict[str, Any], **kwargs) -> "MultiStepAgent":
        """Create agent from a dictionary representation.

        Args:
            agent_dict (`dict[str, Any]`): Dictionary representation of the agent.
            **kwargs: Additional keyword arguments that will override agent_dict values.

        Returns:
            `MultiStepAgent`: Instance of the agent class.
        """
        # Load model
        model_info = agent_dict["model"]
        model_class = getattr(importlib.import_module("smolagents.models"), model_info["class"])
        model = model_class.from_dict(model_info["data"])
        # Load tools
        tools = []
        for tool_info in agent_dict["tools"]:
            tools.append(Tool.from_code(tool_info["code"]))
        # Load managed agents
        managed_agents = []
        for managed_agent_dict in agent_dict["managed_agents"]:
            agent_class = getattr(importlib.import_module("smolagents.agents"), managed_agent_dict["class"])
            managed_agent = agent_class.from_dict(managed_agent_dict, **kwargs)
            managed_agents.append(managed_agent)
        # Extract base agent parameters
        agent_args = {
            "model": model,
            "tools": tools,
            "managed_agents": managed_agents,
            "prompt_templates": agent_dict.get("prompt_templates"),
            "max_steps": agent_dict.get("max_steps"),
            "verbosity_level": agent_dict.get("verbosity_level"),
            "planning_interval": agent_dict.get("planning_interval"),
            "name": agent_dict.get("name"),
            "description": agent_dict.get("description"),
        }
        # Filter out None values to use defaults from __init__
        agent_args = {k: v for k, v in agent_args.items() if v is not None}
        # Update with any additional kwargs
        agent_args.update(kwargs)
        # Create agent instance
        return cls(**agent_args)

    @classmethod
    def from_hub(
        cls,
        repo_id: str,
        token: str | None = None,
        trust_remote_code: bool = False,
        **kwargs,
    ):
        """
        Loads an agent defined on the Hub.

        <Tip warning={true}>

        Loading a tool from the Hub means that you'll download the tool and execute it locally.
        ALWAYS inspect the tool you're downloading before loading it within your runtime, as you would do when
        installing a package using pip/npm/apt.

        </Tip>

        Args:
            repo_id (`str`):
                The name of the repo on the Hub where your tool is defined.
            token (`str`, *optional*):
                The token to identify you on hf.co. If unset, will use the token generated when running
                `huggingface-cli login` (stored in `~/.huggingface`).
            trust_remote_code(`bool`, *optional*, defaults to False):
                This flags marks that you understand the risk of running remote code and that you trust this tool.
                If not setting this to True, loading the tool from Hub will fail.
            kwargs (additional keyword arguments, *optional*):
                Additional keyword arguments that will be split in two: all arguments relevant to the Hub (such as
                `cache_dir`, `revision`, `subfolder`) will be used when downloading the files for your agent, and the
                others will be passed along to its init.
        """
        if not trust_remote_code:
            raise ValueError(
                "Loading an agent from Hub requires to acknowledge you trust its code: to do so, pass `trust_remote_code=True`."
            )

        # Get the agent's Hub folder.
        download_kwargs = {"token": token, "repo_type": "space"} | {
            key: kwargs.pop(key)
            for key in [
                "cache_dir",
                "force_download",
                "proxies",
                "revision",
                "local_files_only",
            ]
            if key in kwargs
        }

        download_folder = Path(snapshot_download(repo_id=repo_id, **download_kwargs))
        return cls.from_folder(download_folder, **kwargs)

    @classmethod
    def from_folder(cls, folder: str | Path, **kwargs):
        """Loads an agent from a local folder.

        Args:
            folder (`str` or `Path`): The folder where the agent is saved.
            **kwargs: Additional keyword arguments that will be passed to the agent's init.
        """
        # Load agent.json
        folder = Path(folder)
        agent_dict = json.loads((folder / "agent.json").read_text())

        # Load managed agents from their respective folders, recursively
        managed_agents = []
        for managed_agent_name, managed_agent_class_name in agent_dict["managed_agents"].items():
            agent_cls = getattr(importlib.import_module("smolagents.agents"), managed_agent_class_name)
            managed_agents.append(agent_cls.from_folder(folder / "managed_agents" / managed_agent_name))
        agent_dict["managed_agents"] = {}

        # Load tools
        tools = []
        for tool_name in agent_dict["tools"]:
            tool_code = (folder / "tools" / f"{tool_name}.py").read_text()
            tools.append({"name": tool_name, "code": tool_code})
        agent_dict["tools"] = tools

        # Add managed agents to kwargs to override the empty list in from_dict
        if managed_agents:
            kwargs["managed_agents"] = managed_agents

        return cls.from_dict(agent_dict, **kwargs)

    def push_to_hub(
        self,
        repo_id: str,
        commit_message: str = "Upload agent",
        private: bool | None = None,
        token: bool | str | None = None,
        create_pr: bool = False,
    ) -> str:
        """
        Upload the agent to the Hub.

        Parameters:
            repo_id (`str`):
                The name of the repository you want to push to. It should contain your organization name when
                pushing to a given organization.
            commit_message (`str`, *optional*, defaults to `"Upload agent"`):
                Message to commit while pushing.
            private (`bool`, *optional*, defaults to `None`):
                Whether to make the repo private. If `None`, the repo will be public unless the organization's default is private. This value is ignored if the repo already exists.
            token (`bool` or `str`, *optional*):
                The token to use as HTTP bearer authorization for remote files. If unset, will use the token generated
                when running `huggingface-cli login` (stored in `~/.huggingface`).
            create_pr (`bool`, *optional*, defaults to `False`):
                Whether to create a PR with the uploaded files or directly commit.
        """
        repo_url = create_repo(
            repo_id=repo_id,
            token=token,
            private=private,
            exist_ok=True,
            repo_type="space",
            space_sdk="gradio",
        )
        repo_id = repo_url.repo_id
        metadata_update(
            repo_id,
            {"tags": ["smolagents", "agent"]},
            repo_type="space",
            token=token,
            overwrite=True,
        )

        with tempfile.TemporaryDirectory() as work_dir:
            self.save(work_dir)
            logger.info(f"Uploading the following files to {repo_id}: {','.join(os.listdir(work_dir))}")
            return upload_folder(
                repo_id=repo_id,
                commit_message=commit_message,
                folder_path=work_dir,
                token=token,
                create_pr=create_pr,
                repo_type="space",
            )


class ToolCallingAgent(MultiStepAgent):
    """
    This agent uses JSON-like tool calls, using method `model.get_tool_call` to leverage the LLM engine's tool calling capabilities.

    Args:
        tools (`list[Tool]`): [`Tool`]s that the agent can use.
        model (`Model`): Model that will generate the agent's actions.
        prompt_templates ([`~agents.PromptTemplates`], *optional*): Prompt templates.
        planning_interval (`int`, *optional*): Interval at which the agent will run a planning step.
        stream_outputs (`bool`, *optional*, default `False`): Whether to stream outputs during execution.
        max_tool_threads (`int`, *optional*): Maximum number of threads for parallel tool calls.
            Higher values increase concurrency but resource usage as well.
            Defaults to `ThreadPoolExecutor`'s default.
        **kwargs: Additional keyword arguments.
    """

    def __init__(
        self,
        tools: list[Tool],
        model: Model,
        prompt_templates: PromptTemplates | None = None,
        planning_interval: int | None = None,
        stream_outputs: bool = False,
        max_tool_threads: int | None = None,
        **kwargs,
    ):
        prompt_templates = prompt_templates or yaml.safe_load(
            importlib.resources.files("smolagents.prompts").joinpath("toolcalling_agent.yaml").read_text()
        )
        super().__init__(
            tools=tools,
            model=model,
            prompt_templates=prompt_templates,
            planning_interval=planning_interval,
            **kwargs,
        )
        # Streaming setup
        self.stream_outputs = stream_outputs
        if self.stream_outputs and not hasattr(self.model, "generate_stream"):
            raise ValueError(
                "`stream_outputs` is set to True, but the model class implements no `generate_stream` method."
            )
        # Tool calling setup
        self.max_tool_threads = max_tool_threads

    @property
    def tools_and_managed_agents(self):
        """Returns a combined list of tools and managed agents."""
        return list(self.tools.values()) + list(self.managed_agents.values())

    def initialize_system_prompt(self) -> str:
        system_prompt = populate_template(
            self.prompt_templates["system_prompt"],
            variables={
                "tools": self.tools,
                "managed_agents": self.managed_agents,
                "custom_instructions": self.instructions,
            },
        )
        return system_prompt

    def _step_stream(
        self, memory_step: ActionStep
    ) -> Generator[ChatMessageStreamDelta | ToolCall | ToolOutput | ActionOutput]:
        """
        Perform one step in the ReAct framework: the agent thinks, acts, and observes the result.
        Yields ChatMessageStreamDelta during the run if streaming is enabled.
        At the end, yields either None if the step is not final, or the final answer.
        """
        memory_messages = self.write_memory_to_messages()

        input_messages = memory_messages.copy()

        # Add new step in logs
        memory_step.model_input_messages = input_messages

        try:
            if self.stream_outputs and hasattr(self.model, "generate_stream"):
                output_stream = self.model.generate_stream(
                    input_messages,
                    stop_sequences=["Observation:", "Calling tools:"],
                    tools_to_call_from=self.tools_and_managed_agents,
                )

                chat_message_stream_deltas: list[ChatMessageStreamDelta] = []
                with Live("", console=self.logger.console, vertical_overflow="visible") as live:
                    for event in output_stream:
                        chat_message_stream_deltas.append(event)
                        live.update(
                            Markdown(agglomerate_stream_deltas(chat_message_stream_deltas).render_as_markdown())
                        )
                        yield event
                chat_message = agglomerate_stream_deltas(chat_message_stream_deltas)
            else:
                chat_message: ChatMessage = self.model.generate(
                    input_messages,
                    stop_sequences=["Observation:", "Calling tools:"],
                    tools_to_call_from=self.tools_and_managed_agents,
                )
                if chat_message.content is None and chat_message.raw is not None:
                    log_content = str(chat_message.raw)
                else:
                    log_content = str(chat_message.content) or ""

                self.logger.log_markdown(
                    content=log_content,
                    title="Output message of the LLM:",
                    level=LogLevel.DEBUG,
                )

            # Record model output
            memory_step.model_output_message = chat_message
            memory_step.model_output = chat_message.content
            memory_step.token_usage = chat_message.token_usage
            print(f"\nTotal token usage for step {memory_step.step_number} of '{self.name}': {chat_message.token_usage}\n")
        except Exception as e:
            raise AgentGenerationError(f"Error while generating output:\n{e}", self.logger) from e

        if chat_message.tool_calls is None or len(chat_message.tool_calls) == 0:
            try:
                chat_message = self.model.parse_tool_calls(chat_message)
            except Exception as e:
                raise AgentParsingError(f"Error while parsing tool call from model output: {e}", self.logger)
        else:
            for tool_call in chat_message.tool_calls:
                tool_call.function.arguments = parse_json_if_needed(tool_call.function.arguments)
        final_answer, got_final_answer = None, False
        for output in self.process_tool_calls(chat_message, memory_step):
            yield output
            if isinstance(output, ToolOutput):
                if output.is_final_answer:
                    if len(chat_message.tool_calls) > 1:
                        raise AgentExecutionError(
                            "If you want to return an answer, please do not perform any other tool calls than the final answer tool call!",
                            self.logger,
                        )
                    if got_final_answer:
                        raise AgentToolExecutionError(
                            "You returned multiple final answers. Please return only one single final answer!",
                            self.logger,
                        )
                    final_answer = output.output
                    got_final_answer = True

                    # Manage state variables
                    if isinstance(final_answer, str) and final_answer in self.state.keys():
                        final_answer = self.state[final_answer]
        yield ActionOutput(
            output=final_answer,
            is_final_answer=got_final_answer,
        )

    def process_tool_calls(
        self, chat_message: ChatMessage, memory_step: ActionStep
    ) -> Generator[ToolCall | ToolOutput]:
        """Process tool calls from the model output and update agent memory.

        Args:
            chat_message (`ChatMessage`): Chat message containing tool calls from the model.
            memory_step (`ActionStep)`: Memory ActionStep to update with results.

        Yields:
            `ToolCall | ToolOutput`: The tool call or tool output.
        """
        parallel_calls: dict[str, ToolCall] = {}
        assert chat_message.tool_calls is not None
        for chat_tool_call in chat_message.tool_calls:
            tool_call = ToolCall(
                name=chat_tool_call.function.name, arguments=chat_tool_call.function.arguments, id=chat_tool_call.id
            )
            yield tool_call
            parallel_calls[tool_call.id] = tool_call

        # Helper function to process a single tool call
        def process_single_tool_call(tool_call: ToolCall) -> ToolOutput:
            tool_name = tool_call.name
            tool_arguments = tool_call.arguments or {}
            self.logger.log(
                Panel(Text(f"Calling tool: '{tool_name}' with arguments: {tool_arguments}")),
                level=LogLevel.INFO,
            )
            tool_call_result = self.execute_tool_call(tool_name, tool_arguments)
            tool_call_result_type = type(tool_call_result)
            if tool_call_result_type in [AgentImage, AgentAudio]:
                if tool_call_result_type == AgentImage:
                    observation_name = "image.png"
                elif tool_call_result_type == AgentAudio:
                    observation_name = "audio.mp3"
                # TODO: tool_call_result naming could allow for different names of same type
                self.state[observation_name] = tool_call_result
                observation = f"Stored '{observation_name}' in memory."
            else:
                observation = str(tool_call_result).strip()
            self.logger.log(
                f"Observations: {observation.replace('[', '|')}",  # escape potential rich-tag-like components
                level=LogLevel.INFO,
            )
            is_final_answer = tool_name == "final_answer"

            return ToolOutput(
                id=tool_call.id,
                output=tool_call_result,
                is_final_answer=is_final_answer,
                observation=observation,
                tool_call=tool_call,
            )

        # Process tool calls in parallel
        outputs = {}
        if len(parallel_calls) == 1:
            # If there's only one call, process it directly
            tool_call = list(parallel_calls.values())[0]
            tool_output = process_single_tool_call(tool_call)
            outputs[tool_output.id] = tool_output
            yield tool_output
        else:
            # If multiple tool calls, process them in parallel
            with ThreadPoolExecutor(self.max_tool_threads) as executor:
                futures = [
                    executor.submit(process_single_tool_call, tool_call) for tool_call in parallel_calls.values()
                ]
                for future in as_completed(futures):
                    tool_output = future.result()
                    outputs[tool_output.id] = tool_output
                    yield tool_output

        memory_step.tool_calls = [parallel_calls[k] for k in sorted(parallel_calls.keys())]
        memory_step.observations = memory_step.observations or ""
        for tool_output in [outputs[k] for k in sorted(outputs.keys())]:
            memory_step.observations += tool_output.observation + "\n"
        memory_step.observations = (
            memory_step.observations.rstrip("\n") if memory_step.observations else memory_step.observations
        )

    def _substitute_state_variables(self, arguments: dict[str, str] | str) -> dict[str, Any] | str:
        """Replace string values in arguments with their corresponding state values if they exist."""
        if isinstance(arguments, dict):
            return {
                key: self.state.get(value, value) if isinstance(value, str) else value
                for key, value in arguments.items()
            }
        return arguments

    def execute_tool_call(self, tool_name: str, arguments: dict[str, str] | str) -> Any:
        """
        Execute a tool or managed agent with the provided arguments.

        The arguments are replaced with the actual values from the state if they refer to state variables.

        Args:
            tool_name (`str`): Name of the tool or managed agent to execute.
            arguments (dict[str, str] | str): Arguments passed to the tool call.
        """
        # Check if the tool exists
        available_tools = {**self.tools, **self.managed_agents}
        if tool_name not in available_tools:
            raise AgentToolExecutionError(
                f"Unknown tool {tool_name}, should be one of: {', '.join(available_tools)}.", self.logger
            )

        # Get the tool and substitute state variables in arguments
        tool = available_tools[tool_name]
        arguments = self._substitute_state_variables(arguments)
        is_managed_agent = tool_name in self.managed_agents

        try:
            validate_tool_arguments(tool, arguments)
        except (ValueError, TypeError) as e:
            raise AgentToolCallError(str(e), self.logger) from e
        except Exception as e:
            error_msg = f"Error executing tool '{tool_name}' with arguments {str(arguments)}: {type(e).__name__}: {e}"
            raise AgentToolExecutionError(error_msg, self.logger) from e

        try:
            # Call tool with appropriate arguments
            if isinstance(arguments, dict):
                return tool(**arguments) if is_managed_agent else tool(**arguments, sanitize_inputs_outputs=True)
            else:
                return tool(arguments) if is_managed_agent else tool(arguments, sanitize_inputs_outputs=True)

        except Exception as e:
            # Handle execution errors
            if is_managed_agent:
                error_msg = (
                    f"Error executing request to team member '{tool_name}' with arguments {str(arguments)}: {e}\n"
                    "Please try again or request to another team member"
                )
            else:
                error_msg = (
                    f"Error executing tool '{tool_name}' with arguments {str(arguments)}: {type(e).__name__}: {e}\n"
                    "Please try again or use another tool"
                )
            raise AgentToolExecutionError(error_msg, self.logger) from e


class CodeAgent(MultiStepAgent):
    """
    In this agent, the tool calls will be formulated by the LLM in code format, then parsed and executed.

    Args:
        tools (`list[Tool]`): [`Tool`]s that the agent can use.
        model (`Model`): Model that will generate the agent's actions.
        prompt_templates ([`~agents.PromptTemplates`], *optional*): Prompt templates.
        additional_authorized_imports (`list[str]`, *optional*): Additional authorized imports for the agent.
        planning_interval (`int`, *optional*): Interval at which the agent will run a planning step.
        executor_type (`Literal["local", "e2b", "docker", "wasm"]`, default `"local"`): Type of code executor.
        executor_kwargs (`dict`, *optional*): Additional arguments to pass to initialize the executor.
        max_print_outputs_length (`int`, *optional*): Maximum length of the print outputs.
        stream_outputs (`bool`, *optional*, default `False`): Whether to stream outputs during execution.
        use_structured_outputs_internally (`bool`, default `False`): Whether to use structured generation at each action step: improves performance for many models.

            <Added version="1.17.0"/>
        code_block_tags (`tuple[str, str]` | `Literal["markdown"]`, *optional*): Opening and closing tags for code blocks (regex strings). Pass a custom tuple, or pass 'markdown' to use ("```(?:python|py)", "\\n```"), leave empty to use ("<code>", "</code>").
        **kwargs: Additional keyword arguments.
    """

    def __init__(
        self,
        tools: list[Tool],
        model: Model,
        prompt_templates: PromptTemplates | None = None,
        additional_authorized_imports: list[str] | None = None,
        planning_interval: int | None = None,
        executor_type: Literal["local", "e2b", "docker", "wasm"] = "local",
        executor_kwargs: dict[str, Any] | None = None,
        max_print_outputs_length: int | None = None,
        stream_outputs: bool = False,
        use_structured_outputs_internally: bool = False,
        code_block_tags: str | tuple[str, str] | None = None,
        **kwargs,
    ):
        self.additional_authorized_imports = additional_authorized_imports if additional_authorized_imports else []
        self.authorized_imports = sorted(set(BASE_BUILTIN_MODULES) | set(self.additional_authorized_imports))
        self.max_print_outputs_length = max_print_outputs_length
        self._use_structured_outputs_internally = use_structured_outputs_internally
        if self._use_structured_outputs_internally:
            prompt_templates = prompt_templates or yaml.safe_load(
                importlib.resources.files("smolagents.prompts").joinpath("structured_code_agent.yaml").read_text()
            )
        else:
            prompt_templates = prompt_templates or yaml.safe_load(
                importlib.resources.files("smolagents.prompts").joinpath("code_agent.yaml").read_text()
            )

        if isinstance(code_block_tags, str) and not code_block_tags == "markdown":
            raise ValueError("Only 'markdown' is supported for a string argument to `code_block_tags`.")
        self.code_block_tags = (
            code_block_tags
            if isinstance(code_block_tags, tuple)
            else ("```python", "```")
            if code_block_tags == "markdown"
            else ("<code>", "</code>")
        )

        super().__init__(
            tools=tools,
            model=model,
            prompt_templates=prompt_templates,
            planning_interval=planning_interval,
            **kwargs,
        )
        self.stream_outputs = stream_outputs
        if self.stream_outputs and not hasattr(self.model, "generate_stream"):
            raise ValueError(
                "`stream_outputs` is set to True, but the model class implements no `generate_stream` method."
            )
        if "*" in self.additional_authorized_imports:
            self.logger.log(
                "Caution: you set an authorization for all imports, meaning your agent can decide to import any package it deems necessary. This might raise issues if the package is not installed in your environment.",
                level=LogLevel.INFO,
            )
        if executor_type not in {"local", "e2b", "docker", "wasm"}:
            raise ValueError(f"Unsupported executor type: {executor_type}")
        self.executor_type = executor_type
        self.executor_kwargs: dict[str, Any] = executor_kwargs or {}
        self.python_executor = self.create_python_executor()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.cleanup()

    def cleanup(self):
        """Clean up resources used by the agent, such as the remote Python executor."""
        if hasattr(self.python_executor, "cleanup"):
            self.python_executor.cleanup()

    def create_python_executor(self) -> PythonExecutor:
        if self.executor_type == "local":
            return LocalPythonExecutor(
                self.additional_authorized_imports,
                **{"max_print_outputs_length": self.max_print_outputs_length} | self.executor_kwargs,
            )
        else:
            if self.managed_agents:
                raise Exception("Managed agents are not yet supported with remote code execution.")
            remote_executors = {
                "e2b": E2BExecutor,
                "docker": DockerExecutor,
                "wasm": WasmExecutor,
            }
            return remote_executors[self.executor_type](
                self.additional_authorized_imports, self.logger, **self.executor_kwargs
            )

    def initialize_system_prompt(self) -> str:
        system_prompt = populate_template(
            self.prompt_templates["system_prompt"],
            variables={
                "tools": self.tools,
                "managed_agents": self.managed_agents,
                "authorized_imports": (
                    "You can import from any package you want."
                    if "*" in self.authorized_imports
                    else str(self.authorized_imports)
                ),
                "custom_instructions": self.instructions,
                "code_block_opening_tag": self.code_block_tags[0],
                "code_block_closing_tag": self.code_block_tags[1],
            },
        )
        return system_prompt

    def _step_stream(
        self, memory_step: ActionStep
    ) -> Generator[ChatMessageStreamDelta | ToolCall | ToolOutput | ActionOutput]:
        """
        Perform one step in the ReAct framework: the agent thinks, acts, and observes the result.
        Yields ChatMessageStreamDelta during the run if streaming is enabled.
        At the end, yields either None if the step is not final, or the final answer.
        """
        memory_messages = self.write_memory_to_messages()

        input_messages = memory_messages.copy()
        ### Generate model output ###
        memory_step.model_input_messages = input_messages
        stop_sequences = ["Observation:", "Calling tools:"]
        if self.code_block_tags[1] not in self.code_block_tags[0]:
            # If the closing tag is contained in the opening tag, adding it as a stop sequence would cut short any code generation
            stop_sequences.append(self.code_block_tags[1])
        try:
            additional_args: dict[str, Any] = {}
            if self._use_structured_outputs_internally:
                additional_args["response_format"] = CODEAGENT_RESPONSE_FORMAT
            if self.stream_outputs:
                output_stream = self.model.generate_stream(
                    input_messages,
                    stop_sequences=stop_sequences,
                    **additional_args,
                )
                chat_message_stream_deltas: list[ChatMessageStreamDelta] = []
                with Live("", console=self.logger.console, vertical_overflow="visible") as live:
                    for event in output_stream:
                        chat_message_stream_deltas.append(event)
                        live.update(
                            Markdown(agglomerate_stream_deltas(chat_message_stream_deltas).render_as_markdown())
                        )
                        yield event
                chat_message = agglomerate_stream_deltas(chat_message_stream_deltas)
                memory_step.model_output_message = chat_message
                output_text = chat_message.content
            else:
                chat_message: ChatMessage = self.model.generate(
                    input_messages,
                    stop_sequences=stop_sequences,
                    **additional_args,
                )
                memory_step.model_output_message = chat_message
                output_text = chat_message.content
                self.logger.log_markdown(
                    content=output_text,
                    title="Output message of the LLM:",
                    level=LogLevel.DEBUG,
                )

            if not self._use_structured_outputs_internally:
                # This adds the end code sequence (i.e. the closing code block tag) to the history.
                # This will nudge subsequent LLM calls to finish with this end code sequence, thus efficiently stopping generation.
                if output_text and not output_text.strip().endswith(self.code_block_tags[1]):
                    output_text += self.code_block_tags[1]
                    memory_step.model_output_message.content = output_text

            memory_step.token_usage = chat_message.token_usage
            print(f"\nTotal token usage for step {memory_step.step_number} of '{self.name}': {chat_message.token_usage}\n")
            memory_step.model_output = output_text
        except Exception as e:
            raise AgentGenerationError(f"Error in generating model output:\n{e}", self.logger) from e

        ### Parse output ###
        try:
            if self._use_structured_outputs_internally:
                code_action = json.loads(output_text)["code"]
                code_action = extract_code_from_text(code_action, self.code_block_tags) or code_action
            else:
                code_action = parse_code_blobs(output_text, self.code_block_tags)
            code_action = fix_final_answer_code(code_action)
            memory_step.code_action = code_action
        except Exception as e:
            error_msg = f"Error in code parsing:\n{e}\nMake sure to provide correct code blobs."
            raise AgentParsingError(error_msg, self.logger)

        tool_call = ToolCall(
            name="python_interpreter",
            arguments=code_action,
            id=f"call_{len(self.memory.steps)}",
        )
        yield tool_call
        memory_step.tool_calls = [tool_call]

        ### Execute action ###
        self.logger.log_code(title="Executing parsed code:", content=code_action, level=LogLevel.INFO)
        try:
            code_output = self.python_executor(code_action)
            execution_outputs_console = []
            if len(code_output.logs) > 0:
                execution_outputs_console += [
                    Text("Execution logs:", style="bold"),
                    Text(code_output.logs),
                ]
            observation = "Execution logs:\n" + code_output.logs
        except Exception as e:
            if hasattr(self.python_executor, "state") and "_print_outputs" in self.python_executor.state:
                execution_logs = str(self.python_executor.state["_print_outputs"])
                if len(execution_logs) > 0:
                    execution_outputs_console = [
                        Text("Execution logs:", style="bold"),
                        Text(execution_logs),
                    ]
                    memory_step.observations = "Execution logs:\n" + execution_logs
                    self.logger.log(Group(*execution_outputs_console), level=LogLevel.INFO)
            error_msg = str(e)
            if "Import of " in error_msg and " is not allowed" in error_msg:
                self.logger.log(
                    "[bold red]Warning to user: Code execution failed due to an unauthorized import - Consider passing said import under `additional_authorized_imports` when initializing your CodeAgent.",
                    level=LogLevel.INFO,
                )
            raise AgentExecutionError(error_msg, self.logger)

        truncated_output = truncate_content(str(code_output.output))
        observation += "Last output from code snippet:\n" + truncated_output
        memory_step.observations = observation

        if not code_output.is_final_answer:
            execution_outputs_console += [
                Text(
                    f"Out: {truncated_output}",
                ),
            ]
        self.logger.log(Group(*execution_outputs_console), level=LogLevel.INFO)
        memory_step.action_output = code_output.output
        yield ActionOutput(output=code_output.output, is_final_answer=code_output.is_final_answer)

    def to_dict(self) -> dict[str, Any]:
        """Convert the agent to a dictionary representation.

        Returns:
            `dict`: Dictionary representation of the agent.
        """
        agent_dict = super().to_dict()
        agent_dict["authorized_imports"] = self.authorized_imports
        agent_dict["executor_type"] = self.executor_type
        agent_dict["executor_kwargs"] = self.executor_kwargs
        agent_dict["max_print_outputs_length"] = self.max_print_outputs_length
        return agent_dict

    @classmethod
    def from_dict(cls, agent_dict: dict[str, Any], **kwargs) -> "CodeAgent":
        """Create CodeAgent from a dictionary representation.

        Args:
            agent_dict (`dict[str, Any]`): Dictionary representation of the agent.
            **kwargs: Additional keyword arguments that will override agent_dict values.

        Returns:
            `CodeAgent`: Instance of the CodeAgent class.
        """
        # Add CodeAgent-specific parameters to kwargs
        code_agent_kwargs = {
            "additional_authorized_imports": agent_dict.get("authorized_imports"),
            "executor_type": agent_dict.get("executor_type"),
            "executor_kwargs": agent_dict.get("executor_kwargs"),
            "max_print_outputs_length": agent_dict.get("max_print_outputs_length"),
            "code_block_tags": agent_dict.get("code_block_tags"),
        }
        # Filter out None values
        code_agent_kwargs = {k: v for k, v in code_agent_kwargs.items() if v is not None}
        # Update with any additional kwargs
        code_agent_kwargs.update(kwargs)
        # Call the parent class's from_dict method
        return super().from_dict(agent_dict, **code_agent_kwargs)


class SupervisorAgent(MultiStepAgent):
    """
    A strategic oversight agent that intervenes only at critical moments and leverages a functional verification agent.
    """
    def __init__(self, model: Model, summary_model: Model, verification_agent: Any, **kwargs):
        """
        Initialize SupervisorAgent.
        Args:
            model (Model): The main model used for strategic decision-making (e.g., GPT-4.1).
            summary_model (Model): The lightweight model used for observation purification (e.g., GPT-4.1).
            verification_agent (Any): The verification agent used for fact-checking.
        """
        super().__init__(
            model=model,
            name="SupervisorAgent",
            tools=[],
            **kwargs
        )
        self.summary_model = summary_model
        self.verification_agent = verification_agent
        self.VISIT_THRESHOLD = 5
        self.reset() # initialization

    def initialize_system_prompt(self) -> str:
        return ""
    
    def reset(self):
        """
        Reset all internal states before handling a completely new global task.
        """
        print("--- [Supervisor] Global trace has been reset. ---")
        self.global_trace: List[Session] = []
        self._session_counter = 0
        self._active_sessions: Dict[str, int] = {}  # quickly find currently running sessions.
        self.key_evidence: Dict[str, Dict[str, str]] = {}   # stores final_answer of search_agent
        self._evidence_counter = 0
        self.visited_urls = {}   # create a dictionary to store the number of times the URL was accessed.
        self.threshold_warning_issued = {} # key: session_id, value: boolean
    
    def record_step(self, step: ActionStep, agent_name: str, local_task: str, parent_agent_name: Optional[str] = None):
        """Record a step unconditionally into the global tracking history."""
        
        # Extract step_task directly from the current step.
        step_task = self._extract_task_from_step(step)

        session_id = self._get_or_create_session(agent_name, local_task, parent_agent_name)
        session = next((s for s in self.global_trace if s['session_id'] == session_id), None)
        
        # Add robustness checks
        if not session:
            print(f"--- [Supervisor Record] Error: Could not find or create session for agent '{agent_name}'. Step will not be recorded. ---")
            return
        tool_call_str = self._format_tool_call(step.tool_calls[0]) if step.tool_calls else "No tool called (planning or final answer)."

        if step.observations:
            # Extract the first line of the observation string.
            first_line = step.observations.split('\n', 1)[0]
            # Check if the first line contains the pattern we are looking for.
            if first_line.startswith("Address:"):
                try:
                    # Extract URL from "Address: [URL]"
                    url = first_line.split("Address:", 1)[1].strip()
                    # update counter
                    current_count = self.visited_urls.get(url, 0)
                    self.visited_urls[url] = current_count + 1
                    print(f"\n--- [Supervisor] Recorded interaction for URL: {url}. New count: {self.visited_urls[url]} ---\n")
                except IndexError:
                    pass

        new_step: StepRecord = {
            "step_number": step.step_number,
            "step_task": step_task,
            "tool_call": tool_call_str,
            "observation_summary": self._create_observation_summary(step.observations),
            # "observation_summary": (step.observations or "No observation.")[:200] + "...",
            "timestamp": time.time()
        }
        session['steps'].append(new_step)

    def end_session(self, agent_name: str, status: str = "completed"):
        """End the current active session of an Agent."""
        session_id = self._active_sessions.pop(agent_name, None) # Remove from active list
        if session_id:
            session = next((s for s in self.global_trace if s['session_id'] == session_id), None)
            if session:
                session['status'] = status
                session['end_time'] = time.time()
                print(f"--- [Supervisor] Session {session_id} for agent '{agent_name}' ended with status: {status}. ---")
                
    def _get_or_create_session(self, agent_name: str, task: str, parent_agent_name: Optional[str] = None) -> int:
        """
        Get or create a new Agent session and return its ID.
        Core logic: If an agent already has an active session but the task has changed,
        automatically end the old session and create a new one.
        """
        # Step 1: Check whether there is an active session for this agent
        active_session_id = self._active_sessions.get(agent_name)

        if active_session_id:
            # Step 2: If yes, verify whether the session’s task matches the current task
            session = next((s for s in self.global_trace if s['session_id'] == active_session_id), None)
            # Ensure the session object exists and the task matches
            if session and session['task'] == task:
                # The task is unchanged; continue using the current session
                return active_session_id
            else:
                # The task has changed, or the session data is inconsistent! End the old session
                # end_session will remove agent_name from _active_sessions
                self.end_session(agent_name, status="completed_by_new_task")
                # The following logic will create a new session for this agent

        # Step 3: If execution reaches here, no valid active session exists — create a new one
        self._session_counter += 1
        new_session_id = self._session_counter
        
        # Determine the parent session ID
        parent_session_id = self._active_sessions.get(parent_agent_name) if parent_agent_name else None

        # Create a new session object
        new_session: Session = {
            "session_id": new_session_id,
            "agent_name": agent_name,
            "parent_session_id": parent_session_id,
            "task": task,
            "steps": [],
            "status": "running",
            "start_time": time.time(),
            "end_time": None
        }
        
        # Add the new session to the global trace list and the active sessions dictionary
        self.global_trace.append(new_session)
        self._active_sessions[agent_name] = new_session_id
        
        print(f"--- [Supervisor] Started new Session {new_session_id} for agent '{agent_name}'. ---")
        
        return new_session_id
    
    def _format_tool_call(self, tool_call: Any) -> str:
        """Format a tool call object into a concise string."""
        # Avoid overly long arguments affecting readability
        args_str = json.dumps(tool_call.arguments)
        if len(args_str) > 150:
            args_str = args_str[:150] + "..."
        return f"Called tool '{tool_call.name}' with arguments: {args_str}"
    
    def _format_trace_for_prompt(self) -> str:
        """Format the global trace history into timestamped, hierarchical text suitable for prompt injection."""
        if not self.global_trace:
            return "No actions have been taken yet."

        root_start_time = self.global_trace[0]['start_time']    # Get the start time of the entire task as a baseline for relative time calculation
        formatted_lines = ["Execution History (Parent > Child):"]
        session_map = {s['session_id']: s for s in self.global_trace}
        
        def format_session(session_id, indent_level=0):
            session = session_map.get(session_id)
            if not session: return
            
            indent = "  " * indent_level
            
            # Calculate and format the start and end times of the session.
            session_start_delta = session['start_time'] - root_start_time
            status_str = f"(Status: {session['status']}, Started at T+{session_start_delta:.2f}s"
            if session.get('end_time'):
                session_end_delta = session['end_time'] - root_start_time
                duration = session['end_time'] - session['start_time']
                status_str += f", Ended at T+{session_end_delta:.2f}s, Duration: {duration:.2f}s)"
            else:
                status_str += ")"
            
            formatted_lines.append(f"{indent}* Session {session_id}: Agent '{session['agent_name']}' {status_str}")
            formatted_lines.append(f"{indent}  Task: {session['task']}")
            
            for step in session['steps']:
                # Calculate and format the timestamp for each step.
                step_time_delta = step['timestamp'] - root_start_time
                formatted_lines.append(f"{indent}  - Step {step['step_number']} (at T+{step_time_delta:.2f}s): {step['tool_call']}")
                formatted_lines.append(f"{indent}    Observation Summary: {step['observation_summary']}")

            # Recursively format subsessions
            for child_session in self.global_trace:
                if child_session.get('parent_session_id') == session_id:
                    format_session(child_session['session_id'], indent_level + 1)

        # Formatting starts from the root session (which has no parent session).
        for s in self.global_trace:
            if s.get('parent_session_id') is None:
                format_session(s['session_id'])
                
        return "\n".join(formatted_lines)
    
    def _format_local_trace_for_prompt(self, agent_name: str) -> str:
        """Format only the local tracing history of the current agent."""
        session_id = self._active_sessions.get(agent_name)
        if not session_id:
            return "No actions have been taken by this agent in the current session."
        
        session = next((s for s in self.global_trace if s['session_id'] == session_id), None)
        if not session or not session['steps']:
            return "No actions have been taken by this agent in the current session."

        formatted_lines = [f"Recent Local History for '{agent_name}':"]
        for step in session['steps'][-8:]: # show only the last 8 steps to save tokens.
             formatted_lines.append(f"- Step {step['step_number']}: {step['tool_call']}")
             formatted_lines.append(f"  Observation Summary: {step['observation_summary']}")
        return "\n".join(formatted_lines)
    
    def _create_observation_summary(self, observation: Optional[str]) -> str:
        """Intelligently create a summary based on the content of observation."""
        if not observation:
            return "No observation."

        # Rule 1: If the content is HTML, indicate it is an HTML document
        if observation.strip().startswith("<!DOCTYPE html>"):
            return "Received a long HTML document."

        # Rule 2: For formatted final_answer, extract key information
        if "### 1. Task outcome (short version):" in observation:
            try:
                short_answer = observation.split("### 1. Task outcome (short version):")[1].split("###")[0].strip()
                return f"Agent provided a final_answer. Short version: '{short_answer[:100]}...'"
            except IndexError:
                return "Agent provided a final_answer."

        # Rule 3: For search results, extract the title
        if "## Search Results" in observation:
            try:
                first_result_line = observation.split('\n')[1]
                return f"Received search results. Top result: '{first_result_line[:100]}...'"
            except IndexError:
                return "Received search results."

        # Default rule: truncate the content
        return (observation or "No observation.")[:150] + "..."
    
    def filter_search_results(self, observation: str) -> str:
        """
        Filter out URLs that have already been visited from the web_search observation results.
        """
        # If the observation is not a search result or the visited list is empty, return as is
        if "## Search Results" not in observation or not self.visited_urls:
            return observation

        print("--- [Supervisor] Filtering search results... ---")
        lines = observation.split('\n')
        filtered_lines = []
        
        # Regular expression to extract URLs from lines in the format "0. |Title](url)"
        url_pattern = re.compile(r"\]\((https?://.*?)\)")

        for line in lines:
            match = url_pattern.search(line)
            if match:
                url = match.group(1)
                # Skip the line if the URL has already been visited above the threshold
                if self.visited_urls.get(url, 0) >= self.VISIT_THRESHOLD:
                    print(f"--- [Supervisor] Filtering out URL (visit count >= {self.VISIT_THRESHOLD}): {url} ---")
                    continue
            filtered_lines.append(line)
        
        return '\n'.join(filtered_lines)

    def supervise_and_correct(self, step: ActionStep, agent_name: str, local_task: str, global_task: str, is_final_check: bool = False) -> ActionStep:
        print(f"\n--- [Supervisor] Reviewing action from '{agent_name}' ---")

        effective_agent_name = agent_name
        effective_local_task = local_task

        # Handle the case where search_agent calls final_answer_tool to return the final result
        if step.observations and "<summary_of_work>" in step.observations and agent_name == "manager_agent":
            print("--- [Supervisor] Detected final_answer from a sub-agent. Correcting context... ---")
            child_agent_name = None
            # In this step of manager_agent, the tool_call is the invocation of the sub-agent
            if step.tool_calls and step.tool_calls[0].name == 'python_interpreter':
                # Parse the sub-agent’s name from the python_interpreter's code arguments
                code_to_execute = step.tool_calls[0].arguments
                # Use a regex to look for the pattern "agent_name("
                # This pattern matches any word ending with _agent followed by an opening parenthesis
                match = re.search(r'(\w+_agent)\(', code_to_execute)
                
                if match:
                    # The first captured group is the sub-agent’s name
                    child_agent_name = match.group(1)

            if child_agent_name:
                # 从全局追踪历史中找到这个子 agent 的 session
                child_session = next((s for s in reversed(self.global_trace) if s.get('agent_name') == child_agent_name), None)
                
                if child_session:
                    effective_agent_name = child_agent_name
                    effective_local_task = child_session.get('task', local_task)
                    print(f"--- [Supervisor] Context switched to: agent='{effective_agent_name}', task='{effective_local_task[:100]}...' ---")
                else:
                    print(f"--- [Supervisor] WARNING: Could not find session for child agent '{child_agent_name}'. Using manager's context. ---")

        agent_name = effective_agent_name
        local_task = effective_local_task

        # 1. Determine supervision type (error handling, summarization, or fact-checking)
        supervision_type = "default"
        current_state_summary = ''
        is_inefficient, reason = self._check_for_inefficiency(agent_name)

        # Skip summary-type checks & inspect_text_tool
        if step.observations and ("1. Task outcome (short version):" in step.observations or "Additional Context on the Document and Question Asked" in step.observations or step.tool_calls[0].name =="inspect_text_tool"):
            if "<summary_of_work>" in step.observations:
                supervision_type = "sub_agent_result_synthesis"
            else:
                print("--- [Supervisor] Detected final_answer or 'inspect_text_tool' calling in observations, returning step. ---\n")
                return step
        
        elif step.error:
            supervision_type = "error_analysis"

        elif is_inefficient: # Inefficiency detection
            supervision_type = "inefficiency_analysis"
            current_state_summary = reason  # Use the detected specific reason as dynamic context

        elif step.observations and len(step.observations) > 3000: # Observation content too long
            supervision_type = "basic_extraction"

            if len(step.observations) > 600000:  # Set an extremely large value, effectively disabled
                supervision_type = "summarization" # Use intelligent summarization
    

        print(f"\n--- [Supervisor] Current supervison type: {supervision_type}\n")
        
        local_trace_str=self._format_local_trace_for_prompt(agent_name)
        global_trace_str=self._format_trace_for_prompt() if supervision_type == "inefficiency_analysis" or "final_verification" else None

        interaction_summary, basic_summary = self._summarize_interaction(step)

        if supervision_type == "basic_extraction":
            corrected_obs = basic_summary
            meta_prefix = "[Supervisor's Note: The original observation was too long and has been processed. The key information relevant to your task is provided below.]\n"
            final_obs = meta_prefix + corrected_obs
            print(f"\n--- [Supervisor] Original observation length: {len(step.observations)}")
            print(f"\n--- [Supervisor] Corrected observation: {final_obs}")
            step.observations = final_obs
            return step

        # 2. Build prompts for different supervision types
        prompt = self._build_prompt(
            supervision_type, 
            interaction_summary, 
            agent_name, 
            local_task, 
            global_task,
            current_state_summary,
            local_trace_str=local_trace_str,
            global_trace_str=global_trace_str
        )
        
        # 3. Let the Supervisor LLM generate verification/correction tasks
        response_message = self.model.generate([{"role": "user", "content": prompt}])
        response_str = response_message.content.strip() if response_message.content else ""
        token_usage_for_this_call = response_message.token_usage
        # verification_result = self.verification_agent.run(task=verification_task, reset=True)
        if token_usage_for_this_call:
            # Create a temporary, format-compliant “dummy” log object (dummy step log)
            dummy_timing = SimpleNamespace(duration=0.0) # We don't care about duration; 0 is fine
            dummy_step_log = SimpleNamespace(
                token_usage=token_usage_for_this_call,
                timing=dummy_timing
            )
            self.monitor.update_metrics(dummy_step_log)
        
        # 4. Execute actions based on the LLM response
        try:
            response_json = json.loads(response_str)

            analysis = response_json.get("analysis", "No analysis provided.")
            action = response_json.get("action", "approve")
            parameters = response_json.get("parameters", {})

            print(f"--- [Supervisor Analysis]: {analysis}")
            
            if action == "correct_observation":
                corrected_obs = parameters.get("new_observation", step.observations)
                if not isinstance(corrected_obs, str):
                    corrected_obs_str = "\n".join(map(str, corrected_obs))  # Choose an LLM-friendly format, e.g., a list joined by newlines
                else:
                    corrected_obs_str = corrected_obs
                # Add metadata prefix
                if supervision_type != "error_analysis":
                    meta_prefix = "[Supervisor's Note: The original observation was too long and has been processed. The key information relevant to your task is provided below.]\n"
                else:
                    meta_prefix = "[Supervisor's Note: The original observation contained errors and has been corrected. The corrected information is provided below.]\n"
                final_obs = meta_prefix + corrected_obs_str
                obs_len = len(step.observations) if step.observations is not None else 0
                print(f"--- [Supervisor] Original observation length: {obs_len}")
                print(f"--- [Supervisor] Corrected observation: {final_obs}")
                step.observations = final_obs
                step.error = None # Clear the error state after correction

            elif action == "run_verification":
                verification_task = parameters.get("task")
                if verification_task:
                    print(f"--- [Supervisor] Delegating verification task: '{verification_task}' ---")
                    # Use the fully featured verification_agent
                    verification_result = self.verification_agent.run(task=verification_task, reset=True)
                    # Tally token usage
                    verification_token_usage = self.verification_agent.monitor.get_total_token_counts()
                    if verification_token_usage:
                        # Create a temporary, format-compliant “dummy” log object (dummy step log)
                        dummy_timing = SimpleNamespace(duration=0.0) # We don't care about duration; 0 is fine
                        dummy_step_log = SimpleNamespace(
                            token_usage=verification_token_usage,
                            timing=dummy_timing
                        )
                        self.monitor.update_metrics(dummy_step_log)

                    print(f"--- [Supervisor] Verification result: '{verification_result}' ---")
                    
                    # Append the verification result to the observation
                    step.observations = (step.observations or "") + f"\n\n[Supervisor Verification]: {verification_result}"

            elif action == "provide_guidance":
                guidance = parameters.get("guidance", "")
                if guidance:
                    print(f"--- [Supervisor] Providing guidance to agent: '{guidance}' ---")
                    # Append guidance to the observation
                    step.observations = (step.observations or "") + f"\n\n[Supervisor Guidance]: {guidance} [Supervisor Analysis]: {analysis}"
                else:
                    print("--- [Supervisor] No guidance provided. Approving by default. ---")

            
            else:   # Includes action == "approve" or any other unexpected value
                if action != "approve":
                    print(f"--- [Supervisor] Received unexpected action '{action}'. Approving by default. ---")
                else:
                    print("--- [Supervisor] Approved the action. ---")

        except (json.JSONDecodeError, KeyError) as e:
            print(f"Supervisor could not parse the response: {e}. Approving by default.")
            # continue

        return step

    def _check_for_inefficiency(self, agent_name: str) -> Tuple[bool, Optional[str]]:
        """
        Check the global memory to determine whether a specific Agent exhibits inefficient behavior.
        Returns (is_inefficient, reason_description).
        """
        STEP_THRESHOLD = 8
        LOOP_THRESHOLD = 5

        # 1. Quickly locate the current agent's session_id from active_sessions
        session_id = self._active_sessions.get(agent_name)
        if not session_id:
            return False, None
        
        # 2. Find the corresponding session in global_trace
        session = next((s for s in self.global_trace if s['session_id'] == session_id), None)
        if not session:
            return False, None

        steps = session.get('steps', [])

        # --- Detection Rule 1: Number of steps exceeds the threshold ---
        # if len(steps) >= STEP_THRESHOLD and not self.threshold_warning_issued.get(session_id):
        if len(steps) >= STEP_THRESHOLD and len(steps) % STEP_THRESHOLD == 0:
            reason = (f"Agent '{agent_name}' has reached the step limit of {len(steps)} "
                      f"for its current task '{session['task']}'. Intervention is required.")
            return True, reason
        # --- Detection Rule 2: Repeatedly executing an inefficient action ---
        if len(steps) >= LOOP_THRESHOLD:
            # Retrieve the tool_calls of the most recent LOOP_THRESHOLD steps
            recent_tool_calls = [step['tool_call'] for step in steps[-LOOP_THRESHOLD:]]
            # Check whether these tool_calls are all identical
            if len(set(recent_tool_calls)) == 1 and not self.threshold_warning_issued.get(session_id, False):
                reason = (f"Agent '{agent_name}' seems to be stuck in a loop, "
                        f"repeatedly performing the same action: '{recent_tool_calls[-1]}'.")
                self.threshold_warning_issued[session_id] = True # Mark as triggered
                return True, reason
            
        return False, None

    def _summarize_interaction(self, step: ActionStep) -> str:
        """
        Summarize the current step while preserving the full `observations`.
        Only truncate auxiliary information to control token usage.
        """
        summary = []
        summarized_obs = ""
        if step.model_output:
            # We can still truncate the thought
            summary.append(f"- Agent's Thought: {step.model_output[:2000]}...")
        if step.tool_calls:
            for tool_call in step.tool_calls:
                # Truncate overly long arguments
                arguments_str = str(tool_call.arguments)
                if len(arguments_str) > 500:
                    arguments_str = arguments_str[:500] + "..."
                summary.append(f"- Tool Call: {tool_call.name}({json.dumps(arguments_str)})")

        
        if step.observations:
            original_obs = step.observations
            OBS_SUMMARY_THRESHOLD = 3000  # Set a threshold; only observations longer than this need AI summarization
            if len(original_obs) > OBS_SUMMARY_THRESHOLD:
                print(f"--- [Supervisor] Observation is long ({len(original_obs)} chars). Engaging summary model. ---")
                
                # To prevent the input to the summarization model from exceeding its limit, perform a final truncation
                MAX_INPUT_FOR_SUMMARY = 100000
                if len(original_obs) > MAX_INPUT_FOR_SUMMARY:
                    half_len = MAX_INPUT_FOR_SUMMARY // 2
                    obs_for_summary = original_obs[:half_len] + "\n\n[... CONTENT TRUNCATED FOR SUMMARY MODEL ...]\n\n" + original_obs[-half_len:]
                    print(f"--- [Supervisor] Observation too long for summary model. Truncated to {len(obs_for_summary)} chars. ---")
                else:
                    obs_for_summary = original_obs

                # Generate a context-aware prompt
                summary_prompt = self._get_summary_prompt(observation=obs_for_summary)
                
                # Call the summary model to generate a summary
                try:
                    summary_message = self.summary_model.generate([{"role": "user", "content": summary_prompt}])
                    summarized_obs = summary_message.content.strip()
                    print(f"--- [Supervisor] Summarization completed. ---")
                    summary.append(f"- Observations (summarized by Assistant): {summarized_obs}")


                except Exception as e:
                    print(f"--- [Supervisor] Error during summarization: {e}. Falling back to truncation. ---")
                    # If summarization fails, fall back to the original truncation strategy to ensure robustness
                    half_length = OBS_SUMMARY_THRESHOLD // 2
                    truncated_obs = original_obs[:half_length] + "\n\n[... CONTENT TRUNCATED ...]\n\n" + original_obs[-half_length:]
                    summary.append(f"- Observations (fallback truncation): {truncated_obs}")

            else:
                # If the observation is not long, use it directly
                summary.append(f"- Observations: {original_obs}")

        if step.error:
            summary.append(f"- Error: {step.error}")
        
        return "\n".join(summary), summarized_obs
    
    def _extract_task_from_step(self, step: ActionStep) -> str:
        """
        Extract the text from the current ActionStep that best represents the step's task intent.
        Priority: 1) Thought  2) Tool Call
        """
        # Priority 1: Extract intent from the Thought
        if step.model_output and "Thought:" in step.model_output:
            thought = step.model_output.split("Thought:")[-1].strip()
            # Simple cleanup
            thought = thought.replace("<code>", "").replace("</code>", "").strip()
            if thought:
                return thought

        # Priority 2: If there is no valid Thought but there is a tool call, use the tool call as the best task description
        if step.tool_calls:
            return self._format_tool_call(step.tool_calls[0])

        return "No specific step task found in recent history."

    def _build_prompt(self, supervision_type: str, summary: str, agent_name: str, local_task: str, global_task: str, current_state_summary: str = "", local_trace_str: str = "", global_trace_str: Optional[str] = None) -> str:
        # general prompt template
        base_prompt = f"""
        Role: You are an expert supervisor in a multi-agent system. Your role is to monitor an agent's actions, ensure alignment with the main goal, correct errors, and optimize the workflow.
        Objective: The overall objective (Global Task) is: "{global_task}"

        Agent context:
        You are currently reviewing an action from the agent '{agent_name}'.
        This agent's specific sub-task (Local Task) is: '{local_task}'.
        Here is the current local execution trace: {local_trace_str}
        Here is the summary of the agent's latest thought process and the resulting observation:
        {summary}
        {current_state_summary}
        
        Rules:
        1.  Assess Necessity: First, assess if intervention is truly necessary. If the agent's action and observation are correct and productive, use the "approve" action. Avoid unnecessary interventions.
        2.  Be Decisive: When an intervention is needed, choose the most effective action to move the project forward.
        3.  Output Format: Your response MUST be a valid JSON object.

        Actions:
        Your available actions are:
        -  `approve`: The agent's action is correct and requires no changes.
        -  `correct_observation`: The observation contains errors or can be significantly improved (e.g., filtered, summarized, extracted). You will provide a corrected version.
        -  `provide_guidance`: The observation is correct, but the agent's thinking or next step is flawed. You will provide a hint or corrected reasoning to guide the agent.
        -  `run_verification`: You have doubts about the factual accuracy of the observation and need an external assistant to verify it.

        Your response MUST be a JSON object with the following structure:
        {{
            "analysis": "Your brief analysis of the situation, explaining your reasoning for the chosen action.",
            "action": "ONE of the available actions: ['approve', 'correct_observation', 'provide_guidance', 'run_verification']",
            "parameters": {{
                "new_observation": "IF action is 'correct_observation', provide the refined observation here.",
                "guidance": "IF action is 'provide_guidance', provide a clear hint or instruction for the agent's next thought process.",
                "task": "IF action is 'run_verification', provide the verification question for the assistant."
            }}
        }}
        """

        
        if supervision_type == "error_analysis":
            return base_prompt + f"""
            **Role**: You are an expert Debugger and AI Diagnostician. Your primary goal is to understand the root cause of an error and provide the most effective solution to get the agent back on track.

            **Situation**: The agent's last action resulted in a critical error, which is detailed in the "summary" of the agent's action below. **Approval is not an option; you must intervene.**

            **--- Your Debugging Framework (MANDATORY) ---**
            Before generating your JSON output, you MUST follow this structured thinking process:

            **Step 1: Analyze the Error**
            - What is the precise error message and type (e.g., `Tool Error`, `Python Exception`, `APIError`)?

            **Step 2: Examine the Context**
            - Review the `local_execution_trace` and the agent's `thought` process leading to the error.
            - What was the agent *trying* to accomplish?
            - Was the tool call or code it executed (`summary` section) syntactically correct but logically flawed?

            **Step 3: Root Cause Diagnosis**
            - Based on the error and context, what is the single most likely root cause?
            - (e.g., "The agent passed a natural language string to a tool expecting a mathematical expression.", "The agent is trying to access a file that does not exist.")

            **Step 4: Formulate a Solution Strategy**
            - Based on the root cause, determine the best intervention:
                - If the error can be fixed by correcting the agent's **next thought process or action**, choose `provide_guidance`. This is the most common case for logical errors.
                - If the error was caused by faulty information in the **previous observation** that the agent is now acting upon, choose `correct_observation`.
                - If you lack critical information to solve the error and need to consult an external source, choose `run_verification`.

            **--- YOUR ACTIONABLE OUTPUT (JSON) ---**
            Based on your diagnosis, provide your final decision in the JSON format.
            """

        elif supervision_type == "sub_agent_result_synthesis":
            return base_prompt + f"""
            **Role**: You are an expert Intelligence Analyst working for a manager agent. Your task is to process a verbose report from a sub-agent (e.g., a search specialist) and synthesize a direct, comprehensive, and clean answer for your manager.

            **--- YOUR INPUTS ---**

            **1. The Manager's Request (Immediate Goal)**:
            - "{local_task}"

            **2. The Overall Mission (Global Goal)**:
            - "{global_task}"

            **3. The Sub-Agent's Full Field Report (Raw Observation)**:
            ```
            {summary} 
            ```
            (Note: The 'summary' variable here contains the sub-agent's full, multi-part final_answer)

            **--- YOUR CRITICAL TASK ---**

            Your sole task is to read the ENTIRE "Field Report" (including the short version, detailed version, and the summary of work) and synthesize a single, clean, and self-contained response that **fully and completely** answers the "Manager's Request".
            **Critical Rule for Synthesis**:
            **Preserve Semantic Structure**: When synthesizing, you MUST maintain the original information's hierarchy. If the source contains headings, chapters, articles, or numbered/bulleted lists, these structural elements **MUST be preserved** in your output to give context to the data points below them. **Do not flatten a structured document into a simple, unstructured block of text.**
            **Your Internal Thought Process (MANDATORY)**:
            1.  **Deconstruct the Manager's Request**: What are the specific pieces of information the manager is asking for? Create a mental checklist.
            2.  **Scan the Entire Report**: Read all parts of the sub-agent's report to find the answers for your checklist. The most valuable details are often in the "extremely detailed version" or the "summary of work".
            3.  **Synthesize, Don't Just Extract**: Combine the findings into a coherent, fluent, and direct answer. Do not simply copy the "short version". Your answer must be comprehensive enough to prevent the manager from needing to ask follow-up questions.

            **Example**:
            - **Manager's Request**: "Find the number of encoder layers in the BERT-Base model."
            - **Sub-Agent's Report**: (A long text containing "Short version: 12 layers", "Detailed version: ...Section 3 of the paper states L=12 for BERT-Base...", etc.)
            - **Your Ideal Synthesized Output**: "The BERT-Base model has 12 encoder layers (L=12), as specified in Section 3 of the original paper by Devlin et al., 2018."

            **Action**: Your action MUST be `"correct_observation"`.
            **Parameter**: Provide your final, synthesized answer in the `"new_observation"` parameter.
            """

        
        elif supervision_type == "inefficiency_analysis":
            return base_prompt + f"""
            **Role**: You are a pragmatic and experienced AI workflow strategist. Your primary goal is to ensure the agent team achieves its task in the most efficient way **from its current state**.

            **Situation**: An inefficiency trigger has been activated for agent '{agent_name}'. **This is a flag for you to review, NOT a confirmation of a problem.** The agent might be engaged in a necessary, methodical process.

            **Global Execution Trace**:
            {global_trace_str}

            **--- Your Decision Framework (MANDATORY) ---**
            Before generating your JSON output, you MUST follow this structured thinking process:

            **Step 1: Goal & Plan Inference**
            - Based on the `Global Execution Trace`, what is the agent's immediate, implicit plan?
            - (e.g., "The agent is clearly trying to collect all rows of a data table by repeatedly using `page_down`.")

            **Step 2: Progress Assessment**
            - Is the agent making tangible progress towards its inferred goal?
            - Is each new step yielding new, relevant information (even if it's just more rows of the same table)?
            - How close is the agent to completing this sub-task? (e.g., "It is on page 10 of 13, it is very close to getting all the data.")

            **Step 3: Cost-Benefit Analysis of Intervention**
            - **Compare two costs**:
                - **Cost A**: The estimated cost (time, tokens) of letting the agent **continue** its current, perhaps clumsy, path to completion.
                - **Cost B**: The estimated cost of **interrupting** the agent, guiding it to a new path, and having it **start over** on that new path.
            - **CRITICAL QUESTION**: Is the agent "one step away" from solving its sub-task? If so, interrupting it is almost always the wrong decision, even if a theoretically "better" path exists.

            **Step 4: Decision and Justification**
            - Based on the analysis above, decide between `approve` and `provide_guidance`.

            **--- YOUR ACTIONABLE OUTPUT (JSON) ---**

            You must choose ONE of the following two actions:

            **1. If you decide the agent should continue:**
            - **Condition**: The agent is making clear, incremental progress AND is close to completing its sub-task (Cost A < Cost B).
            - **Action**: MUST be `"approve"`.
            - **Analysis**: Briefly explain *why* the agent's current path, while perhaps repetitive, is the most pragmatic way forward from its current state. (e.g., "The agent is methodically paginating through a table to gather all data. Although repetitive, this is a valid and necessary process. It is on page 10 of 13 and about to succeed. Intervention would be disruptive.")

            **2. If you decide the agent is truly stuck:**
            - **Condition**: The agent is in a non-productive loop (e.g., getting the same observation repeatedly) OR the alternative path is overwhelmingly more efficient and the agent is not close to finishing (Cost B << Cost A).
            - **Action**: MUST be `"provide_guidance"`.
            - **Analysis**: Briefly explain the root cause of the inefficiency.
            - **Guidance**: The `guidance` parameter MUST contain a clear, concrete, and actionable instruction that represents a *significantly* better strategy. (e.g., "Instead of scrolling, use the `web_search` tool with the query 'who had the most BB for the 1977 Yankees' to get the answer directly.")
            """
        
        else: # default general review [just as a backup, should not reach here often]
            return base_prompt + """
            This is a general review. Analyze the agent's action for correctness, and alignment with the global task.
            - **IMPORTANT**: Do not handle major errors, long observations, or obvious inefficient loops. Those are handled by specialized triggers. Focus on subtle issues.
            - **Critically evaluate source reliability in context.** An official source is generally trustworthy, but may lack historical precision (e.g., using modern country names for past events). For nuanced historical facts, if the agent relies on a single source, consider guiding it to perform cross-validation with an independent source (like Wikipedia or a historical archive).
            - If everything is perfect and aligns with the task, set action to "approve".
            - If you see a minor issue you can fix, or you can make the observation more concise or relevant, set action to "provide_guidance" and provide this refined text in the "guidance" parameter.
            - If you spot a potential factual error or hallucination that needs checking, set action to "run_verification" and provide a "task" parameter for your verification assistant.
            """

    def _get_summary_prompt(self, observation: str) -> str:
        """
        A general, stateless Prompt designed to structurally cleanse information.
        """
        
        return f"""
        # Role: AI Agent Observation Compressor
        You are a specialized data compression model for an AI agent. Your sole purpose is to process raw observations (HTML, text, etc.) and reduce their token count while strictly preserving their structural integrity and all potentially useful information.

        **## Core Principles ##**
        1.  **Context-Agnostic:** You have NO knowledge of the agent's overall goal or past actions. Do NOT try to infer the task. Your compression must be generic and unbiased, preserving information that could be useful for ANY potential task.
        2.  **Preservation Over Compression:** It is critically important to avoid over-summarization. Losing a potentially key piece of information is a greater failure than not compressing enough. The output must retain enough detail for the agent to make informed decisions.
        3.  **Structural Integrity:** The output's structure (headings, lists, paragraphs, HTML hierarchy) must mirror the input's structure. Do not merge distinct sections.
        4.  **Preserve Metadata**: Always keep leading lines like `"Address: ..."`, `"Viewport: ..."` verbatim. 

        **## Compression Rules ##**
        Based on the type of content, apply the following rules:

        ### **Type 1: For HTML Content**
        Your goal is to simplify the HTML to its semantic and structural core, removing presentation-focused noise.
        1.  **Simplify Tags:** Remove non-essential attributes.
            -   **REMOVE attributes like:** `class`, `id`, `style`, `onclick`, `onmouseover`, and any `data-*` or `js-*` attributes. These are primarily for styling and scripting, not for content structure.
            -   **KEEP essential attributes:** `href`, `src`, `alt`, `title`, `aria-label`, `placeholder`, `value`. These attributes contain crucial information for navigation and interaction.
        2.  **Remove Non-Visible Content:** Completely remove `<script>`, `<style>`, and HTML comment `` blocks.
        3.  **Preserve Content:** Keep ALL text content within tags exactly as it is. Do not summarize the text inside the HTML.
        4.  **Whitespace:** Condense multiple spaces, newlines, and tabs in the HTML structure into a single space where appropriate to improve readability without losing structure.
        **Example:**
        * **Original:** `<td class='datacolBoxR' style='padding: 5px;'><a href="/wiki/some_link" title="Some Link">25</a></td>`
        * **Compressed:** `<td><a href="/wiki/some_link" title="Some Link">25</a></td>`

        ### **Type 2: For Plain Text Content**
        Your goal is to make the text more concise without losing factual information or its original layout.
        1.  **Retain Key Information:** Fully preserve all named entities (e.g., people, organizations, locations), numbers, dates, codes, IDs, and any factual data.
        2.  **Condense Prose:** For descriptive sentences or paragraphs, rephrase them to be more direct. Remove filler words, redundant phrases, and overly elaborate adjectives. However, do NOT eliminate the sentence entirely.
        3.  **Maintain Structure:** If the input text has multiple paragraphs, bullet points, or numbered lists, the output MUST have the same structure. Do not flatten a list into a single paragraph.
        **Example:**
        * **Original:** "The company, officially known as The International Business Machines Corporation (IBM), is a very large and influential American multinational technology corporation that has its headquarters located in Armonk, New York, and it was originally founded all the way back in 1911."
        * **Compressed:** "The International Business Machines Corporation (IBM) is an American multinational technology corporation headquartered in Armonk, New York, founded in 1911."

        **## Final Instruction ##**
        Process the following observation according to the rules above. Provide only the compressed output, without any extra text, explanation, or preamble.
        {observation}
        """