"""
Evaluate LLM on Sudoku puzzles using an API.

We call the LLM repeatedly:
  1) Provide an initial puzzle prompt.
  2) LLM responds with a single forced placement (e.g., <ANSWER>\nr3c6: 5\n</ANSWER>).
  3) We check if that placement is valid and correct based on the puzzle's known solution.
  4) If correct, we update the board and continue; if incorrect, we stop.
  5) Continue until the puzzle is solved or we reach a maximum number of steps.

Example Usage:
--------------
export OPENAI_API_KEY="your_openai_api_key"
export DATASET="challenge_100"
export API="openai"
export MODEL="gpt-4o-mini-2024-07-18"
python -m eval.run \
    --dataset ${DATASET} \
    --output_csv ../data/benchmark_results/${DATASET}/${MODEL}.csv \
    --api ${API} \
    --model ${MODEL} \
    --batch_size 20

Output:
-------
A CSV file with columns:
[
    "data_source",
    "puzzle_id",
    "model",
    "num_empty_cells",
    "shuffle_seed",
    "n_response_idx",
    "n_history_turns",
    "setting",
    "conversation",
    "num_rounds",
    "num_correct_placements",
    "final_solved",
    "final_board",
    "total_input_tokens",
    "total_output_tokens",
]

Plus a summary of average correctness/final-solved rates in stdout.
"""

import argparse
import asyncio
import json
import os
import sys
from typing import Any, Dict, List, Optional, Union
import uuid

# import aiohttp
# import anthropic
import datasets
import jinja2
# import openai
import pandas as pd
from tqdm import tqdm
# from transformers import AutoTokenizer
# try:
#     from vllm import AsyncLLMEngine, SamplingParams
#     from vllm.engine.arg_utils import AsyncEngineArgs
# except ImportError:
#     print("vllm not installed. Please install vllm to use it.")
#     AsyncLLMEngine = None
#     SamplingParams = None
#     AsyncEngineArgs = None

# Add smolagents imports
try:
    from smolagents import CodeAgent, LiteLLMModel
except ImportError:
    print("smolagents not installed. Please run `uv add smolagents litellm`")
    CodeAgent = None
    LiteLLMModel = None

# Add openai-agents imports
try:
    from agents import Agent as OpenAIAgent, Runner, Usage as OpenAIUsage, ModelSettings as OpenAIModelSettings # Import necessary components and alias Agent and Usage
except ImportError:
    print("openai-agents not installed. Please run `uv add openai-agents`")
    OpenAIAgent = None
    Runner = None
    OpenAIUsage = None
    OpenAIModelSettings = None

# Add langgraph imports
try:
    from langgraph.prebuilt import create_react_agent
    from langchain_openai import ChatOpenAI
except ImportError:
    print("langgraph or langchain-openai not installed. Please run `uv add langgraph langchain-openai`")
    create_react_agent = None
    ChatOpenAI = None

# Add pydantic-ai imports
try:
    from pydantic_ai import Agent as PydanticAgent
except ImportError:
    print("pydantic-ai not installed. Please run `uv add pydantic-ai`")
    PydanticAgent = None

# Add agno imports
try:
    from agno.agent import Agent as AgnoAgent, RunResponse as AgnoRunResponse
    from agno.models.openai import OpenAIChat as AgnoOpenAIChat
except ImportError:
    print("agno not installed. Please run `uv add agno`")
    AgnoAgent = None
    AgnoRunResponse = None
    AgnoOpenAIChat = None

# Add autogen imports
try:
    from autogen_agentchat.agents import AssistantAgent
    from autogen_agentchat.messages import TextMessage
    from autogen_ext.models.openai import OpenAIChatCompletionClient
except ImportError:
    print("autogen not installed. Please run `uv add autogen-agentchat autogen-ext[openai]`")
    AssistantAgent = None
    TextMessage = None
    OpenAIChatCompletionClient = None

from eval.prompts import (
    BOARD_PROMPT,
    PREFILLED_ASSISTANT_RESPONSE,
    RULE_PROMPT,
)
from eval.utils import (
    extract_action_from_response,
    pretty_print_visual_elements,
    random_fill_hints,
    smolagents_output_to_string,
    convert_to_pydanticai_messages,
)
from sudoku_ds import (
    SudokuAction,
    SudokuBoard,
)


async def process_one(
    args: argparse.Namespace,
    request: Dict,
    model: str,
) -> Dict:
    # Load data
    rules = request["rules"]
    current_board_ascii = request["initial_board"]
    solution_ascii = request["solution"]
    rows = request["rows"]
    cols = request["cols"]
    visual_elements = request["visual_elements"]
    if pd.isna(visual_elements) or visual_elements == "":
        visual_elements = None
    n_history_turns = request["n_history_turns"]

    # Construct setting string
    settings = []
    if n_history_turns == -1:
        settings.append("full-history")
    else:
        assert n_history_turns >= 0
        settings.append(f"{n_history_turns}-history-turns")
    if len(settings) == 0:
        setting = "default"
    else:
        setting = "_".join(settings)

    # Pretty print visual elements
    if visual_elements is None:
        pretty_visual_elements = None
    else:
        visual_elements = json.loads(visual_elements)
        pretty_visual_elements = pretty_print_visual_elements(visual_elements)

    # Construct boards
    solution_board = SudokuBoard.from_ascii(solution_ascii, rows, cols)
    current_board = SudokuBoard.from_ascii(current_board_ascii, rows, cols)
    max_rounds = current_board.to_ascii(unfilled=".").count(".")

    # Initial conversation
    rule_prompt = jinja2.Template(RULE_PROMPT).render(
        rows=rows,
        cols=cols,
        rules=rules,
        pretty_visual_elements=pretty_visual_elements,
    )
    # `history_conversation`` is for recording
    # Actual input conversation will be constructed before calling API
    history_conversation = [
        {"role": "user", "content": rule_prompt},
        {"role": "assistant", "content": PREFILLED_ASSISTANT_RESPONSE}
    ]

    # Initialize smolagents CodeAgent if api is smolagents
    agent = None
    runner_input = None # Initialize runner_input for openai_agents
    total_usage = None # Initialize usage counter

    if args.agent_framework == "smolagents":
        if CodeAgent is None or LiteLLMModel is None:
            raise ImportError("smolagents is not installed. Please run `pip install -r requirements.txt`")
        # Use args.model as model_id for LiteLLMModel
        llm_model = LiteLLMModel(
            model_id=model,
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_tokens,
            # top_k is not directly supported by LiteLLMModel, configure via model params if needed
        )
        agent = CodeAgent(
            tools=[], # No tools needed for Sudoku
            model=llm_model,
        )
    elif args.agent_framework == "openai_agents":
        if OpenAIAgent is None or Runner is None:
             raise ImportError("openai-agents is not installed. Please run `uv add openai-agents`")
        # Instructions combine rules and initial assistant prompt
        agent_instructions = f"{rule_prompt}\n\n{PREFILLED_ASSISTANT_RESPONSE}"
        agent = OpenAIAgent( # Use the aliased name
            name="SudokuSolver",
            instructions=agent_instructions,
            model=model, # Use the model specified in args
            model_settings=OpenAIModelSettings(
                temperature=args.temperature,
                top_p=args.top_p,
                max_tokens=args.max_tokens,
            )
        )

        # Initialize usage counter for openai_agents
        if OpenAIUsage is None:
            raise ImportError("openai-agents is not installed. Please run `uv add openai-agents`")
        total_usage = OpenAIUsage()
    elif args.agent_framework == "langgraph":
        if ChatOpenAI is None or create_react_agent is None:
             raise ImportError("langgraph or langchain-openai not installed. Please run `uv add langgraph langchain-openai`")

        # Process model name for langgraph with ChatOpenAI
        processed_model_name = model
        if "/" in model:
            prefix, suffix = model.split("/", 1)
            if prefix == "openai":
                processed_model_name = suffix
            else:
                raise ValueError(f"Unsupported model prefix for langgraph: {prefix}. Only 'openai/' is supported.")
        # If no "/" found, use the original model name

        if prefix == "openai":
            llm = ChatOpenAI(
                model=processed_model_name,
                temperature=args.temperature,
                top_p=args.top_p,
                max_tokens=args.max_tokens,
            )

        # LangGraph's ReAct agent doesn't explicitly take instructions like OpenAI Agents.
        agent = create_react_agent(llm, tools=[])

        # Initialize usage counter for langgraph
        total_usage = {"input_tokens": 0, "output_tokens": 0}

    elif args.agent_framework == "pydanticai":
        if PydanticAgent is None:
            raise ImportError("pydantic-ai is not installed. Please run `uv add pydantic-ai`")

        # Process model name for pydantic-ai
        # PydanticAI expects format like "openai:gpt-4"
        processed_model_id = model
        if "/" in model:
            prefix, suffix = model.split("/", 1)
            if prefix == "openai": # Currently only support openai prefix conversion
                processed_model_id = f"{prefix}:{suffix}"
            # Add other provider conversions if needed
            # else: raise ValueError(f"Unsupported model prefix for pydantic-ai: {prefix}")
        # If no known prefix, use the original model name (might need adjustments based on pydantic-ai support)

        agent = PydanticAgent(
            model=processed_model_id,
            model_settings={
                'temperature': args.temperature,
                'max_tokens': args.max_tokens,
                'top_p': args.top_p,
            },
        )

        # Initialize usage counter for pydantic-ai
        total_usage = {"input_tokens": 0, "output_tokens": 0}

    elif args.agent_framework == "agno":
        if AgnoAgent is None or AgnoOpenAIChat is None:
            raise ImportError("agno not installed. Please run `uv add agno`")

        # Process model name for agno (expects format like 'gpt-4.1-...')
        processed_model_name = model
        if "/" in model:
            prefix, suffix = model.split("/", 1)
            if prefix == "openai":
                processed_model_name = suffix
            else:
                # Assuming non-openai prefixes might be handled directly or raise error later
                # Adjust this logic based on supported models for agno
                pass # Use suffix or original model name depending on agno's expectation

        # Currently only supporting OpenAI models with agno
        if prefix == "openai":
             llm = AgnoOpenAIChat(
                 id=processed_model_name,
                 temperature=args.temperature,
                 max_tokens=args.max_tokens,
                 top_p=args.top_p,
             )
        else:
            raise ValueError(f"Unsupported model provider for agno: {prefix}. Only 'openai/' is supported.")

        agno_tools = []
        if args.use_reasoning_tools:
            try:
                from agno.tools.reasoning import ReasoningTools
                agno_tools.append(ReasoningTools(add_instructions=True))
            except ImportError:
                 print("Warning: ReasoningTools not found in agno. Install agno with reasoning support if needed.")
                 # Optionally raise an error or proceed without the tool

        agent = AgnoAgent(
            model=llm,
            tools=agno_tools, # Use the dynamically created list of tools
        )
        
        # Initialize usage counter for agno
        total_usage = {"input_tokens": 0, "output_tokens": 0}

    elif args.agent_framework == "autogen":
        if AssistantAgent is None or TextMessage is None or OpenAIChatCompletionClient is None:
            raise ImportError("autogen not installed. Please run `uv add autogen-agentchat autogen-ext[openai]`")

        # Process model name for AutoGen (expects 'openai/gpt-...' or similar)
        processed_model_name_for_client = model
        if "/" in model:
            prefix, suffix = model.split("/", 1)
            if prefix != "openai": # Currently only support openai prefix
                 raise ValueError(f"Unsupported model prefix for autogen: {prefix}. Only 'openai/' is supported.")
            # Use the suffix for OpenAIChatCompletionClient
            processed_model_name_for_client = suffix
        # else:
             # If no prefix, assume it's an OpenAI model name
             # processed_model_name_for_client = model # Already initialized

        model_client = OpenAIChatCompletionClient(
            model=processed_model_name_for_client,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            top_p=args.top_p,
        )

        # ReAct prompt (reference: https://microsoft.github.io/autogen/0.2/docs/topics/prompting-and-reasoning/react/)
        ReAct_prompt = """Answer the following questions as best you can. You have access to tools provided.

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take
Action Input: the input to the action
Observation: the result of the action
... (this process can repeat multiple times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question
"""

        agent = AssistantAgent(
            name="SudokuSolverAgent",
            model_client=model_client,
            system_message=ReAct_prompt,
        )

        # Initialize usage counter for autogen
        total_usage = {"input_tokens": 0, "output_tokens": 0}

    num_correct_placements = 0
    assistant_response = None # Initialize assistant_response
    for round_idx in range(max_rounds):
        round_str = f"Round {round_idx + 1} / {max_rounds}"

        ##################
        ## Get response ##
        ################## 

        # Construct user prompt describing the current board
        board_prompt = jinja2.Template(BOARD_PROMPT).render(
            current_board=current_board.to_spaced_ascii(unfilled="."),
        )
        history_conversation.append({"role": "user", "content": board_prompt})

        # Construct input conversation
        # If full history, include all history turns
        if n_history_turns == -1:
            input_conversation = [
                {"role": message["role"], "content": message["content"]}
                for message in history_conversation
            ]
        # Otherwise
        # - First two prompts are fixed (rule prompt and prefilled assistant response)
        # - Last prompt is the current board
        # - In between, we add the most recent history turns
        else:
            input_conversation = [
                {"role": message["role"], "content": message["content"]}
                for message in \
                    history_conversation[:2] \
                    + history_conversation[2:-1][-2*n_history_turns:] \
                    + history_conversation[-1:]
            ]

        # Call agent
        if args.agent_framework == "smolagents":
            try:
                # For the first turn, combine rule and board prompts, and reset history
                if round_idx == 0:
                    # Use rule_prompt + PREFILLED_ASSISTANT_RESPONSE + current_board_prompt as the first user message
                    # Note: smol-agent's run expects a single string. We format it simply.
                    initial_user_message = f"User: {rule_prompt}\n\nAssistant: {PREFILLED_ASSISTANT_RESPONSE}\n\nUser: {board_prompt}"
                    # We need to run agent in a separate thread or use async version if available
                    # For now, using sync version, assuming it's okay for asyncio context (might block)
                    # TODO: Check smolagents documentation for async run or use asyncio.to_thread
                    # The initial 'history_conversation' is not directly used by agent.run's state
                    # We need to manually manage the state via reset=True/False
                    result = await asyncio.to_thread(
                        agent.run,
                        initial_user_message,
                        reset=True # Reset agent's internal memory
                    )
                # For subsequent turns, use only the current board prompt and don't reset
                else:
                    result = await asyncio.to_thread(
                        agent.run,
                        board_prompt,
                        reset=False # Maintain agent's internal memory
                    )
                assistant_response = smolagents_output_to_string(result)
            except Exception as e:
                # TODO: Implement retry logic similar to the original call_api if needed
                print(f"[Fail] {round_str}. Error calling smolagent: {e}")
                # Use the previous response if available, otherwise break
                if assistant_response is None:
                    break
                # If there was a previous response, try to reuse it or handle error
                print(f"Using previous response due to error.")

        elif args.agent_framework == "openai_agents":
            try:
                # For the first turn, runner_input is just the board prompt
                if round_idx == 0:
                    # The instructions already contain the rules and prefilled response
                    # So the first user message is just the initial board state
                    runner_input = [{"role": "user", "content": board_prompt}]
                # For subsequent turns, append the new board prompt to the previous conversation history
                else:
                    # Ensure runner_input is a list (should be from result.to_input_list())
                    if not isinstance(runner_input, list):
                         print(f"[Fail] {round_str}. Invalid runner_input state.")
                         break
                    runner_input.append({"role": "user", "content": board_prompt})

                # Run the agent
                # TODO: Add trace context manager if needed for detailed tracing
                result = await Runner.run(agent, runner_input)
                assistant_response = result.final_output

                # Prepare input for the next turn using the full history from the result
                runner_input = result.to_input_list()

                # Accumulate token usage
                if total_usage is not None:
                    for resp in result.raw_responses:
                        total_usage.add(resp.usage)

            except Exception as e:
                print(f"[Fail] {round_str}. Error calling OpenAI Agent: {e}")
                # Use the previous response if available, otherwise break
                if assistant_response is None:
                    break
                print(f"Using previous response due to error.")

        elif args.agent_framework == "langgraph":
            try:
                # Invoke the LangGraph agent asynchronously
                # The history is passed in the input dictionary under the 'messages' key
                result = await agent.ainvoke({"messages": input_conversation})

                # Extract the assistant's response message object (should be a dict)
                assistant_message_dict = result["messages"][-1]
                assistant_response = assistant_message_dict.content

                # Accumulate token usage if available in response_metadata
                usage = assistant_message_dict.response_metadata["token_usage"]
                total_usage["input_tokens"] += usage.get('prompt_tokens', 0)
                total_usage["output_tokens"] += usage.get('completion_tokens', 0)

            except Exception as e:
                print(f"[Fail] {round_str}. Error calling LangGraph Agent: {e}")
                # Use the previous response if available, otherwise break
                if assistant_response is None:
                    break
                print(f"Using previous response due to error.")

        elif args.agent_framework == "pydanticai":
            try:
                # Convert history to PydanticAI format
                pydantic_history = convert_to_pydanticai_messages(input_conversation[:-1], model)

                # Run PydanticAI agent
                result = await agent.run(user_prompt=input_conversation[-1]["content"], message_history=pydantic_history)
                assistant_response = result.data

                # Get and accumulate token usage
                usage = result.usage()
                total_usage["input_tokens"] += usage.request_tokens or 0
                total_usage["output_tokens"] += usage.response_tokens or 0
            except Exception as e:
                print(f"[Fail] {round_str}. Error calling PydanticAI Agent: {e}")
                # Use the previous response if available, otherwise break
                if assistant_response is None:
                    break
                print(f"Using previous response due to error.")

        elif args.agent_framework == "agno":
            try:
                # Invoke the Agno agent asynchronously
                # Agno doesn't manage history internally, so we pass the constructed input_conversation
                result: AgnoRunResponse = await agent.arun(messages=input_conversation)

                # Extract the assistant's response
                assistant_response = result.content

                # Get token usage for this round and accumulate
                usage = result.metrics
                total_usage["input_tokens"] += usage.get('prompt_tokens', [0])[-1] # Get the last value (current round)
                total_usage["output_tokens"] += usage.get('completion_tokens', [0])[-1] # Get the last value (current round)

            except Exception as e:
                print(f"[Fail] {round_str}. Error calling Agno Agent: {e}")
                # Use the previous response if available, otherwise break
                if assistant_response is None:
                    break
                print(f"Using previous response due to error.")

        elif args.agent_framework == "autogen":
            try:
                # Convert current history to AutoGen TextMessage list
                seed_msgs = []
                for m in input_conversation:
                    seed_msgs.append(TextMessage(source=m["role"], content=m["content"]))

                # Run AutoGen agent asynchronously
                # AutoGen's run might need adjustment based on how it handles multi-turn history internally
                # The current implementation passes the full history each time.
                result = await agent.run(task=seed_msgs)

                # Extract the assistant's response
                assistant_response = result.messages[-1].content

                # Get token usage
                usage = result.messages[-1].models_usage
                total_usage["input_tokens"] += usage.prompt_tokens
                total_usage["output_tokens"] += usage.completion_tokens

            except Exception as e:
                 print(f"[Fail] {round_str}. Error calling AutoGen Agent: {e}")
                 # Use the previous response if available, otherwise break
                 if assistant_response is None:
                     break # Break if error and no previous response
                 print(f"Using previous response due to error.")

        else:
             print(f"[Fail] {round_str}. Unsupported Agent Framework: {args.agent_framework}")
             break

        # Teriminate if no response
        if not assistant_response:
            print(f"{round_str}. No response from server.")
            break
        print(assistant_response)

        # Update conversation
        history_conversation.append({"role": "assistant", "content": assistant_response})

        #################################
        ## Solution-independent checks ##
        ################################# 

        # Extract action from response
        action = extract_action_from_response(assistant_response)
        # Terminate if no action found
        if not action:
            print(f"[Fail] {round_str}. No valid action found in response.")
            break

        # Convert to SudokuAction
        try:
            r_str, c_str, val_str = action
            sudoku_action = SudokuAction.from_tokens([
                "<vl>", f"<value{val_str}>", f"<r{r_str}>", f"<c{c_str}>"
            ])
        # Terminate if action parsing fails
        except Exception as e:
            print(f"[Fail] {round_str}. Error parsing action: {e}.")
            break

        # Update board state
        try:
            current_board.execute_action(sudoku_action)
        # Terminate if action execution fails
        except Exception as e:
            print(f"[Fail] {round_str}. Error executing action: {e}")
            break

        ###############################
        ## Solution-dependent checks ##
        ###############################

        # Check correctness
        action_row, action_col = sudoku_action.coordinates[0]
        ref = solution_board.get_cell(action_row, action_col).value.value
        hyp = sudoku_action.value.value 
        if hyp == ref:
            print(f"[Pass] {round_str}.")
            num_correct_placements += 1
        # Terminate if incorrect placement
        else:
            print(f"[Fail] {round_str}. Incorrect placement at {action_row}, {action_col}.")
            break

        # Teriminate if all cells are filled
        if '.' not in current_board.to_ascii(unfilled="."):
            print(f"[Pass] {round_str}. All cells filled.")
            break

    ##########################
    ## Final solution match ##
    ##########################

    # Check if solution is correct
    final_board_ascii = current_board.to_ascii(unfilled=".")
    final_solved = 1 if (final_board_ascii == solution_ascii) else 0

    # Get token counts
    total_input_tokens = 0
    total_output_tokens = 0
    if args.agent_framework == "smolagents" and agent is not None:
        token_counts = agent.monitor.get_total_token_counts()
        total_input_tokens = token_counts.get("input", 0)
        total_output_tokens = token_counts.get("output", 0)
    elif args.agent_framework == "openai_agents" and total_usage is not None:
        total_input_tokens = total_usage.input_tokens
        total_output_tokens = total_usage.output_tokens
    elif args.agent_framework in ["langgraph", "pydanticai", "agno", "autogen"] and total_usage is not None:
        total_input_tokens = total_usage["input_tokens"]
        total_output_tokens = total_usage["output_tokens"]

    # Determine the model name to save in results
    model_name_to_save = args.model_save_name if args.model_save_name else f"{args.agent_framework}-{model}"
    if args.agent_framework == "agno" and args.use_reasoning_tools:
        model_name_to_save = args.model_save_name if args.model_save_name else f"{args.agent_framework}-reasoning-{model}"

    return {
        # From input
        "data_source": args.dataset,
        "puzzle_id": request["puzzle_id"],
        "model": model_name_to_save,
        "num_empty_cells": request["num_empty_cells"],
        "shuffle_seed": request["shuffle_seed"],
        "n_response_idx": request["n_response_idx"],
        "n_history_turns": n_history_turns,
        "setting": setting,
        "initial_board": request["initial_board"],
        # From output
        "conversation": json.dumps(history_conversation),
        "num_rounds": round_idx + 1,
        "num_correct_placements": num_correct_placements,
        "final_solved": final_solved,
        "final_board": final_board_ascii,
        # Added token counts
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
    }


async def process_batch(
    args: argparse.Namespace,
    requests: List[Dict],
    model: str,
    batch_size: int = 1
) -> List[Dict]:
    semaphore = asyncio.Semaphore(batch_size)
    async def process_with_semaphore(request):
        async with semaphore:
            return await process_one(
                args=args,
                request=request,
                model=model,
            )
    
    tasks = [process_with_semaphore(request) for request in requests]
    outputs = []
    
    # Process requests with progress bar
    with tqdm(total=len(tasks), desc="Processing requests") as pbar:
        for coro in asyncio.as_completed(tasks):
            result = await coro
            outputs.append(result)
            pbar.update(1)
    
    return outputs


def construct_request(
    puzzle_id: str,
    author: str,
    rules: str,
    visual_elements: Optional[str],
    initial_board: str,
    solution: str,
    rows: int,
    cols: int,
    num_empty_cells: int,
    shuffle_seed: Optional[int],
    n_response_idx: int,
    n_history_turns: int,
) -> Optional[Dict]:
    # Fill hints if needed
    if num_empty_cells > 0:
        initial_board = random_fill_hints(
            initial_board,
            solution,
            num_empty_cells,
            shuffle_seed,
        )
        if initial_board is None:
            return None
    return {
        "puzzle_id": puzzle_id,
        "author": author,
        "rules": rules,
        "visual_elements": visual_elements,
        "initial_board": initial_board,
        "solution": solution,
        "rows": rows,
        "cols": cols,
        "num_empty_cells": num_empty_cells,
        "shuffle_seed": shuffle_seed,
        "n_response_idx": n_response_idx,
        "n_history_turns": n_history_turns,
    }
    

def main():
    parser = argparse.ArgumentParser(description="Evaluate LLM on Sudoku puzzles in a multi-round manner.")

    # Filepaths
    parser.add_argument("--dataset", type=str, required=True, choices=["challenge_100", "nikoli_100", "ctc"],
                        help="Dataset to evaluate on.")
    parser.add_argument("--output_csv", type=str, required=True,
                        help="Output CSV path.")

    # Subset of puzzles to evaluate
    parser.add_argument("--iloc_start", type=int, default=0,
                        help="Start index of puzzles to evaluate.")
    parser.add_argument("--iloc_end", type=int, default=None,
                        help="End index of puzzles to evaluate (exclusive).")
    parser.add_argument("--ilocs", type=int, nargs="+",
                        help="Specific puzzle indices to evaluate. Overrides start/end.")

    # Eval setting
    parser.add_argument("--puzzle_size", type=int, default=None,
                        help="Filter puzzles by size (e.g., 4 for 4x4). If None, use all sizes.")
    # The number of evaluations for each puzzle is the product of the following four arguments.
    parser.add_argument("--num_empty_cells", type=int, nargs="+", default=[0, 10, 20],
                        help="Number of empty cells in the intial board after hint fill in random cells. "
                             "0 means the original board.")
    parser.add_argument("--shuffle_seeds", type=int, nargs="+", default=[0],
                        help="Shuffle seeds for the random hint fill. Only used if num_empty_cells > 0.")
    parser.add_argument("--n_response_idxs", type=int, nargs="+", default=[0],
                        help="If you want to run multiple trials per puzzle/hint/seed. E.g., [0,1,2,3,4] for 5 runs.")
    parser.add_argument("--n_history_turns", type=int, nargs="+", default=[5],
                        help="Number of history turns to include in each LLM prompt. -1 means full history.")

    # Model
    parser.add_argument("--agent_framework", type=str, default="smolagents",
                        choices=["smolagents", "openai_agents", "langgraph", "pydanticai", "agno", "autogen"],
                        help="Agent Framework or direct API to use for evaluation.")
    parser.add_argument("--model", type=str, required=True,
                        help="Model name or path (e.g., 'openai/gpt-4o-mini' for OpenAI models with agent frameworks).")
    parser.add_argument("--model_save_name", type=str,
                        help="Model name in saved result. If not provided, use --model.")
    parser.add_argument("--max_tokens", type=int, default=32768,
                        help="Max tokens in each LLM response.")
    parser.add_argument("--temperature", type=float, default=0.1,
                        help="LLM temperature.")
    parser.add_argument("--top_p", type=float, default=0.95,
                        help="Top-p sampling probability.")
    parser.add_argument("--top_k", type=int, default=40,
                        help="Top-k sampling.")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size for parallel processing.")
    parser.add_argument("--max_retries", type=int, default=3,
                        help="Max retries for API calls.")
    parser.add_argument("--retry_delay", type=float, default=5.0,
                        help="Delay (in second) between retries.")

    # Agno specific
    parser.add_argument("--use_reasoning_tools", action="store_true",
                        help="Use ReasoningTools with agno framework (only applicable if agent_framework is 'agno').")

    # vLLM specific
    parser.add_argument("--tensor_parallel_size", type=int, default=1,
                        help="Tensor parallel size for vLLM.")
    parser.add_argument("--pipeline_parallel_size", type=int, default=1,
                        help="Pipeline parallel size for vLLM.")
    parser.add_argument("--draft_model", type=str,
                        help="Use the draft model.")
    
    args = parser.parse_args()

    # Add check for n_history_turns when using agent frameworks that manage history
    if args.agent_framework in ["smolagents", "openai_agents"]:
        if any(n != -1 for n in args.n_history_turns):
            raise ValueError(
                f"--n_history_turns must be [-1] when using agent_framework='{args.agent_framework}'. "
                f"{args.agent_framework} manages its own history."
            )

    # Warning if --use_reasoning_tools is used with other frameworks
    if args.agent_framework != "agno" and args.use_reasoning_tools:
        print("Warning: --use_reasoning_tools is only applicable when agent_framework is 'agno'. Ignoring.")

    # Sanity check
    assert args.num_empty_cells != [0] or len(args.shuffle_seeds) == 1, \
        "shuffle_seed is only used when providing hints (i.e. num_empty_cells > 0)."

    # Load puzzle
    dataset = datasets.load_dataset("SakanaAI/Sudoku-Bench", args.dataset, split="test")

    # Filter by puzzle size if specified
    if args.puzzle_size is not None:
        print(f"Filtering dataset for puzzle size: {args.puzzle_size}x{args.puzzle_size}")
        original_count = len(dataset)
        dataset = dataset.filter(lambda example: example.get('rows') == args.puzzle_size and example.get('cols') == args.puzzle_size)
        filtered_count = len(dataset)
        print(f"Filtered dataset from {original_count} to {filtered_count} puzzles.")
        if filtered_count == 0:
            print(f"Warning: No puzzles found for size {args.puzzle_size}x{args.puzzle_size} in dataset {args.dataset}. Exiting.")
            return

    # Use a subset of puzzles if specified
    if args.ilocs is not None:
        ilocs = args.ilocs
    else:
        end_idx = args.iloc_end if args.iloc_end is not None else len(dataset)
        ilocs = range(args.iloc_start, end_idx)
    puzzle_rows = [dataset[i] for i in ilocs]
    print(f"Number of puzzles to evaluate: {len(puzzle_rows)}")

    # Construct requests
    requests = []
    for puzzle_row in puzzle_rows:
        for nhist in args.n_history_turns:
            for ne in args.num_empty_cells:
                for sseed in args.shuffle_seeds:
                    for nr_idx in args.n_response_idxs:
                        request = construct_request(
                            puzzle_id=puzzle_row["puzzle_id"],
                            author=puzzle_row["author"],
                            rules=puzzle_row["rules"],
                            visual_elements=puzzle_row["visual_elements"],
                            initial_board=puzzle_row["initial_board"],
                            solution=puzzle_row["solution"],
                            rows=puzzle_row["rows"],
                            cols=puzzle_row["cols"],
                            num_empty_cells=ne,
                            shuffle_seed=sseed,
                            n_response_idx=nr_idx,
                            n_history_turns=nhist,
                        )
                        if request is not None:
                            requests.append(request)
    print(f"Number of requests to process: {len(requests)}")

    # Process batch
    all_results = asyncio.run(process_batch(
        args=args,
        batch_size=args.batch_size,
        requests=requests,
        model=args.model
    ))

    # Convert results to DataFrame
    res_df = pd.DataFrame(all_results)
    if len(res_df) == 0:
        print("No results to save. Possibly no puzzles or an error occurred.")
        return

    # Print summary
    # We'll measure average number of correct placements and fraction of puzzles solved.
    group_cols = ["num_empty_cells", "setting", "model"]
    summary = (
        res_df
        .groupby(group_cols)
        .agg({
            "num_correct_placements": "mean",
            "final_solved": "mean"
        })
        .reset_index()
    )
    with pd.option_context("display.max_rows", None, "display.precision", 2):
        print(summary)

    # Save results to CSV
    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    res_df.to_csv(args.output_csv, index=False)
    print(f"\nResults saved to {args.output_csv}")


if __name__ == "__main__":
    main()