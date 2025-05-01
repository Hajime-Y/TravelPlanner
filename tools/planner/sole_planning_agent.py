import os
import re
import sys
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "../..")))
os.chdir(os.path.dirname(os.path.abspath(__file__)))
from agents.prompts import planner_agent_prompt # direct strategy prompt only
# from utils.func import get_valid_name_city,extract_before_parenthesis, extract_numbers_from_filenames
import json
import time
# from langchain.callbacks import get_openai_callback # Remove langchain callback if not used with smolagents

from tqdm import tqdm
# from tools.planner.apis import Planner, ReactPlanner, ReactReflectPlanner # Removed old planner classes
# import openai
import argparse
from datasets import load_dataset
from typing import Any

# Add smolagents imports
try:
    from smolagents import CodeAgent, LiteLLMModel
except ImportError:
    print("smolagents not installed. Please run `pip install -r requirements.txt`")
    sys.exit(1)


def load_line_json_data(filename):
    data = []
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f.read().strip().split('\n'):
            unit = json.loads(line)
            data.append(unit)
    return data

def extract_numbers_from_filenames(directory):
    # Define the pattern to match files
    pattern = r'annotation_(\d+).json'

    # List all files in the directory
    files = os.listdir(directory)

    # Extract numbers from filenames that match the pattern
    numbers = [int(re.search(pattern, file).group(1)) for file in files if re.match(pattern, file)]

    return numbers

def smolagents_output_to_string(output: Any) -> str:
    """
    Convert smolagents `agent.run` output (AgentText, AgentImage, AgentAudio, etc.)
    into a single unified string representation.
    """
    # If the object provides a to_string() method, call it
    to_string_method = getattr(output, "to_string", None)
    if callable(to_string_method):
        return to_string_method()

    # If the output is a sequence, convert each element recursively and concatenate
    if isinstance(output, (list, tuple)):
        return "".join(smolagents_output_to_string(item) for item in output)

    # Otherwise, fall back to the built-in str()
    return str(output)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--set_type", type=str, default="validation")
    parser.add_argument("--model_name", type=str, default="openai/gpt-4.1-2025-04-14", help="Model name for the LLM (e.g., 'gpt-3.5-turbo-1106', 'claude-3-opus-20240229').")
    parser.add_argument("--output_dir", type=str, default="./")
    # parser.add_argument("--strategy", type=str, default="direct") # Strategy fixed to direct
    parser.add_argument("--agent_framework", type=str, default="smolagents", choices=["smolagents"], help="Agent framework to use.") # Add agent_framework argument
    args = parser.parse_args()
    directory = f'{args.output_dir}/{args.set_type}'
    if args.set_type == 'train':
        query_data_list  = load_dataset('osunlp/TravelPlanner','train')['train']
    elif args.set_type == 'validation':
        query_data_list  = load_dataset('osunlp/TravelPlanner','validation')['validation']
    elif args.set_type == 'test':
        query_data_list  = load_dataset('osunlp/TravelPlanner','test')['test']
    numbers = [i for i in range(1,len(query_data_list)+1)]

    # Strategy fixed to direct
    strategy = "direct"
    
    for number in tqdm(numbers[:]):

        # Initialize agent based on framework
        agent = None
        if args.agent_framework == "smolagents":
            llm_model = LiteLLMModel(model_id=args.model_name)
            # Use the direct strategy prompt (planner_agent_prompt) for system prompt/instructions
            agent = CodeAgent(
                tools=[], # No tools defined for sole planning
                model=llm_model,
            )
            print(f"Initialized smolagents CodeAgent with model: {args.model_name}")
        else:
            print(f"Error: Unsupported agent_framework '{args.agent_framework}'")
            sys.exit(1)
        
        query_data = query_data_list[number-1]
        reference_information = query_data['reference_information']
        while True:
            # Format the prompt string
            prompt = planner_agent_prompt.format(text=reference_information, query=query_data['query'])

            # Call the agent
            if args.agent_framework == "smolagents":
                result_obj = agent.run(prompt, reset=True)
                planner_results = smolagents_output_to_string(result_obj)
            else:
                print(f"Error: Unsupported agent_framework '{args.agent_framework}'")
                sys.exit(1)

            if planner_results != None:
                break

        print(planner_results)
        # check if the directory exists
        if not os.path.exists(os.path.join(f'{args.output_dir}/{args.set_type}')):
            os.makedirs(os.path.join(f'{args.output_dir}/{args.set_type}'))
        if not os.path.exists(os.path.join(f'{args.output_dir}/{args.set_type}/generated_plan_{number}.json')):
            result =  [{}]
        else:
            result = json.load(open(os.path.join(f'{args.output_dir}/{args.set_type}/generated_plan_{number}.json')))
        # if args.strategy in ['react','reflexion']:
        #     result[-1][f'{args.model_name}_{args.strategy}_sole-planning_results_logs'] = scratchpad 
        # Replace '/' with '-' in model name for the result key
        model_name_for_key = args.model_name.replace("/", "-")
        result[-1][f'{args.agent_framework}_{model_name_for_key}_{strategy}_sole-planning_results'] = planner_results
        # write to json file
        with open(os.path.join(f'{args.output_dir}/{args.set_type}/generated_plan_{number}.json'), 'w') as f:
            json.dump(result, f, indent=4, ensure_ascii=False)
