from tqdm import tqdm
import json
import argparse
from datasets import load_dataset


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--set_type", type=str, default="validation")
    parser.add_argument("--model_name", type=str, default="gpt-3.5-turbo-1106")
    parser.add_argument("--mode", type=str, default="two-stage")
    parser.add_argument("--strategy", type=str, default="direct")
    parser.add_argument("--output_dir", type=str, default="./")
    parser.add_argument("--submission_file_dir", type=str, default="./")
    parser.add_argument("--agent_framework", type=str, default=None, choices=["smolagents", "openai_agents", "langgraph", "pydanticai", "agno", "agno_reasoning", "autogen", "autogen_reasoning"], help="Agent framework used for generation.")

    args = parser.parse_args()

    model_name_for_path = args.model_name.replace("/", "-")
    if args.mode == 'two-stage':
        suffix = ''
    elif args.mode == 'sole-planning':
        if args.agent_framework:
            suffix = f'_{args.agent_framework}_{args.strategy}'
        else:
            suffix = f'_{args.strategy}'

    if args.set_type == 'train':
        query_data_list  = load_dataset('osunlp/TravelPlanner','train')['train']
    elif args.set_type == 'validation':
        query_data_list  = load_dataset('osunlp/TravelPlanner','validation')['validation']
    elif args.set_type == 'test':
        query_data_list  = load_dataset('osunlp/TravelPlanner','test')['test']

    idx_number_list = [i for i in range(1,len(query_data_list)+1)]

    submission_list = []

    for idx in tqdm(idx_number_list):
        generated_plan = json.load(open(f'{args.output_dir}/{args.set_type}/generated_plan_{idx}.json'))
        if args.mode == 'sole-planning' and args.agent_framework:
            model_name_for_key = args.model_name.replace("/", "-")
            parsed_result_key = f'{args.agent_framework}_{model_name_for_key}_{args.strategy}_sole-planning_parsed_results'
        else:
            parsed_result_key = f'{args.model_name}{suffix}_{args.mode}_parsed_results'
        plan = generated_plan[-1][parsed_result_key]
        submission_list.append({"idx":idx,"query":query_data_list[idx-1]['query'],"plan":plan})

    with open(f'{args.submission_file_dir}/{args.set_type}_{model_name_for_path}{suffix}_{args.mode}_submission.jsonl','w',encoding='utf-8') as w:
        for unit in submission_list:
            output = json.dumps(unit)
            w.write(output + "\n")
        w.close()
