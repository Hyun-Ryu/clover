import os
import json
import argparse
import numpy as np
from tqdm import tqdm
from utils import OpenAIModel


class LogicalDependencyParser:
    def __init__(self, args):
        self.args = args
        self.data_path = args.data_path
        self.dataset_name = args.dataset_name
        self.split = args.split
        self.model_name = args.model_name
        self.save_path = args.save_path
        self.exp_description = args.exp_description
        self.nl_types = args.nl_types
        self.openai_api = OpenAIModel(args.api_key, args.model_name, args.stop_words, args.max_new_tokens)
        self.load_prompt_templates()
    
    def load_prompt_templates(self):
        prompt_file = f'./models/prompts/{self.dataset_name}/parse.txt'
        with open(prompt_file, 'r') as f:
            self.prompt_template = f.read()

    def load_raw_dataset(self):
        with open(os.path.join(self.data_path, self.dataset_name, f'{self.split}-prep.json')) as f:
            raw_dataset = json.load(f)
        return raw_dataset
    
    def prompt_creator(self, declarations, sentence):
        full_prompt = self.prompt_template.replace('[[DECLARATIONS]]', declarations).replace('[[SENTENCE]]', sentence)      
        return full_prompt
    
    def nl2structure(self, nl_type, chunk):
        if nl_type not in self.nl_types:
            return [None] * len(chunk)

        # create prompts
        full_prompts = []
        full_info = []
        for example in chunk:
            declarations = example["declarations"]
            try:
                constraints = example[nl_type].split('\n')
            except:
                constraints = ["None"]
            
            for constraint in constraints:
                full_prompts.append(self.prompt_creator(declarations, constraint))
                full_info.append({"id": example["id"], "sentence": constraint})
        
        # api call
        batch_outputs = self.openai_api.batch_generate(full_prompts)

        # structurize outputs
        id_tmp = 'none'
        output_tmp = []
        constraints_accumulation_list = []
        for info, output in zip(full_info, batch_outputs):
            if info["id"] != id_tmp:
                if len(output_tmp) != 0:
                    constraints_accumulation_list.append(output_tmp)
                    output_tmp = []
                id_tmp = info["id"]
            output_tmp.append(output)
        constraints_accumulation_list.append(output_tmp)
        return constraints_accumulation_list

    def batch_logical_dependency_parsing(self, batch_size = 10):
        # load raw dataset
        raw_dataset = self.load_raw_dataset()
        print(f"Loaded {len(raw_dataset)} examples from {self.split} split.")

        outputs = []
        # split dataset into chunks
        dataset_chunks = [raw_dataset[i:i + batch_size] for i in range(0, len(raw_dataset), batch_size)]
        for chunk in tqdm(dataset_chunks):
            constraints_list = self.nl2structure(self.exp_description, chunk)

            assert len(chunk) == len(constraints_list)
            for sample, constraints_structure in zip(chunk, constraints_list):
                output = {
                    'id': sample['id'],
                    'context': sample['context'],
                    'question': sample['question'],
                    'options': sample['options'],
                    'answer': sample['answer'],
                    'declarations': sample['declarations'],
                    'constraints': sample['constraints'],
                    'queries': sample['queries'],
                    'additional_constraints': sample['additional_constraints'],
                    'queries_function': sample['queries_function'],
                    f'raw_{self.exp_description}_structure': constraints_structure,
                }
                outputs.append(output)
        
        # remove examples with duplicate ids from the result
        outputs = list({output['id']: output for output in outputs}.values())
        print(f"Generated {len(outputs)} examples.")

        # save outputs
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        
        with open(os.path.join(self.save_path, f'{self.dataset_name}_{self.split}_{self.model_name}_{self.exp_description}.json'), 'w') as f:
            json.dump(outputs, f, indent=2, ensure_ascii=False)
        
        self.raw_structures = outputs
    
    def output_reordering(self):
        raw_dataset = self.raw_structures

        outputs = []
        for data in raw_dataset:
            if data[self.exp_description] is None:
                final_dict = {
                    "id": data["id"],
                    "context": data["context"],
                    "question": data["question"],
                    "options": data["options"],
                    "answer": data["answer"],
                    "declarations": data["declarations"],
                    "constraints": data["constraints"],
                    "queries": data["queries"],
                    "additional_constraints": data["additional_constraints"],
                    'queries_function': data['queries_function'],
                    f"{self.exp_description}_structure": None,
                }
                outputs.append(final_dict)
                continue

            output_list = []
            constraints_structure = data[f"raw_{self.exp_description}_structure"]
            for constraint_structure in constraints_structure:
                constraint_structure = constraint_structure.replace("### Structures\n", "")
                constraint_structure += '\n\n'
                if "### Declarations" in constraint_structure:
                    tmp_list = [constraint_structure.split("### Declarations")[0][6:].strip()]
                else:
                    tmp_list = [tmp[2:].strip() for tmp in constraint_structure.split("### ")[1:]]
                tmp_list += [None]*(10 - len(tmp_list))
                output_list.append(tmp_list)

            output_array = np.array(output_list)
            constraints = data[self.exp_description].split('\n')
            final_list = []
            for i in range(10):
                tmp = output_array[:,i].tolist()
                lines_constraints = ''
                for sentence, structure in zip(constraints, tmp):
                    if structure is None:
                        continue

                    lines_constraints += '# Sentence\n'
                    lines_constraints += sentence
                    lines_constraints += '\n# Structure\n'
                    lines_constraints += structure
                    lines_constraints += '\n\n'

                lines_constraints = lines_constraints[:-2]
                if lines_constraints == '':
                    break
                else:
                    final_list.append(lines_constraints)
            
            # save the file
            final_dict = {
                "id": data["id"],
                "context": data["context"],
                "question": data["question"],
                "options": data["options"],
                "answer": data["answer"],
                "declarations": data["declarations"],
                "constraints": data["constraints"],
                "queries": data["queries"],
                "additional_constraints": data["additional_constraints"],
                'queries_function': data['queries_function'],
                f"{self.exp_description}_structure": final_list,
            }
            outputs.append(final_dict)

        # save the json file
        with open(os.path.join(self.save_path, f'{self.dataset_name}_{self.split}_{self.model_name}_{self.exp_description}_reordered.json'), 'w') as f:
            json.dump(outputs, f, indent=2, ensure_ascii=False)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='./data')
    parser.add_argument('--dataset_name', type=str)
    parser.add_argument('--split', type=str, default='dev')
    parser.add_argument('--save_path', type=str, default='./outputs/logical_dependency_structure')
    parser.add_argument('--api_key', type=str)
    parser.add_argument('--model_name', type=str, default='gpt-4o')
    parser.add_argument('--stop_words', type=str, default='------')
    parser.add_argument('--max_new_tokens', type=int, default=1024) # 2048 if needed
    parser.add_argument('--exp_description', type=str, default='constraints')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    logical_dependency_parser = LogicalDependencyParser(args)
    logical_dependency_parser.batch_logical_dependency_parsing()
    logical_dependency_parser.output_reordering()