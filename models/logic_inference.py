import os
import json
import argparse
from tqdm import tqdm
from sat_solver.sat_problem_solver import Z3_Program
from backup_answer_generation import Backup_Answer_Generator


class LogicInferenceEngine:
    def __init__(self, args):
        self.args = args
        self.data_path = args.data_path
        self.dataset_name = args.dataset_name
        self.split = args.split
        self.model_name = args.model_name
        self.save_path = args.save_path
        self.backup_strategy = args.backup_strategy
        self.verification_method = args.verification_method
        self.dataset = self.load_logic_programs()
        self.program_executor = Z3_Program
        self.backup_generator = Backup_Answer_Generator(self.dataset_name, self.backup_strategy, self.args.backup_LLM_result_path)
    
    def load_logic_programs(self):
        with open(os.path.join(self.data_path, f'{self.dataset_name}_{self.split}_{self.model_name}_{self.verification_method}.json')) as f:
            dataset = json.load(f)
        print(f"Loaded {len(dataset)} examples from {self.split} split.")
        return dataset

    def save_results(self, outputs):
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        
        with open(os.path.join(self.save_path, f'{self.dataset_name}_{self.split}_{self.model_name}_{self.verification_method}_backup-{self.backup_strategy}.json'), 'w') as f:
            json.dump(outputs, f, indent=2, ensure_ascii=False)
    
    def safe_execute_program(self, id, logic_program):
        program = self.program_executor(logic_program, self.dataset_name)

        # cannot parse the program
        if program.flag == False:
            answer = self.backup_generator.get_backup_answer(id)
            return answer, 'parsing error', ''
        # execuate the program
        answer, error_message = program.execute_program()
        # not executable
        if answer is None:
            answer = self.backup_generator.get_backup_answer(id)
            return answer, 'execution error', error_message
        # successfully executed
        answer = program.answer_mapping(answer)
        return answer, 'success', ''
    
    def inference_on_dataset_arlsat(self):
        outputs = []
        error_count = 0
        
        for example in tqdm(self.dataset):
            # preprocess query function
            if '\n' in example['queries_function']:
                raw_query_function = example['queries_function'].split('\n')[1]
            else:
                raw_query_function = example['queries_function']
            query_function_list = []
            for func_ in raw_query_function.split(','):
                query_function_list.append(func_.strip()[:-2])
            query_function_list.reverse()

            # re-organize the logic program
            logic_program = '# Declarations\n' + example['declarations'] + '\n\n# Constraints\n'

            for constraint in example['constraints_translation']:
                code = constraint["selected_program"]
                nl = constraint["nl"]
                logic_program += f'{code} ::: {nl}\n'
            
            if example['additional_constraints_translation'] is not None:
                for additional_constraint in example['additional_constraints_translation']:
                    code = additional_constraint["selected_program"]
                    nl = additional_constraint["nl"]
                    logic_program += f'{code} ::: {nl}\n'
            
            logic_program += '\n# Options\nQuestion ::: \n'
            for query in example['queries_translation']:
                code = query["selected_program"]
                if query["selected_program"] is None:
                    code = "None"
                for query_function in query_function_list:
                    code = f'{query_function}({code})'
                logic_program += f'{code} ::: ()\n'
            logic_program = logic_program[:-1]
            
            # execute the logic program
            answer, flag, error_message = self.safe_execute_program(example['id'], logic_program)
            if not flag == 'success':
                error_count += 1

            # create output
            output = {
                'id': example['id'], 
                'declarations': example['declarations'],
                'constraints': example['constraints'],
                'queries': example['queries'],
                'additional_constraints': example['additional_constraints'],
                'answer': example['answer'],
                'flag': flag,
                'predicted_answer': answer,
                'raw_logic_programs': logic_program,
            }
            outputs.append(output)
        
        print(f"Error count: {error_count}")
        self.save_results(outputs)
        self.cleanup()
    
    def inference_on_dataset_zebralogic(self):
        outputs = []
        error_count = 0
        
        for example in tqdm(self.dataset):
            # re-organize the logic program
            logic_program = '# Declarations\n' + example['declarations'] + '\n\n# Constraints\n'

            # add Distinct() constraint -- excluded in the constraint extraction stage
            for line in example['declarations'].split('\n'):
                if 'Function' in line:
                    function_name = line.split('=')[0].strip()
                    logic_program += f'Distinct([p:people], {function_name}(p))\n'

            for constraint in example['constraints_translation']:
                code = constraint["selected_program"]
                if code is None:
                    code = "None"
                else:
                    code = code.split('\n')[0].replace('`', '').strip()
                nl = constraint["nl"]
                logic_program += f'{code} ::: {nl}\n'
            logic_program += '\n# Options\nQuestion ::: \n'

            query = example['queries']
            logic_program += f'is_valid({query}) ::: ()'
            
            # execute the logic program
            answer, flag, error_message = self.safe_execute_program(example['id'], logic_program)
            if not flag == 'success':
                error_count += 1

            # create output
            output = {
                'id': example['id'], 
                'declarations': example['declarations'],
                'constraints': example['constraints'],
                'queries': example['queries'],
                'additional_constraints': example['additional_constraints'],
                'answer': example['answer'],
                'flag': flag,
                'predicted_answer': answer,
                'raw_logic_programs': logic_program,
            }
            outputs.append(output)
        
        print(f"Error count: {error_count}")
        self.save_results(outputs)
        self.raw_outputs = outputs
        self.cleanup()

    def cleanup(self):
        complied_krb_dir = './models/compiled_krb'
        if os.path.exists(complied_krb_dir):
            print('removing compiled_krb')
            os.system(f'rm -rf {complied_krb_dir}')
    
    def eval(self):
        raw_dataset = self.raw_outputs

        if self.dataset_name == 'ZebraLogic':
            count, tot = 0, 0
            hardest_sizes =  ['4x6', '5x5', '5x6', '6x4', '6x5', '6x6']
            puzzle_dict = dict()
            for size in hardest_sizes:
                puzzle_dict[size] = 0
            
            for data in raw_dataset:
                tot += 1
                if data["flag"] == "success" and "A" == data["predicted_answer"]:
                    count += 1
                    puzzle_size = data['id'].split('-')[2]
                    puzzle_dict[puzzle_size] += 1
            
            print("accuracy: %d/%d, %.2f%%" % (count, tot, count/tot*100))
            print(puzzle_dict)

        else:
            for metric in ['exe_rate', 'exe_acc', 'prog_acc', 'backup_acc']:
                count, tot = 0, 0
                for data in raw_dataset:
                    if metric == 'exe_acc':
                        tmp1 = data["flag"] == "success"
                    else:
                        tmp1 = True
                    if tmp1:
                        tot += 1
                    
                    if metric == 'exe_rate':
                        tmp2 = data["flag"] == "success"
                    elif metric == 'backup_acc':
                        tmp2 = data["answer"] == data["predicted_answer"]
                    else:
                        tmp2 = data["flag"] == "success" and data["answer"] == data["predicted_answer"]
                    if tmp2:
                        count += 1
                print("%s accuracy: %d/%d, %.2f%%" % (metric, count, tot, count/tot*100))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='./outputs/first_order_logic_verification')
    parser.add_argument('--dataset_name', type=str)
    parser.add_argument('--split', type=str, default='dev')
    parser.add_argument('--save_path', type=str, default='./outputs/logic_inference')
    parser.add_argument('--backup_strategy', type=str, default='random', choices=['random', 'LLM'])
    parser.add_argument('--backup_LLM_result_path', type=str, default='../baselines/results')
    parser.add_argument('--model_name', type=str, default='gpt-4o')
    parser.add_argument('--timeout', type=int, default=60)
    parser.add_argument('--verification_method', type=str, default='logic_cs') # logic_cs, logic_cp, logic_cp_lm
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    engine = LogicInferenceEngine(args)
    if args.dataset_name == 'AR-LSAT':
        engine.inference_on_dataset_arlsat()
    elif args.dataset_name == 'ZebraLogic':
        engine.inference_on_dataset_zebralogic()
    else:
        raise ValueError
    engine.eval()