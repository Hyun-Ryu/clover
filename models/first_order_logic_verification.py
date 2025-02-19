import os
import json
import argparse
from datetime import datetime
from collections import Counter
from sat_solver.sat_code_verifier import Z3_Verifier
from utils import _logger


def main(args):
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    # logging
    log_file_name = os.path.join(args.save_path, f"logs_{datetime.now().strftime('%d_%m_%Y_%H_%M_%S')}_{args.dataset_name}_{args.split}_{args.model_name}_{args.verification_method}.log")
    logger = _logger(log_file_name)
    logger.debug("=" * 45)
    logger.debug(f'Dataset: {args.dataset_name}-{args.split}')
    logger.debug(f'Method:  {args.verification_method}')
    logger.debug(f'Model:   {args.model_name}')
    logger.debug("=" * 45)

    format_ = '_'.join(args.exp_descriptions)
    with open(os.path.join(args.data_path, f'{args.dataset_name}_{args.split}_{args.model_name}_{format_}_reordered.json')) as f:
        raw_dataset = json.load(f)

    output_dataset = []
    for data in raw_dataset:
        declarations = data["declarations"]
        if args.dataset_name == 'ZebraLogic':
            lines = []
            for decl in declarations.split('\n'):
                if 'Function' in decl:
                    func_name = decl.split('=')[0].strip()
                    lines += [f'Distinct([p:people], {func_name}(p))']
            declarations += '\n'
            declarations += '\n'.join(lines)

        z3_verifier = Z3_Verifier(args, declarations)

        for mode in args.exp_descriptions:
            if data[mode] is None:
                continue
            for index, item in enumerate(data[f"{mode}_translation"]):
                nl_sentence = item['nl']
                programs = item['programs']

                # Count occurrences of each program
                program_counts = Counter(programs).most_common()

                # Prepare the result for the current item
                program_candidates = {
                    "nl": nl_sentence,
                    "programs": [{"program": program, "count": count} for program, count in program_counts]
                }
                if args.verification_method == 'logic_cp_lm':
                    if len(program_candidates["programs"]) == 1:
                        program_selected = program_candidates["programs"][0]["program"]
                    else:
                        program_selected = z3_verifier.verify_candidates_lm(nl_sentence, program_candidates, logger)
                    item["selected_program"] = program_selected
                else:
                    program_selected, count_selected, program_candidates_groups = z3_verifier.verify_candidates_voting(program_candidates, logger)
                    if args.verification_method == 'logic_cs':
                        pass
                    elif args.verification_method == 'logic_cp':
                        if program_selected is None:
                            pass
                        else:
                            program_selected_cp, count_selected_cp = z3_verifier.verify_candidates_counterexample(nl_sentence, program_candidates_groups, logger)
                            if program_selected_cp is not None:
                                program_selected, count_selected = program_selected_cp, count_selected_cp
                    item["selected_program"] = program_selected
                    item["selected_program_count"] = count_selected
                    item["candidate_programs"] = program_candidates_groups
                data[f"{mode}_translation"][index] = item
        output_dataset.append(data)
    
    # save the result json file
    with open(os.path.join(args.save_path, f'{args.dataset_name}_{args.split}_{args.model_name}_{args.verification_method}.json'), 'w') as f:
        json.dump(output_dataset, f, indent=2, ensure_ascii=False)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='./outputs/first_order_logic_translation')
    parser.add_argument('--dataset_name', type=str)
    parser.add_argument('--split', type=str, default='dev')
    parser.add_argument('--save_path', type=str, default='./outputs/first_order_logic_verification')
    parser.add_argument('--api_key', type=str, default='none')
    parser.add_argument('--model_name', type=str, default='gpt-4o')
    parser.add_argument('--stop_words', type=str, default='------')
    parser.add_argument('--max_new_tokens', type=int, default=1024)
    parser.add_argument('--exp_descriptions', nargs='+')
    parser.add_argument('--verification_method', type=str, default='logic_cp') # logic_cs, logic_cp (our's), logic_cp_lm
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    main(args)