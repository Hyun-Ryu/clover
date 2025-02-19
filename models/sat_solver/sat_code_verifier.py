import os
from os.path import join
import subprocess
from subprocess import check_output
from collections import OrderedDict
from utils import OpenAIModel
from .code_translator import *

class Z3_Verifier:
    def __init__(self, args, declarations:str) -> None:
        self.declarations = declarations
        try:
            self.parse_declaration_statements(declarations.split('\n'))
        except Exception as e:
            self.standard_code = None
            self.flag = False
            return
        
        self.flag = True
        self.dataset_name = args.dataset_name
        self.weekdays_position = False

        self.openai_api = OpenAIModel(args.api_key, args.model_name, args.stop_words, args.max_new_tokens)
        self.verification_method = args.verification_method

        # create the folder to save the Pyke program
        cache_dir = os.path.join(os.path.dirname(__file__), '.cache_program')
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
        self.cache_dir = cache_dir
    
    def parse_declaration_statements(self, declaration_statements):
        enum_sort_declarations = OrderedDict()
        int_sort_declarations = OrderedDict()
        function_declarations = OrderedDict()
        pure_declaration_statements = [x for x in declaration_statements if "Sort" in x or "Function" in x]
        variable_constraint_statements = [x for x in declaration_statements if not "Sort" in x and not "Function" in x]
        for s in pure_declaration_statements:
            # Weekdays/Position: EnumSort -> IntSort
            if "EnumSort" in s and (("Monday" in s or "monday" in s or "Tuesday" in s or "tuesday" in s)\
                                    or ("Top" in s or "top" in s or "Bottom" in s or "bottom" in s)):
                s = s.replace("EnumSort", "IntSort")
                # Assign integers to variables
                tmp = "And("
                sort_member_str = s.split("=")[1].strip()[len("IntSort("):-1]
                sort_members = [x.strip() for x in sort_member_str[1:-1].split(",")]
                for i, x in enumerate(sort_members):
                    tmp += (x + " == " + str(i+1) + ", ")
                variable_constraint_statements.append(tmp[:-2] + ")")
                # Specify the range of codomain
                self.weekdays_position = True

            if "EnumSort" in s:
                sort_name = s.split("=")[0].strip()
                sort_member_str = s.split("=")[1].strip()[len("EnumSort("):-1]
                sort_members = [x.strip() for x in sort_member_str[1:-1].split(",")]
                enum_sort_declarations[sort_name] = sort_members
            elif "IntSort" in s:
                sort_name = s.split("=")[0].strip()
                sort_member_str = s.split("=")[1].strip()[len("IntSort("):-1]
                sort_members = [x.strip() for x in sort_member_str[1:-1].split(",")]
                int_sort_declarations[sort_name] = sort_members
            elif "Function" in s:
                function_name = s.split("=")[0].strip()
                if "->" in s and "[" not in s:
                    function_args_str = s.split("=")[1].strip()[len("Function("):]
                    function_args_str = function_args_str.replace("->", ",").replace("(", "").replace(")", "")
                    function_args = [x.strip() for x in function_args_str.split(",")]
                    function_declarations[function_name] = function_args
                elif "->" in s and "[" in s:
                    function_args_str = s.split("=")[1].strip()[len("Function("):-1]
                    function_args_str = function_args_str.replace("->", ",").replace("[", "").replace("]", "")
                    function_args = [x.strip() for x in function_args_str.split(",")]
                    function_declarations[function_name] = function_args
                else:
                    # legacy way
                    function_args_str = s.split("=")[1].strip()[len("Function("):-1]
                    function_args = [x.strip() for x in function_args_str.split(",")]
                    function_declarations[function_name] = function_args
            else:
                raise RuntimeError("Unknown declaration statement: {}".format(s))

        declared_enum_sorts = OrderedDict()
        declared_lists = OrderedDict()
        self.declared_int_lists = OrderedDict()

        declared_functions = function_declarations
        already_declared = set()
        for name, members in enum_sort_declarations.items():
            # all contained by other enum sorts
            if all([x not in already_declared for x in members]):
                declared_enum_sorts[name] = members
                already_declared.update(members)
            declared_lists[name] = members

        for name, members in int_sort_declarations.items():
            self.declared_int_lists[name] = members
            # declared_lists[name] = members

        self.declared_enum_sorts = declared_enum_sorts
        self.declared_int_sorts = int_sort_declarations
        self.declared_lists = declared_lists
        self.declared_functions = declared_functions
        self.variable_constraints = variable_constraint_statements

        return True
    
    def to_standard_code(self, constraints_of_interest):
        declaration_lines = []
        # translate enum sorts
        for name, members in self.declared_enum_sorts.items():
            declaration_lines += CodeTranslator.translate_enum_sort_declaration(name, members)

        # translate int sorts
        for name, members in self.declared_int_sorts.items():
            declaration_lines += CodeTranslator.translate_int_sort_declaration(name, members)

        # translate lists
        for name, members in self.declared_lists.items():
            declaration_lines += CodeTranslator.translate_list_declaration(name, members)

        scoped_list_to_type = {}
        for name, members in self.declared_lists.items():
            if all(x.isdigit() for x in members):
                scoped_list_to_type[name] = CodeTranslator.ListValType.INT
            else:
                scoped_list_to_type[name] = CodeTranslator.ListValType.ENUM

        for name, members in self.declared_int_lists.items():
            scoped_list_to_type[name] = CodeTranslator.ListValType.INT
        
        # translate functions
        for name, args in self.declared_functions.items():
            declaration_lines += CodeTranslator.translate_function_declaration(name, args)
        
        variable_constraint_lines = []

        for constraint in self.variable_constraints:
            variable_constraint_lines += CodeTranslator.translate_constraint(constraint, scoped_list_to_type)
        
        # additional function scope control
        for name, args in self.declared_functions.items():
            if args[-1] in scoped_list_to_type and scoped_list_to_type[args[-1]] == CodeTranslator.ListValType.INT:
                # FIX
                if args[-1] in self.declared_int_lists:
                    continue
                
                list_range = [int(x) for x in self.declared_lists[args[-1]]]
                assert list_range[-1] - list_range[0] == len(list_range) - 1
                scoped_vars = [x[0] + str(i) for i, x in enumerate(args[:-1])]
                func_call = f"{name}({', '.join(scoped_vars)})"

                additional_cons = "ForAll([{}], And({} <= {}, {} <= {}))".format(
                    ", ".join([f"{a}:{b}" for a, b in zip(scoped_vars, args[:-1])]),
                    list_range[0], func_call, func_call, list_range[-1]
                )
                variable_constraint_lines += CodeTranslator.translate_constraint(additional_cons, scoped_list_to_type)
        
        # additional function scope control - weekdays/position
        for name, args in self.declared_functions.items():
            if args[-1] in scoped_list_to_type and scoped_list_to_type[args[-1]] == CodeTranslator.ListValType.INT:
                # FIX
                if not self.weekdays_position:
                    continue
                list_range = list(range(1, len(self.declared_int_lists[args[-1]])+1))
                assert list_range[-1] - list_range[0] == len(list_range) - 1
                scoped_vars = [x[0] + str(i) for i, x in enumerate(args[:-1])]
                func_call = f"{name}({', '.join(scoped_vars)})"

                additional_cons = "ForAll([{}], And({} <= {}, {} <= {}))".format(
                    ", ".join([f"{a}:{b}" for a, b in zip(scoped_vars, args[:-1])]),
                    list_range[0], func_call, func_call, list_range[-1]
                )
                variable_constraint_lines += CodeTranslator.translate_constraint(additional_cons, scoped_list_to_type)
        
        pure_constraint_lines = []

        assert len(constraints_of_interest) == 2
        for constraint in constraints_of_interest:
            pure_constraint_lines += CodeTranslator.translate_constraint(constraint, scoped_list_to_type)
        
        return CodeTranslator.assemble_verifier_code(declaration_lines, variable_constraint_lines, pure_constraint_lines, self.declared_functions.items())
    
    def is_equivalent(self, condition_p, condition_q):
        constraints_of_interest = [condition_p, condition_q]
        try:
            standard_code = self.to_standard_code(constraints_of_interest)
        except:
            return False
        output, error_message = self.execute_program(standard_code)
        if output is None:
            return False
        bar = output.index("------")
        p_q = output[:bar]
        q_p = output[bar+1:]
        if p_q[1] == 'True' and q_p[1] == 'True':   # p<->q
            return True
        return False
    
    def verify_candidates_voting(self, program_candidates, logger):
        logger.debug("\nNL: {}".format(program_candidates["nl"]))
        program_candidates_wo_error = {
            "nl": program_candidates["nl"],
            "programs": []
        }
        # exclude execution error codes
        for i, programs in enumerate(program_candidates["programs"]):
            condition_p = programs["program"]
            constraints_of_interest = [condition_p, condition_p]
            try:
                standard_code = self.to_standard_code(constraints_of_interest)
                output, error_message = self.execute_program(standard_code)
                if output is None:
                    continue
                program_candidates_wo_error["programs"].append({
                    "program": programs["program"],
                    "count": programs["count"]
                })
            except:
                continue
        
        if len(program_candidates_wo_error["programs"]) == 0:
            logger.debug("No executable programs generated")
            return None, None, None
        
        # Integrate equivalent programs
        programs = program_candidates_wo_error["programs"]
        n = len(programs)
        groups = {}
        parent = list(range(n))

        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]

        def union(x, y):
            rootX = find(x)
            rootY = find(y)
            if rootX != rootY:
                parent[rootY] = rootX

        # Check equivalence and union the indices
        for i in range(n):
            for j in range(i+1, n):
                condition_p = programs[i]["program"]
                condition_q = programs[j]["program"]
                constraints_of_interest = [condition_p, condition_q]
                try:
                    standard_code = self.to_standard_code(constraints_of_interest)
                except:
                    continue
                output, error_message = self.execute_program(standard_code)
                if output is None:
                    continue
                bar = output.index("------")
                p_q = output[:bar]
                q_p = output[bar+1:]
                if p_q[1] == 'True' and q_p[1] == 'True':   # p<->q
                    union(i, j)

        # Collect groups
        for i in range(n):
            root = find(i)
            if root in groups:
                groups[root].append(i)
            else:
                groups[root] = [i]
        
        # Structurize data
        program_candidates_groups = []
        cnt = 0
        for group, indices in groups.items():
            data_ = {"program": [], "count": []}
            for ind in indices:
                data_["program"].append(programs[ind]["program"])
                data_["count"].append(programs[ind]["count"])
            if sum(data_["count"]) > cnt:
                cnt = sum(data_["count"])
                candidate = data_["program"][0]
            program_candidates_groups.append(data_)

        return candidate, cnt, program_candidates_groups
    
    def verify_candidates_counterexample(self, nl_sentence, program_candidates_groups, logger):
        flag_incorrect = False
        for i, program_candidates_group in enumerate(program_candidates_groups):
            # initialize p
            if i == 0:
                candidate, candidate_count = program_candidates_group["program"][0], sum(program_candidates_group["count"])
                if len(program_candidates_groups) == 1:
                    logger.debug(f"only condition: {candidate}, count: {candidate_count}")
                continue
            if flag_incorrect and  i != len(program_candidates_groups)-1:
                candidate, candidate_count = program_candidates_group["program"][0], sum(program_candidates_group["count"])
                flag_incorrect = False
                continue

            # assign p and q
            condition_p = candidate
            condition_q = program_candidates_group["program"][0]
            constraints_of_interest = [condition_p, condition_q]
            try:
                standard_code = self.to_standard_code(constraints_of_interest)
            except:
                continue
            output, error_message = self.execute_program(standard_code)
            if output is None:
                continue
            logger.debug(f"condition p: {condition_p}")
            logger.debug(f"condition q: {condition_q}")
            logger.debug(f"{output}, {error_message}")
            
            # identify inclusion-exclusion relationship of those two logics
            # p->q, q->p, p<->q, p^q exsits, p^q = None
            bar = output.index("------")
            p_q = output[:bar]
            q_p = output[bar+1:]
            
            # update the candidate and its occurence - logically more reliable one
            if p_q[1] == 'False':  # if there is a solution satisfying p^~q
                solution = self.postprocess_solution(p_q)
                answer_p_q, full_prompt, raw_answer = self.verification_generation(nl_sentence, solution, logger)
                logger.debug(raw_answer)
                if answer_p_q == 'no':
                    candidate, candidate_count = program_candidates_group["program"][0], sum(program_candidates_group["count"])
            if q_p[1] == 'False':  # if there is a solution satisfying q^~p
                solution = self.postprocess_solution(q_p)
                answer_q_p, full_prompt, raw_answer = self.verification_generation(nl_sentence, solution, logger)
                logger.debug(raw_answer)
                if answer_q_p == 'yes':
                    candidate, candidate_count = program_candidates_group["program"][0], sum(program_candidates_group["count"])
            
            if p_q[1] == 'False' and q_p[1] == 'False':
                if answer_p_q == answer_q_p:
                    flag_incorrect = True
                    logger.debug('Both p and q are incorrect logic.')
        
        if flag_incorrect:
            return None, None

        return candidate, candidate_count

    def postprocess_solution(self, raw_solution):
        raw_solution = raw_solution[2:]
        if self.dataset_name == 'ZebraLogic':
            lines = []
            lines += ['people -> ' + ', '.join([name for name, args in self.declared_functions.items()])]

            for name, members in self.declared_enum_sorts.items():
                if name == 'people':
                    people_list = members
                    break
            for person in people_list:
                tmp_list = []
                for func_assgn in raw_solution:
                    if person in func_assgn:
                        tmp_list.append(func_assgn.split('->')[-1].strip())
                lines += [f'{person} -> ' + ', '.join(tmp_list)]
            solution = lines
        elif self.dataset_name == 'AR-LSAT':
            solution = raw_solution
        else:
            raise ValueError('Wrong dataset name')

        return '\n'.join(solution)


    def verify_candidates_lm(self, nl_sentence, program_candidates, logger):
        programs_list = []
        for i, programs in enumerate(program_candidates["programs"]):
            programs_list.append(programs["program"])
        formulas = '\n'.join(programs_list)

        # LLM api
        prompt_file = './models/prompts/baseline/verification-llm_instruction.txt'
        with open(prompt_file, 'r') as f:
            prompt_template = f.read()
        full_prompt = prompt_template.replace('[[SENTENCE]]', nl_sentence).replace('[[DECLARATIONS]]', self.declarations).replace('[[FORMULAS]]', formulas)
        raw_answer = self.openai_api.generate(full_prompt)

        logger.debug('Prompt:')
        logger.debug(full_prompt.split('------')[1].strip())
        logger.debug(raw_answer)
        logger.debug('\n\n')

        # post-process the raw answer
        return raw_answer.split('Formula:')[-1].strip()
    
    def execute_program(self, standard_code):
        filename = join(self.cache_dir, f'tmp.py')
        with open(filename, "w") as f:
            f.write(standard_code)
        try:
            output = check_output(["python", filename], stderr=subprocess.STDOUT, timeout=1.0)
        except subprocess.CalledProcessError as e:
            outputs = e.output.decode("utf-8").strip().splitlines()[-1]
            return None, outputs
        except subprocess.TimeoutExpired:
            return None, 'TimeoutError'
        output = output.decode("utf-8").strip()
        result = output.splitlines()
        if len(result) == 0:
            return None, 'No Output'
        
        return result, ""

    def verification_prompt(self, nl_sentence, solution):
        prompt_file = f'./models/prompts/{self.dataset_name}/verify.txt'
        with open(prompt_file, 'r') as f:
            prompt_template = f.read()
        
        full_prompt = prompt_template.replace('[[SENTENCE]]', nl_sentence).replace('[[DECLARATIONS]]', self.declarations).replace('[[SOLUTION]]', solution)
        return full_prompt
    
    def verification_generation(self, nl_sentence, solution, logger):
        ### call api and get the verification results ###
        try:
            full_prompt = self.verification_prompt(nl_sentence, solution)
            raw_answer = self.openai_api.generate(full_prompt)
            # post-process the raw answer
            raw_answer_ = raw_answer.split('Answer:')[-1].strip()

            if "yes" in raw_answer_.lower():
                answer = "yes"
            elif "no" in raw_answer_.lower():
                answer = "no"
            else:
                answer = "unknown"
        except:
            logger.debug(f'Error in generating answer(y/n) for the sentence: {nl_sentence}')
        
        return answer, full_prompt, raw_answer