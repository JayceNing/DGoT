# author: Jayce Ning

import os
import re
import logging
import datetime
import time
import json
import csv
from statistics import fmean
from typing import Dict, List, Callable, Set, Union
from graph_of_thoughts import controller, operations, prompter, parser
from utils import read_pmc, read_pm, rouge1_f_test_introduction, rouge1_f_gold_summary
from tqdm import tqdm
import argparse
import tiktoken

generate_prompt_nums = {}
cut_abstract_nums = {}


class GenAbstractPrompter(prompter.Prompter):
    """
    GenAbstractPrompter provides the generation of prompts specific to the article
    information example for the language models.

    Inherits from the Prompter class and implements its abstract methods.
    """

    generate_abstract_prompt_start = """Please generate the abstrace of the article based on the following information <origin> of the object article.
    If there are references for the article, the title and abstract of the reference will also be listed in <reference>. 
    Minimizing redundancy and retaining valid information as much as possible. 
    Only the summaries generated between tags <Abstract> and </Abstract> are output, and no other text is output.
"""
    generate_abstract_prompt_block = """
<origin>
{origin}
</origin>
{reference}
"""

    generate_abstract_prompt_cot_start = """Please generate the abstrace of the article based on the following information <origin> of the object article.
You can generate any intermediate thoughts and texts you want, but the final output should be an abstract, placed between the two tags <abstract> and </abstract>.
For instance you might want to follow this approach:
1. Mainly generate abstracts based on the <introduction> section of the original literature <origin>.
2. Referring to other chapters of the original literature, if deemed useful for abstract generation, integrate them into the abstract section.
3. If there are references <reference>, summarize the content of the references in one sentence.
4. If the summarized reference content is related to the generated original abstract, integrate it into the abstract section.

Here is the relevant information of the source article:
"""

    improve_abstract_prompt_start = """The following Abstract <A> is summarized in the relevant information of the original literature <origin> and its references <reference>.
Please improve the Abstract <A> by adding more information and removing redundancy. Output only the improved Abstract, placed between the two tags <Abstract> and </Abstract>, without any additional text.

Here is the generated abstract <A>:
"""

    improve_abstract_prompt_block = """
<A>
{abstract}
</A>
"""

    improve_abstract_prompt_end = """
Here is the origin article information <origin>:
<origin>
{origin}
</origin>
{reference}
"""

    aggregate_full_prompt_base = """The following Abstract <A1> - <A{num_abstract}> are summarized in the relevant information of the original literature <origin> and its references <reference>.
Combine the generated abstract <A1> - <A{num_abstract}> into a new one, maximizing their advantages and overall information retention, while minimizing redundancy.
Output only the new abstract between the tags <Abstract> and </Abstract>, without any additional text.   

Here are the generated abstract <A1> - <A{num_abstract}>:
"""

    aggregate_full_prompt_block = """
<A{num}>
{abstract}
</A{num}>
"""

    aggregate_full_prompt_end = """
Here is the origin article information <origin>:
<origin>
{origin}
</origin>
{reference}
"""

    def __init__(self, max_input_prompt_tokens):
        super().__init__()

        self.max_input_prompt_tokens = max_input_prompt_tokens

    def aggregation_prompt(self, state_dicts: List[Dict], **kwargs) -> str:
        """
        Generate an aggregation prompt for the language model.

        :param state_dicts: The thought states that should be aggregated.
        :type state_dicts: List[Dict]
        :param kwargs: Additional keyword arguments.
        :return: The aggregation prompt.
        :rtype: str
        """

        global generate_prompt_nums
        global cut_abstract_nums
        generate_prompt_nums[self.max_input_prompt_tokens] += 1

        encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")

        prompt = self.aggregate_full_prompt_base.format(
            num_abstract=len(state_dicts),
        )
        for i, state_dict in enumerate(state_dicts):
            prompt += self.aggregate_full_prompt_block.format(
                abstract=state_dict["current"], num=i + 1
            )

        base_len = len(encoding.encode(prompt))

        origin = ''
        reference = ''
        origin += '<title>' + state_dicts[0]["origin_title"] + '</title>'
        origin += '<introduction>' + state_dicts[0]["origin_introduction"] + '</introduction>'
        for key in state_dicts[0]["origin_info"].keys():
            origin += '<' + key + '>' + state_dicts[0]["origin_info"][key] + '</' + key + '>'

        origin_len = len(encoding.encode(origin))
        if base_len + origin_len > self.max_input_prompt_tokens:
            cut_abstract_num[self.max_input_prompt_tokens] += 1
            prompt += self.aggregate_full_prompt_end.format(
                origin=origin, reference=reference
            )
            prompt_token_count = len(encoding.encode(prompt))
            idx = len(prompt)
            while prompt_token_count > self.max_input_prompt_tokens:
                prompt = prompt[:idx]
                prompt_token_count = len(encoding.encode(prompt))
                # print(prompt_token_count)
                idx = idx - 100
            return prompt
        
        if len(state_dicts[0]["reference_info"].keys())>0:
            reference += '<reference>'
            for key in state_dicts[0]["reference_info"].keys():
                reference += '<title>' + key + '</title>'
                reference += '<abstract>' + state_dicts[0]["reference_info"][key] + '</abstract>'
            reference += '</reference>'
        prompt += self.aggregate_full_prompt_end.format(
            origin=origin, reference=reference
        )

        return prompt

    def generate_prompt(
        self,
        num_branches: int,
        origin_title: str,
        origin_introduction: str,
        origin_info: dict,
        reference_info: dict,
        method: str,
        #parts: Set[str],
        current: str,
        **kwargs,
    ) -> str:
        """
        Generate a generate prompt for the language model.

        :param num_branches: The number of responses the prompt should ask the LM to generate.
        :type num_branches: int
        :param documents: The list of documents to be merged.
        :type documents: List[str]
        :param method: Method for which the generate prompt is generated.
        :type method: str
        :param parts: Indices of the already processed document parts.
        :type parts: Set[str]
        :param current: The intermediate solution.
        :type current: str
        :param kwargs: Additional keyword arguments.
        :return: The generate prompt.
        :rtype: str
        :raise AssertionError: If method is not implemented yet.
        """

        global generate_prompt_nums
        global cut_abstract_nums
        generate_prompt_nums[self.max_input_prompt_tokens] += 1

        encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")

        prompt = ""
        if method.startswith("io") or method.startswith("cot"):
            if method.startswith("io"):
                prompt += self.generate_abstract_prompt_start
            else:
                prompt += self.generate_abstract_prompt_cot_start
            
            base_len = len(encoding.encode(prompt))

            origin = ''
            reference = ''
            origin += '<title>' + origin_title + '</title>'
            origin += '<introduction>' + origin_introduction + '</introduction>'
            for key in origin_info.keys():
                origin += '<' + key + '>' + origin_info[key] + '</' + key + '>'

            origin_len = len(encoding.encode(origin))
            if base_len + origin_len > self.max_input_prompt_tokens:
                cut_abstract_nums[self.max_input_prompt_tokens] += 1
                prompt += self.aggregate_full_prompt_end.format(
                    origin=origin, reference=reference
                )
                prompt_token_count = len(encoding.encode(prompt))
                idx = len(prompt)
                while prompt_token_count > self.max_input_prompt_tokens:
                    prompt = prompt[:idx]
                    prompt_token_count = len(encoding.encode(prompt))
                    # print(prompt_token_count)
                    idx = idx - 100
                return prompt

            if len(reference_info.keys())>0:
                reference += '<reference>'
                for key in reference_info.keys():
                    reference += '<title>' + key + '</title>'
                    reference += '<abstract>' + reference_info[key] + '</abstract>'
                reference += '</reference>'
            prompt += self.generate_abstract_prompt_block.format(
                origin=origin, reference=reference
            )

            return prompt

        elif method.startswith("tot") or method.startswith("got") or method.startswith("dgot") \
            or method.startswith("dgot_aggregate"):
            if current is None or current == "":
                prompt += self.generate_abstract_prompt_start

                base_len = len(encoding.encode(prompt))

                origin = ''
                reference = ''
                origin += '<title>' + origin_title + '</title>'
                origin += '<introduction>' + origin_introduction + '</introduction>'
                for key in origin_info.keys():
                    origin += '<' + key + '>' + origin_info[key] + '</' + key + '>'

                origin_len = len(encoding.encode(origin))
                if base_len + origin_len > self.max_input_prompt_tokens:
                    cut_abstract_nums[self.max_input_prompt_tokens] += 1
                    prompt += self.aggregate_full_prompt_end.format(
                        origin=origin, reference=reference
                    )
                    prompt_token_count = len(encoding.encode(prompt))
                    idx = len(prompt)
                    while prompt_token_count > self.max_input_prompt_tokens:
                        prompt = prompt[:idx]
                        prompt_token_count = len(encoding.encode(prompt))
                        # print(prompt_token_count)
                        idx = idx - 100
                    return prompt

                if len(reference_info.keys())>0:
                    reference += '<reference>'
                    for key in reference_info.keys():
                        reference += '<title>' + key + '</title>'
                        reference += '<abstract>' + reference_info[key] + '</abstract>'
                    reference += '</reference>'
                prompt += self.generate_abstract_prompt_block.format(
                    origin=origin, reference=reference
                )

                return prompt
            else:
                prompt += self.improve_abstract_prompt_start
                base_len = len(encoding.encode(prompt))

                origin = ''
                reference = ''
                origin += '<title>' + origin_title + '</title>'
                origin += '<introduction>' + origin_introduction + '</introduction>'
                for key in origin_info.keys():
                    origin += '<' + key + '>' + origin_info[key] + '</' + key + '>'

                origin_len = len(encoding.encode(origin))
                if base_len + origin_len > self.max_input_prompt_tokens:
                    cut_abstract_nums[self.max_input_prompt_tokens] += 1
                    prompt += self.aggregate_full_prompt_end.format(
                        origin=origin, reference=reference
                    )
                    prompt_token_count = len(encoding.encode(prompt))
                    idx = len(prompt)
                    while prompt_token_count > self.max_input_prompt_tokens:
                        prompt = prompt[:idx]
                        prompt_token_count = len(encoding.encode(prompt))
                        # print(prompt_token_count)
                        idx = idx - 100
                    return prompt

                if len(reference_info.keys())>0:
                    reference += '<reference>'
                    for key in reference_info.keys():
                        reference += '<title>' + key + '</title>'
                        reference += '<abstract>' + reference_info[key] + '</abstract>'
                    reference += '</reference>'
                prompt += self.improve_abstract_prompt_block.format(
                    abstract = current
                )
                prompt += self.improve_abstract_prompt_end.format(
                    origin=origin, reference=reference
                )

                return prompt
        else:
            assert False, "Not implemented yet."

    def score_prompt(self, state_dicts: List[Dict], **kwargs) -> str:
        """
        Generate a score prompt for the language model.

        :param state_dicts: The thought states that should be scored,
                            if more than one, they should be scored together.
        :type state_dicts: List[Dict]
        :param kwargs: Additional keyword arguments.
        :return: The score prompt.
        :rtype: str
        :raise AssertionError: If more than one thought state is supplied.
        """
        pass

    def improve_prompt(self, **kwargs) -> str:
        """
        Generate an improve prompt for the language model.

        :param kwargs: Additional keyword arguments.
        :return: The improve prompt.
        :rtype: str
        """
        pass

    def validation_prompt(self, **kwargs) -> str:
        """
        Generate a validation prompt for the language model.

        :param kwargs: Additional keyword arguments.
        :return: The validation prompt.
        :rtype: str
        """
        pass


class GenAbstractParser(parser.Parser):
    """
    GenAbstractParser provides the parsing of language model reponses specific to the
    article information example.

    Inherits from the Parser class and implements its abstract methods.
    """

    def __init__(self) -> None:
        """
        Inits the response cache.
        """
        self.cache = {}

    def strip_answer_helper(self, text: str, tag: str = "") -> str:
        """
        Helper function to remove tags from a text.

        :param text: The input text.
        :type text: str
        :param tag: The tag to be stripped. Defaults to "".
        :type tag: str
        :return: The stripped text.
        :rtype: str
        """

        text = text.strip()
        if "Output:" in text:
            text = text[text.index("Output:") + len("Output:") :].strip()
        if tag != "":
            start = text.rfind(f"<{tag}>")
            end = text.rfind(f"</{tag}>")
            if start != -1 and end != -1:
                text = text[start + len(f"<{tag}>") : end].strip()
            elif start != -1:
                logging.warning(
                    f"Only found the start tag <{tag}> in answer: {text}. Returning everything after the tag."
                )
                text = text[start + len(f"<{tag}>") :].strip()
            elif end != -1:
                logging.warning(
                    f"Only found the end tag </{tag}> in answer: {text}. Returning everything before the tag."
                )
                text = text[:end].strip()
            else:
                logging.warning(
                    f"Could not find any tag {tag} in answer: {text}. Returning the full answer."
                )
        return text

    def parse_aggregation_answer(
        self, states: List[Dict], texts: List[str]
    ) -> Union[Dict, List[Dict]]:
        """
        Parse the response from the language model for an aggregation prompt.

        :param states: The thought states used to generate the prompt.
        :type states: List[Dict]
        :param texts: The responses to the prompt from the language model.
        :type texts: List[str]
        :return: The new thought states after parsing the respones from the language model.
        :rtype: Union[Dict, List[Dict]]
        """

        new_states = []
        for text in texts:
            text = self.strip_answer_helper(text, "Abstract")
            new_state = states[0].copy()
            new_state["current"] = text
            new_states.append(new_state)
        return new_states

    def parse_generate_answer(self, state: Dict, texts: List[str]) -> List[Dict]:
        """
        Parse the response from the language model for a generate prompt.

        :param state: The thought state used to generate the prompt.
        :type state: Dict
        :param texts: The responses to the prompt from the language model.
        :type texts: List[str]
        :return: The new thought states after parsing the respones from the language model.
        :rtype: List[Dict]
        """
        new_states = []
        for text in texts:
            text = self.strip_answer_helper(text, "Abstract")
            new_state = state.copy()
            new_state["current"] = text
            new_states.append(new_state)
        return new_states

    def parse_score_answer(self, states: List[Dict], texts: List[str]) -> List[float]:
        """
        Parse the response from the language model for a score prompt.

        :param states: The thought states used to generate the prompt.
        :type states: List[Dict]
        :param texts: The responses to the prompt from the language model.
        :type texts: List[str]
        :return: The scores for the thought states.
        :rtype: List[float]
        :raise AssertionError: If the number of thought states is not one.
        """
        pass

    def parse_improve_answer(self, state: Dict, texts: List[str]) -> Dict:
        """
        Parse the response from the language model for an improve prompt.

        :param state: The thought state used to generate the prompt.
        :type state: Dict
        :param texts: The responses to the prompt from the language model.
        :type texts: List[str]
        :return: The new thought state after parsing the responses from the language model.
        :rtype: Dict
        """
        pass

    def parse_validation_answer(self, state: Dict, texts: List[str]) -> bool:
        """
        Parse the response from the language model for a validation prompt.

        :param state: The thought state used to generate the prompt.
        :type state: Dict
        :param texts: The responses to the prompt from the language model.
        :type texts: List[str]
        :return: Whether the thought state is valid or not.
        :rtype: bool
        """
        pass


def io() -> operations.GraphOfOperations:
    """
    Generates the Graph of Operations for the IO method.

    :return: Graph of Operations
    :rtype: GraphOfOperations
    """
    operations_graph = operations.GraphOfOperations()

    operations_graph.append_operation(operations.Generate(1, 1))
    operations_graph.append_operation(operations.Score(1, False, rouge1_f_gold_summary))

    return operations_graph


def cot() -> operations.GraphOfOperations:
    """
    Generates the Graph of Operations for the CoT method.

    :return: Graph of Operations
    :rtype: GraphOfOperations
    """
    operations_graph = operations.GraphOfOperations()

    operations_graph.append_operation(operations.Generate(1, 1))
    operations_graph.append_operation(operations.Score(1, False, rouge1_f_gold_summary))

    return operations_graph


def tot() -> operations.GraphOfOperations:
    """
    Generates the Graph of Operations for the ToT method.

    :return: Graph of Operations
    :rtype: GraphOfOperations
    """
    operations_graph = operations.GraphOfOperations()

    branch_factor = 3

    operations_graph.append_operation(operations.Generate(1, branch_factor))
    operations_graph.append_operation(operations.Score(3, False, rouge1_f_test_introduction))
    keep_best_1 = operations.KeepBestN(1, True)
    operations_graph.append_operation(keep_best_1)

    for _ in range(2):
        operations_graph.append_operation(operations.Generate(1, branch_factor))
        operations_graph.append_operation(operations.Score(3, False, rouge1_f_test_introduction))
        keep_best_2 = operations.KeepBestN(1, True)
        keep_best_2.add_predecessor(keep_best_1)
        operations_graph.append_operation(keep_best_2)
        keep_best_1 = keep_best_2
    
    operations_graph.append_operation(operations.Score(1, False, rouge1_f_gold_summary))

    return operations_graph


def got() -> operations.GraphOfOperations:
    """
    Generates the Graph of Operations for the GoT method, where generate thoughts
    are merged.

    :return: Graph of Operations
    :rtype: GraphOfOperations
    """
    operations_graph = operations.GraphOfOperations()

    branch_factor = 3

    operations_graph.append_operation(operations.Generate(1, branch_factor))
    operations_graph.append_operation(operations.Score(3, False, rouge1_f_test_introduction))
    keep_best = operations.KeepBestN(3, True)
    operations_graph.append_operation(keep_best)
    operations_graph.append_operation(operations.Aggregate(3))
    operations_graph.append_operation(operations.Score(3, False, rouge1_f_test_introduction))
    keep_best2 = operations.KeepBestN(1, True)
    keep_best2.add_predecessor(keep_best)
    operations_graph.append_operation(keep_best2)
    operations_graph.append_operation(operations.Generate(1, branch_factor))
    operations_graph.append_operation(operations.Score(3, False, rouge1_f_test_introduction))
    keep_best3 = operations.KeepBestN(1, True)
    keep_best3.add_predecessor(keep_best2)
    operations_graph.append_operation(keep_best3)
    operations_graph.append_operation(operations.Score(1, False, rouge1_f_gold_summary))

    return operations_graph

def dgot() -> operations.GraphOfOperations:
    """
    Generates the Graph of Operations for the GoT method, where generate thoughts
    are merged.

    :return: Graph of Operations
    :rtype: GraphOfOperations
    """
    operations_graph = operations.GraphOfOperations()

    branch_factor = 3

    operations_graph.append_operation(operations.DGenerateScore(1, branch_factor, rouge1_f_test_introduction, 0.2))
    operations_graph.append_operation(operations.Score(3, False, rouge1_f_test_introduction))
    keep_best = operations.KeepBestN(3, True)
    operations_graph.append_operation(keep_best)
    operations_graph.append_operation(operations.DAggregate(3, rouge1_f_test_introduction, 1, 0.5))
    operations_graph.append_operation(operations.Score(3, False, rouge1_f_test_introduction))
    keep_best2 = operations.KeepBestN(1, True)
    keep_best2.add_predecessor(keep_best)
    operations_graph.append_operation(keep_best2)
    operations_graph.append_operation(operations.DGenerateScore(1, branch_factor, rouge1_f_test_introduction, 0.2))
    operations_graph.append_operation(operations.Score(3, False, rouge1_f_test_introduction))
    keep_best3 = operations.KeepBestN(1, True)
    keep_best3.add_predecessor(keep_best2)
    operations_graph.append_operation(keep_best3)
    operations_graph.append_operation(operations.Score(1, False, rouge1_f_gold_summary))

    return operations_graph

def dgot_aggregate() -> operations.GraphOfOperations:
    """
    Generates the Graph of Operations for the GoT method, where generate thoughts
    are merged.

    :return: Graph of Operations
    :rtype: GraphOfOperations

    train_mean
    0.291453691460055
    0.31990248098859286
    0.32433038825757593
    """
    operations_graph = operations.GraphOfOperations()

    branch_factor = 3

    operations_graph.append_operation(operations.DGenerateScore(1, branch_factor, rouge1_f_test_introduction, 0.29))
    operations_graph.append_operation(operations.Score(3, False, rouge1_f_test_introduction))
    keep_best = operations.KeepBestN(3, True)
    operations_graph.append_operation(keep_best)
    operations_graph.append_operation(operations.DAggregate(3, rouge1_f_test_introduction, 0.31, 0.29))
    operations_graph.append_operation(operations.Score(3, False, rouge1_f_test_introduction))
    keep_best2 = operations.KeepBestN(1, True)
    keep_best2.add_predecessor(keep_best)
    operations_graph.append_operation(keep_best2)
    operations_graph.append_operation(operations.DGenerateScore(1, branch_factor, rouge1_f_test_introduction, 0.32))
    operations_graph.append_operation(operations.Score(3, False, rouge1_f_test_introduction))
    keep_best3 = operations.KeepBestN(1, True)
    keep_best3.add_predecessor(keep_best2)
    operations_graph.append_operation(keep_best3)
    operations_graph.append_operation(operations.Score(1, False, rouge1_f_gold_summary))

    return operations_graph

def dgot_p1() -> operations.GraphOfOperations:
    """
    Generates the Graph of Operations for the GoT method, where generate thoughts
    are merged.

    :return: Graph of Operations
    :rtype: GraphOfOperations

    train_mean
    0.3799134707233617
    0.3552190115399541
    0.35016805066509976
    """
    operations_graph = operations.GraphOfOperations()

    branch_factor = 3

    operations_graph.append_operation(operations.DGenerateScore(1, branch_factor, rouge1_f_test_introduction, 0.38))
    operations_graph.append_operation(operations.Score(3, False, rouge1_f_test_introduction))
    keep_best = operations.KeepBestN(3, True)
    operations_graph.append_operation(keep_best)
    operations_graph.append_operation(operations.DAggregate(3, rouge1_f_test_introduction, 0.355, 0.38))
    operations_graph.append_operation(operations.Score(3, False, rouge1_f_test_introduction))
    keep_best2 = operations.KeepBestN(1, True)
    keep_best2.add_predecessor(keep_best)
    operations_graph.append_operation(keep_best2)
    operations_graph.append_operation(operations.DGenerateScore(1, branch_factor, rouge1_f_test_introduction, 0.35))
    operations_graph.append_operation(operations.Score(3, False, rouge1_f_test_introduction))
    keep_best3 = operations.KeepBestN(1, True)
    keep_best3.add_predecessor(keep_best2)
    operations_graph.append_operation(keep_best3)
    operations_graph.append_operation(operations.Score(1, False, rouge1_f_gold_summary))

    return operations_graph

def dgot_p2() -> operations.GraphOfOperations:
    """
    Generates the Graph of Operations for the GoT method, where generate thoughts
    are merged.

    :return: Graph of Operations
    :rtype: GraphOfOperations

    train_mean
    0.4612847602978985
    0.42190314437333143
    0.4146454347817308
    """
    operations_graph = operations.GraphOfOperations()

    branch_factor = 3

    operations_graph.append_operation(operations.DGenerateScore(1, branch_factor, rouge1_f_test_introduction, 0.46))
    operations_graph.append_operation(operations.Score(3, False, rouge1_f_test_introduction))
    keep_best = operations.KeepBestN(3, True)
    operations_graph.append_operation(keep_best)
    operations_graph.append_operation(operations.DAggregate(3, rouge1_f_test_introduction, 0.42, 0.46))
    operations_graph.append_operation(operations.Score(3, False, rouge1_f_test_introduction))
    keep_best2 = operations.KeepBestN(1, True)
    keep_best2.add_predecessor(keep_best)
    operations_graph.append_operation(keep_best2)
    operations_graph.append_operation(operations.DGenerateScore(1, branch_factor, rouge1_f_test_introduction, 0.41))
    operations_graph.append_operation(operations.Score(3, False, rouge1_f_test_introduction))
    keep_best3 = operations.KeepBestN(1, True)
    keep_best3.add_predecessor(keep_best2)
    operations_graph.append_operation(keep_best3)
    operations_graph.append_operation(operations.Score(1, False, rouge1_f_gold_summary))

    return operations_graph

def dgot_p3() -> operations.GraphOfOperations:
    """
    Generates the Graph of Operations for the GoT method, where generate thoughts
    are merged.

    :return: Graph of Operations
    :rtype: GraphOfOperations

    train_mean
    0.5645194089605905
    0.5065043969825097
    0.49644701529954494
    """
    operations_graph = operations.GraphOfOperations()

    branch_factor = 3

    operations_graph.append_operation(operations.DGenerateScore(1, branch_factor, rouge1_f_test_introduction, 0.56))
    operations_graph.append_operation(operations.Score(3, False, rouge1_f_test_introduction))
    keep_best = operations.KeepBestN(3, True)
    operations_graph.append_operation(keep_best)
    operations_graph.append_operation(operations.DAggregate(3, rouge1_f_test_introduction, 0.50, 0.56))
    operations_graph.append_operation(operations.Score(3, False, rouge1_f_test_introduction))
    keep_best2 = operations.KeepBestN(1, True)
    keep_best2.add_predecessor(keep_best)
    operations_graph.append_operation(keep_best2)
    operations_graph.append_operation(operations.DGenerateScore(1, branch_factor, rouge1_f_test_introduction, 0.49))
    operations_graph.append_operation(operations.Score(3, False, rouge1_f_test_introduction))
    keep_best3 = operations.KeepBestN(1, True)
    keep_best3.add_predecessor(keep_best2)
    operations_graph.append_operation(keep_best3)
    operations_graph.append_operation(operations.Score(1, False, rouge1_f_gold_summary))

    return operations_graph

def run(
    data_ids: List[int],
    methods: List[Callable[[], operations.GraphOfOperations]],
    max_input_prompt_tokens_list: List[int],
    budget: float,
    lm_name: str,
    data_path: str, #= './data/available_induc_test_graph.json',
    save_pmc_folder: str, # = './data/available_induc_test/pmc/',
    save_pm_folder: str, # = './data/available_induc_test/pm/',
) -> float:
    """
    Controller function that executes each specified method for each specified
    sample while the budget is not exhausted.

    :param data_ids: Indices of the sample to be run.
    :type data_ids: List[int]
    :param methods: List of functions to generate Graphs of Operations.
    :type methods: Each function generates a Graph of Operation.
    :param budget: Language model budget for the execution in dollars.
    :type budget: float
    :param lm_name: Name of the language model to be used.
    :type lm_name: str
    :return: Spent budget in dollars.
    :rtype: float
    """

    orig_budget = budget
    # 打开JSON文件
    with open(data_path, 'r') as json_file:
        # 从文件中加载JSON数据
        data = json.load(json_file)

    # data_ids = [0] # 测试用
    if data_ids is None or len(data_ids) == 0:
        data_ids = list(range(len(data['PMCid'])))
    selected_data = [data['PMCid'][i] for i in data_ids]
    selected_data_reference = [data['references'][i] for i in data_ids]

    if not os.path.exists(os.path.join(os.path.dirname(__file__), "results")):
        os.makedirs(os.path.join(os.path.dirname(__file__), "results"))
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    extra_info = f"{lm_name}_{'-'.join([method.__name__ for method in methods])}"
    folder_name = f"results/{extra_info}_{timestamp}"
    os.makedirs(os.path.join(os.path.dirname(__file__), folder_name))

    config = {
        "data": selected_data,
        "methods": [method.__name__ for method in methods],
        "lm": lm_name,
        "budget": budget,
    }
    with open(
        os.path.join(os.path.dirname(__file__), folder_name, "config.json"), "w"
    ) as f:
        json.dump(config, f)

    logging.basicConfig(
        filename=f"{folder_name}/log.log",
        filemode="w",
        format="%(name)s - %(levelname)s - %(message)s",
        level=logging.DEBUG,
    )

    inference_time_dict = {}
    inference_num_dict = {}

    for method in methods:
        for max_input_prompt_tokens in max_input_prompt_tokens_list:
            os.makedirs(
                os.path.join(os.path.dirname(__file__), folder_name, method.__name__+'_'+max_input_prompt_tokens)
            )
            inference_time_dict[method.__name__+'_'+max_input_prompt_tokens] = 0
            inference_num_dict[method.__name__+'_'+max_input_prompt_tokens] = 0
    
    # 遍历每篇文章数据
    for index, data in tqdm(enumerate(selected_data), total=len(selected_data)):
        logging.info(f"Running data {data}")
        if budget <= 0.0:
            logging.error(
                f"Budget has been depleted, stopping. Data {data} has not been run."
            )
            break
        # 读取源文章及参考文献数据
        # 源文章
        #save_pmc_folder = './data/available_induc_test/pmc/'
        origin_path = save_pmc_folder + data + '.json'
        # dict = {'article_title_text': article_title_text, 'abstract_text': abstract_text,
        #       'introduction_text': introduction_text, 'sec_dict': sec_dict}
        with open(origin_path, 'r') as json_file:
            pmc_dict = json.load(json_file)
        # 参考文献
        #save_pm_folder = './data/available_induc_test/pm/'
        reference_path = save_pm_folder + data + '.json'
        with open(reference_path, 'r') as json_file:
            reference_title_abstract_dict = json.load(json_file)

        for method in methods:
            for max_input_prompt_tokens in max_input_prompt_tokens_list:
                logging.info(f"Running method {method.__name__}")
                logging.info(f"Budget left: {budget}")
                # record start time
                start_time = time.time()

                if budget <= 0.0:
                    logging.error(
                        f"Budget has been depleted, stopping. Method {method.__name__} has not been run."
                    )
                    break
                lm = controller.ChatGLM(
                    "./graph_of_thoughts/controller/config.json",
                    model_name=lm_name,
                    cache=False,
                )
                operations_graph = method()
                executor = controller.Controller(
                    lm,
                    operations_graph,
                    GenAbstractPrompter(max_input_prompt_tokens=max_input_prompt_tokens),
                    GenAbstractParser(),
                    {
                        "origin_title": pmc_dict["article_title_text"],
                        "origin_abstract": pmc_dict["abstract_text"],
                        "origin_introduction": pmc_dict["introduction_text"],
                        "origin_info": pmc_dict["sec_dict"],
                        "reference_info": reference_title_abstract_dict,
                        #"parts": set(),
                        "current": "",
                        "method": method.__name__,
                    },
                )
                try:
                    executor.run()
                except Exception as e:
                    logging.error(f"Exception: {e}")

                # record end time
                end_time = time.time()
                execution_time = end_time - start_time
                inference_time_dict[method.__name__+'_'+max_input_prompt_tokens] += execution_time
                inference_num_dict[method.__name__+'_'+max_input_prompt_tokens] += 1

                path = os.path.join(
                    os.path.dirname(__file__),
                    folder_name,
                    method.__name__+'_'+max_input_prompt_tokens,
                    f"{data_ids[0] + index}.json",
                )
                # for operation in operations_graph.operations:
                #     for thought in operation.thoughts:
                #         thought.state["parts"] = list(thought.state["parts"])
                # 输出结果与 gold summary 比较

                executor.output_graph(path)
                budget -= lm.cost

    # save inference time as .txt
    data_to_write = "approach_max_input_prompt_length time(s) inference_num time_per_inference"
    for key, value in inference_time_dict.items():
        data_to_write += f"{key} {value} {inference_num_dict[key]} {value/inference_num_dict[key]}\n"

    file_name = os.path.join(
                os.path.dirname(__file__),
                folder_name,
                "inference_times.txt"
    )

    # write in txt
    with open(file_name, "w") as file:
        file.write(data_to_write)

    # save cut prompt num to .txt
    data_to_write = "max_input_prompt_length cut_num prompt_num cut_num/prompt_num"
    global generate_prompt_nums
    global cut_abstract_nums
    for key, value in cut_abstract_nums.items():
        data_to_write += f"{key} {value} {generate_prompt_nums[key]} {value/generate_prompt_nums[key]}\n"

    file_name = os.path.join(
                os.path.dirname(__file__),
                folder_name,
                "cut_num_times.txt"
    )

    # write in txt
    with open(file_name, "w") as file:
        file.write(data_to_write)

    return orig_budget - budget


if __name__ == "__main__":
    """
    Input (x1, x2, ...): Original Article and it's References
    Output (y): Generated Abstract
    Evaluation: Rouge
    """
    parser = argparse.ArgumentParser(description='Description of your program')
    parser.add_argument('--begin', type=int, default=0, help='Begin index')
    parser.add_argument('--end', type=int, default=100, help='End index')
    parser.add_argument('--mode', type=str, default="test", help='train or test')
    parser.add_argument('--model', type=str, default="internlm2", help='model name')
    args = parser.parse_args()

    mode = args.mode
    if(mode == 'test'):
        data_path = './data/available_induc_test_graph.json'
        save_pmc_folder = './data/available_induc_test/pmc/'
        save_pm_folder = './data/available_induc_test/pm/'
    else:
        data_path = './data/available_induc_train_graph.json'
        save_pmc_folder = './data/available_induc_train/pmc/'
        save_pm_folder = './data/available_induc_train/pm/'

    budget = 3000000000
    samples = [item for item in range(int(args.begin), int(args.end))]
    # approaches = [io, cot, tot, got, dgot_aggregate]
    approaches = [got]
    # max_input_prompt_tokens_list = [256, 512, 1024, 2048, 4096, 8192, 16384]
    max_input_prompt_tokens_list = [256, 512]

    for max_input_prompt_tokens in max_input_prompt_tokens_list:
        generate_prompt_nums[str(max_input_prompt_tokens)] = 0
        cut_abstract_nums[str(max_input_prompt_tokens)] = 0

    spent = run(samples, approaches, max_input_prompt_tokens_list, budget, args.model, data_path, save_pmc_folder, save_pm_folder)

    logging.info(f"Spent {spent} out of {budget} budget.")
