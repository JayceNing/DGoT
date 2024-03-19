# author: Jayce Ning

from typing import Dict, List
import tempfile
import os
from .cal_rouge import test_rouge
import logging
import json
import codecs
global rouge1f_num
rouge1f_num = 0

def cal_rouge_f(gold_summary, generate_abstract):
    try:
        # 创建临时文件并将内容写入
        with tempfile.NamedTemporaryFile(mode='w', delete=False, encoding='utf-8') as temp_file:
            temp_file.write(gold_summary.replace("\n", " "))
            gold_summary_file_path = temp_file.name
        # 创建临时文件并将内容写入
        with tempfile.NamedTemporaryFile(mode='w', delete=False, encoding='utf-8') as temp_file:
            temp_file.write(generate_abstract.replace("\n", " ").replace("<", " ").replace(">", " "))
            generate_abstract_file_path = temp_file.name

        gold_summary = codecs.open(gold_summary_file_path, encoding="utf-8")
        generate_abstract = codecs.open(generate_abstract_file_path, encoding="utf-8")
        results_dict = test_rouge(gold_summary, generate_abstract, 1)
        global rouge1f_num
        if results_dict["rouge_1_f_score"]>0:
            rouge1f_num += 1
            print("==============================================================")
        else:
            pass
        # 删除临时文件
        os.remove(gold_summary_file_path)
        os.remove(generate_abstract_file_path)

        rouge1f = results_dict["rouge_1_f_score"]
        rouge2f = results_dict["rouge_2_f_score"]
        rougelf = results_dict["rouge_l_f_score"]
        print('rouge1f_num:', rouge1f_num)

        return rouge1f, rouge2f, rougelf
    except:
        return 0, 0, 0

def rouge1_f_test_introduction(state: Dict) -> float:
    """
    Function to locally calculate rouge f1 score.

    :param state: Thought state to be scored.
    :type state: Dict
    :return: Number of errors.
    :rtype: float
    """

    try:
        gold_summary = state["origin_introduction"].replace("\n", " ")  # 对于测试集，使用 introduction 作为 gold summary
        generate_abstract = state["current"].replace("\n", " ").replace("<", " ").replace(">", " ")
        #logging.warning(f"=========================================")
        #logging.warning(gold_summary)
        #logging.warning(generate_abstract)

        # 创建临时文件并将内容写入
        with tempfile.NamedTemporaryFile(mode='w', delete=False, encoding='utf-8') as temp_file:
            temp_file.write(gold_summary)
            gold_summary_file_path = temp_file.name
        # 创建临时文件并将内容写入
        with tempfile.NamedTemporaryFile(mode='w', delete=False, encoding='utf-8') as temp_file:
            temp_file.write(generate_abstract)
            generate_abstract_file_path = temp_file.name

        gold_summary = codecs.open(gold_summary_file_path, encoding="utf-8")
        generate_abstract = codecs.open(generate_abstract_file_path, encoding="utf-8")
        results_dict = test_rouge(gold_summary, generate_abstract, 1)
        #logging.warning(results_dict)
        # 删除临时文件
        os.remove(gold_summary_file_path)
        os.remove(generate_abstract_file_path)

        rouge1f = results_dict["rouge_1_f_score"]
        #logging.warning(rouge1f)
        #logging.warning(type(rouge1f))

        return rouge1f
    except:
        return 0

def rouge1_f_gold_summary(state: Dict) -> float:
    """
    Function to locally calculate rouge f1 score.

    :param state: Thought state to be scored.
    :type state: Dict
    :return: Number of errors.
    :rtype: float
    """

    try:
        gold_summary = state["origin_abstract"].replace("\n", " ")  # 对于测试集，使用 introduction 作为 gold summary
        generate_abstract = state["current"].replace("\n", " ").replace("<", " ").replace(">", " ")
        #logging.warning(f"=========================================")
        #logging.warning(gold_summary)
        #logging.warning(generate_abstract)

        # 创建临时文件并将内容写入
        with tempfile.NamedTemporaryFile(mode='w', delete=False, encoding='utf-8') as temp_file:
            temp_file.write(gold_summary)
            gold_summary_file_path = temp_file.name
        # 创建临时文件并将内容写入
        with tempfile.NamedTemporaryFile(mode='w', delete=False, encoding='utf-8') as temp_file:
            temp_file.write(generate_abstract)
            generate_abstract_file_path = temp_file.name

        gold_summary = codecs.open(gold_summary_file_path, encoding="utf-8")
        generate_abstract = codecs.open(generate_abstract_file_path, encoding="utf-8")
        results_dict = test_rouge(gold_summary, generate_abstract, 1)
        #logging.warning(results_dict)
        # 删除临时文件
        os.remove(gold_summary_file_path)
        os.remove(generate_abstract_file_path)

        rouge1f = results_dict["rouge_1_f_score"]
        #logging.warning(rouge1f)
        #logging.warning(type(rouge1f))
        state["rouge"] = results_dict

        return rouge1f
    except:
        return 0
