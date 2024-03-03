import requests
import os
import json
import time
from tqdm import tqdm
from bs4 import BeautifulSoup
import argparse

from utils import read_pmc, read_pm


def download_one_pmc_article(pmc_id, save_path):
    # 构建PMC文章的URL
    # https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pmc&id=5436877
    article_url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pmc&id={pmc_id}"
    
    # 发送GET请求
    response = requests.get(article_url)
    
    # 检查响应
    if response.status_code == 200:
        # 获取文章内容
        article_content = response.text
        # 在这里，您可以对文章内容进行处理，保存到文件或者进行其他操作
        # 例如，保存到文件
        with open(f"{save_path}/{pmc_id}.html", "w", encoding="utf-8") as file:
            file.write(article_content)
        print(f"Article {pmc_id} downloaded successfully.")
        return 1
    else:
        print(f"Error occurred while downloading article {pmc_id}.")
        return 0

def download_one_pm_article(pm_id, save_path):
    # 构建PMC文章的URL
    # https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pmc&id=5436877
    article_url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pubmed&id={pm_id}"
    
    # 发送GET请求
    response = requests.get(article_url)
    
    # 检查响应
    if response.status_code == 200:
        # 获取文章内容
        article_content = response.text
        # 在这里，您可以对文章内容进行处理，保存到文件或者进行其他操作
        # 例如，保存到文件
        with open(f"{save_path}/{pm_id}.html", "w", encoding="utf-8") as file:
            file.write(article_content)
        print(f"Article {pm_id} downloaded successfully.")
        return 1
    else:
        print(f"Error occurred while downloading article {pm_id}.")
        return 0
    
def download_pmc_article(data, mode, num):
    ###################################
    # 下载 PMC 文章
    ###################################
    save_path = f"./data/{mode}/pmc/"
    if not os.path.exists(save_path):
        # 如果文件夹不存在，则创建它
        os.makedirs(save_path)

    PMC_num = 0
    for key in tqdm(data['PMCid'], total=num, desc="Downloading articles"):
        pmc_id = data['PMCid'][key][3:]
        while not download_one_pmc_article(pmc_id, save_path):
            time.sleep(1)
        # res = download_one_pmc_article(pmc_id, save_path)
        PMC_num += 1 # res
        if PMC_num >= num:
            break
    print("下载PMC文章数量为：", PMC_num)

def download_pm_article(data, mode, num):
    ###################################
    # 下载 pubmed 文章
    ###################################
    # 使用集合（set）来存储不重复的元素
    unique_elements = set()

    # 遍历字典的值（列表），将元素添加到集合中
    PMC_num = 0
    for key in data['references']:
        unique_elements.update(data['references'][key])
        PMC_num += 1
        if PMC_num >= num:
            break

    save_path = f"./data/{mode}/pm/"
    if not os.path.exists(save_path):
        # 如果文件夹不存在，则创建它
        os.makedirs(save_path)
        
    pm_num = 0    
    for key in tqdm(unique_elements, total=len(unique_elements), desc="Downloading articles"):
        pm_id = key
        while not download_one_pm_article(pm_id, save_path):
            time.sleep(1)
        # res = download_pm_article(pm_id, save_path)
        pm_num += 1  # res
    print("下载pm文章数量为：", pm_num)

def process_raw_data(mode, selected_data):
    # 遍历每篇文章数据
    save_folder = f'./data/available_induc_{mode}/'
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    save_pmc_folder = f'./data/available_induc_{mode}/pmc/'
    if not os.path.exists(save_pmc_folder):
        os.makedirs(save_pmc_folder)
    save_pm_folder = f'./data/available_induc_{mode}/pm/'
    if not os.path.exists(save_pm_folder):
        os.makedirs(save_pm_folder)


    for index, data in tqdm(enumerate(selected_data), total=len(selected_data)):
        # 读取源文章及参考文献数据
        # 源文章
        origin_path = f'./data/{mode}/pmc/' + data + '.html'
        article_title_text, abstract_text, introduction_text, sec_dict = read_pmc(origin_path)
        if article_title_text == "":
            print("article_title_text error:", data)
        elif abstract_text == "":
            print("abstract_text error:", data)
        elif introduction_text == "":
            print("introduction_text error:", data)
        else:
            dict = {'article_title_text': article_title_text, 'abstract_text': abstract_text,
                'introduction_text': introduction_text, 'sec_dict': sec_dict}
            
            json_file_path = save_pmc_folder + data + '.json'
            with open(json_file_path, 'w') as json_file:
                json.dump(dict, json_file)
        
            # 参考文献
            reference_title_abstract_dict = {}
            for pm_id in selected_data_reference[index]:
                reference_path = f'./data/{mode}/pm/' + pm_id + '.html'
                reference_title_text, reference_abstract_text = read_pm(reference_path)
                reference_title_abstract_dict[reference_title_text] = reference_abstract_text
            json_file_path = save_pm_folder + data + '.json'
            with open(json_file_path, 'w') as json_file:
                json.dump(reference_title_abstract_dict, json_file)

def get_availabe_pmc_data(mode, selected_data):
    available_num = 0
    available_pmc_list = []
    available_pmc_idx = []

    for index, data in enumerate(selected_data):
        path = f'./data/{mode}/pmc/' + data + '.html'
        # 打开并读取HTML文件
        with open(path, 'r', encoding='utf-8') as file:
            html_content = file.read()
        # 使用Beautiful Soup解析HTML内容
        soup = BeautifulSoup(html_content, "html.parser")
        # 查找<sec>标签，并指定sec-type为"intro"
        sec_tag = soup.find_all("sec")
        
        # sec_dict = {}
        
        # 如果找到了<sec>标签
        if sec_tag:
            # print(len(sec_tag))
            # 查找<title>标签
            for tag in sec_tag:
                title_tag = tag.find("title")
                if title_tag:
                    #sec_dict[title_tag.text] = tag.text
                    if title_tag.text in ['Introduction', 'INTRODUCTION', '1. Introduction', '1 INTRODUCTION', 'introduction']:
                        # 查找<abstract>标签
                        abstract_tag = soup.find("abstract")
                        
                        # 提取<abstract>标签中的文本内容
                        if abstract_tag:
                            # abstract_text = abstract_tag.text
                            available_num += 1
                            available_pmc_list.append(data)
                            available_pmc_idx.append(index)
                            
                        else:
                            print("Abstract tag not found in the HTML content.")
                        
        else:
            print("No <sec> tag found in the HTML content.")
            print("wrong data: ", data)


    print("可用文章数量:", available_num)
    return available_pmc_list, available_pmc_idx

def get_availabe_pm_data(mode, available_pmc_idx):
    available_pm_list = []
    available_pm_num = 0
    for reference_list_idx in available_pmc_idx:
        for i in selected_data_reference[reference_list_idx]:
            if i not in available_pm_list:
                # 打开并读取HTML文件
                path = f'./data/{mode}/pm/' + i + '.html'
                with open(path, 'r', encoding='utf-8') as file:
                    html_content = file.read()
                
                # 使用Beautiful Soup解析HTML内容
                soup = BeautifulSoup(html_content, "html.parser")
                
                # 查找<title>标签
                title_tag = soup.find("articletitle")
                
                # 提取<title>标签中的文本内容
                if title_tag:
                    #title_text = title_tag.text
                    # 查找<abstract>标签
                    abstract_tag = soup.find("abstract")
                    
                    # 提取<abstract>标签中的文本内容
                    if abstract_tag:
                        #abstract_text = abstract_tag.text
                        # print("Abstract:", abstract_text)
                        available_pm_list.append(i)
                        available_pm_num += 1
                    else:
                        print("Abstract tag not found in the HTML content.")
    print("pm可用文章数量：", available_pm_num)
    return available_pm_list

def generate_available_dict(selected_data, selected_data_reference, available_pm_list, required_num):
    available_graph_dict = {}
    available_graph_dict['PMCid'] = []
    available_graph_dict['references'] = []

    data_num = 0
    for index, data in enumerate(selected_data):
        if data in available_pmc_list:
            reference_list = selected_data_reference[index]
            if reference_list == []:
                available_graph_dict['PMCid'].append(data)
                available_graph_dict['references'].append([])
                data_num += 1
            else:
                no_flag = 0
                for i in reference_list:
                    if i not in available_pm_list:
                        no_flag += 1
                    else:
                        pass
                if no_flag == 0:
                    available_graph_dict['PMCid'].append(data)
                    available_graph_dict['references'].append(reference_list)
                    data_num += 1
        if data_num >= required_num:
            break
        
    print(len(available_graph_dict['references']))
    return available_graph_dict

if __name__ == "__main__":
    """
    Download 100 train and 100 test articles
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--required_num', type=int, default=100, help='Number of articles data required')
    args = parser.parse_args()

    required_num = args.required_num
    # About half of the downloaded articles are not available
    download_num = 2 * required_num
    modes = ['test', 'train']

    for mode in modes:
        # 打开JSON文件
        with open(f'./PubMedCite/induc_graph/{mode}_graph.json', 'r') as json_file:
            # 从文件中加载JSON数据
            data = json.load(json_file)

        print('Downloading pmc article')
        download_pmc_article(data, mode, download_num)
        print('Downloading pm article')
        # 
        download_pm_article(data, mode, download_num)

        data_ids = list(range(download_num))

        selected_data = [data['PMCid'][f'{i}'][3:] for i in data_ids]
        selected_data_reference = [data['references'][f'{i}'] for i in data_ids]

        print('Processing raw data')
        process_raw_data(mode, selected_data)

        print('Get availabe pmc data')
        available_pmc_list, available_pmc_idx = get_availabe_pmc_data(mode, selected_data)
        print('Get availabe pm data')
        available_pm_list = get_availabe_pm_data(mode, available_pmc_idx)
        print('Generating available dict')
        available_graph_dict = generate_available_dict(selected_data, selected_data_reference, available_pm_list, required_num)

        save_path = f'./data/available_induc_{mode}_graph.json'
        # 将排序后的列表保存为JSON文件
        with open(save_path, 'w') as json_file:
            json.dump(available_graph_dict, json_file)