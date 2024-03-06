import matplotlib.pyplot as plt
import numpy as np
import os
import json

def read_each_task_results(folder_path_list):
    r_1 = []
    r_2 = []
    r_L = []
    prompt_tokens_list = []
    completion_tokens_list = []
    cost_list = []
    avliable_num = 0
    
    for folder_path in folder_path_list:
        # 获取文件夹中的所有文件和子文件夹
        for foldername, subfolders, filenames in os.walk(folder_path):
            # 打印文件
            for filename in filenames:
                #print('文件：' + filename)
                path = os.path.join(folder_path, filename)
                with open(path, 'r') as json_file:
                    data = json.load(json_file)
        
                try:
                    r_1.append(data[-2]['thoughts'][-1]['rouge']['rouge_1_f_score'])
                    r_2.append(data[-2]['thoughts'][-1]['rouge']['rouge_2_f_score'])
                    r_L.append(data[-2]['thoughts'][-1]['rouge']['rouge_l_f_score'])
                    prompt_tokens = data[-1]["prompt_tokens"]
                    completion_tokens = data[-1]["completion_tokens"]
                    cost = data[-1]["cost"]
                
                    prompt_tokens_list.append(prompt_tokens)
                    completion_tokens_list.append(completion_tokens)
                    cost_list.append(cost)
                    avliable_num += 1
                except:
                    #print('err')
                    pass
    mean_r_1 = sum(r_1) / len(r_1)
    mean_r_2 = sum(r_2) / len(r_2)
    mean_r_L = sum(r_L) / len(r_L)
    mean_prompt_tokens = sum(prompt_tokens_list) / len(prompt_tokens_list)
    mean_completion_tokens = sum(completion_tokens_list) / len(completion_tokens_list)
    mean_cost = sum(cost_list) / len(cost_list)
    
    return r_1, r_2, r_L, mean_r_1, mean_r_2, mean_r_L, mean_prompt_tokens, mean_completion_tokens, mean_cost

def process_data_for_all_tasks(folder_path):
    # dict to store r_1 for each task to draw line-box figure
    r_1_distribution_dict = {}
    # dict to store mean cost for each task to draw bar figure
    mean_cost_dict = {}
    
    # save process result to .txt
    data_to_write = "task_name r_1 r_2 r_L mean_prompt_tokens mean_completion_tokens mean_cost\n"
    mean_r_1_list = []
    for foldername, subfolders, filenames in os.walk(folder_path):
        for subfolder in subfolders:
            if len(subfolder.split("_"))>1:
                if subfolder.split("_")[0] == "io":
                    task_name = subfolder.replace("io", "IO").replace("_", " ")
                elif subfolder.split("_")[0] == "cot":
                    task_name = subfolder.replace("cot", "CoT").replace("_", " ")
                elif subfolder.split("_")[0] == "tot":
                    task_name = subfolder.replace("tot", "ToT").replace("_", " ")
                elif subfolder.split("_")[0] == "got":
                    task_name = subfolder.replace("got", "GoT").replace("_", " ")
                elif subfolder.split("_")[0] == "dgot":
                    task_name = subfolder.replace("dgot", "DGoT").replace("_", " ")
            else:
                task_name = subfolder
            print("processing " + task_name)
            r_1, r_2, r_L, mean_r_1, mean_r_2, mean_r_L, mean_prompt_tokens, mean_completion_tokens, mean_cost = read_each_task_results([os.path.join(folder_path, subfolder)])
            data_to_write += f"{task_name} {mean_r_1} {mean_r_2} {mean_r_L} {mean_prompt_tokens} {mean_completion_tokens} {mean_cost}\n"
            mean_r_1_list.append(mean_r_1)
            r_1_distribution_dict[task_name] = r_1
            mean_cost_dict[task_name] = mean_cost
    file_name = os.path.join(folder_path, "results_overview.txt")
    # write in txt
    with open(file_name, "w") as file:
        file.write(data_to_write)

    return r_1_distribution_dict, mean_r_1_list, mean_cost_dict

def draw_line_box_bar_figure(r_1_distribution_dict, mean_r_1_list, mean_cost_dict, folder_path):
    # 创建箱线图的坐标轴
    fig, ax1 = plt.subplots(figsize=(8, 6))
    # 创建柱状图的坐标轴
    ax2 = ax1.twinx()
    # 使用sorted函数和lambda表达式按值排序字典的键  
    sorted_key = sorted(mean_cost_dict, key=lambda k: mean_cost_dict[k], reverse=False)
    bar_data = []
    line_box_data = []
    # 输出排序后的键列表  
    for key in sorted_key:
        bar_data.append(mean_cost_dict[key])
        line_box_data.append(r_1_distribution_dict[key])

    r1_mean_for_label_position = sum(mean_r_1_list)/len(mean_r_1_list)
    print(r1_mean_for_label_position)
    
    # 绘制柱状图
    ax2.bar(list(range(1, 1+len(mean_cost_dict))), bar_data, alpha=0.7, color=(127/255, 127/255, 255/255), label='Mean Cost')
    label = ax1.text(len(mean_cost_dict)+1, r1_mean_for_label_position, "the lower the better", ha='center', va='center', fontsize=12, fontweight='bold',  rotation=90, color=(127/255, 127/255, 255/255))
    ax2.set_ylabel('Mean Cost                                                :', color=(127/255, 127/255, 255/255))
    
    # 指定箱线图的箱体和中位线颜色
    boxprops = dict(facecolor='white', color='black')
    medianprops = dict(color='red')
    
    # 绘制箱线图
    ax1.boxplot(line_box_data, notch=True, patch_artist=True, boxprops=boxprops, medianprops=medianprops)
    
    label = ax1.text(0.10, r1_mean_for_label_position, "the higher the better", ha='center', va='center', fontsize=12, fontweight='bold',  rotation=90)
    ax1.set_ylabel('Socre                                                :')

    xtick_labels = []
    # 设置x轴刻度标签
    if "test_prompt_length" in folder_path:
        for key in sorted_key:
            xtick_labels.append(key.split(" ")[0] + " " + key.split(" ")[1])
    elif "test_nodes_num" in folder_path:
        for key in sorted_key:
            xtick_labels.append(key.split(" ")[0] + " " + key.split(" ")[2])
    else:
        xtick_labels = sorted_key
    ax1.set_xticklabels(xtick_labels)
    ax2.set_xticks(list(range(1, 1+len(mean_cost_dict))))
    # 设置y轴刻度颜色
    ax2.tick_params(axis='y', colors=(127/255, 127/255, 255/255))  # 这里将y轴刻度颜色设置为蓝色
    # 显示图例
    ax2.legend()
    # 去除白边
    plt.tight_layout()
    plt.savefig(os.path.join(folder_path, 'r_1_distribution_and_mean_cost_figure.png'))
    plt.show()