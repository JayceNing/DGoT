import matplotlib.pyplot as plt
import numpy as np
import os
import json
import seaborn as sns
import pandas as pd
import statistics
import math
from scipy.stats import gumbel_r
from .utils import cal_rouge_f

def read_each_task_results(folder_path_list, task_category):
    r_1 = []
    r_2 = []
    r_L = []
    r_i = []
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
                    if "cot" in folder_path.split("/")[-1]:
                        generate_result = data[-2]['thoughts'][-1]['current']
                        gold_summary = data[-2]['thoughts'][-1]['origin_abstract']
                        rouge1f, rouge2f, rougelf = cal_rouge_f(gold_summary, generate_result)
                        r_1.append(rouge1f)
                        r_2.append(rouge2f)
                        r_L.append(rougelf)
                    else:
                        r_1.append(data[-2]['thoughts'][-1]['rouge']['rouge_1_f_score'])
                        r_2.append(data[-2]['thoughts'][-1]['rouge']['rouge_2_f_score'])
                        r_L.append(data[-2]['thoughts'][-1]['rouge']['rouge_l_f_score'])
                    if task_category == "with_r_i":
                        r_i.append(data[-3]['scores'][0])
                    else:
                        generate_result = data[-2]['thoughts'][-1]['current']
                        origin_introduction = data[-2]['thoughts'][-1]['origin_introduction']
                        rouge1f, rouge2f, rougelf = cal_rouge_f(origin_introduction, generate_result)
                        r_i.append(rouge1f)

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
    mean_r_i = sum(r_i) / len(r_i)
    mean_prompt_tokens = sum(prompt_tokens_list) / len(prompt_tokens_list)
    mean_completion_tokens = sum(completion_tokens_list) / len(completion_tokens_list)
    mean_cost = sum(cost_list) / len(cost_list)
    
    return r_1, r_2, r_L, r_i, mean_r_1, mean_r_2, mean_r_L, mean_r_i, mean_prompt_tokens, mean_completion_tokens, mean_cost

def process_data_for_all_tasks(folder_path):
    # dict to store r_1 for each task to draw line-box figure
    r_1_distribution_dict = {}
    r_i_distribution_dict = {}
    # dict to store mean cost for each task to draw bar figure
    mean_cost_dict = {}
    
    # save process result to .txt
    data_to_write = ""
    mean_r_1_list = []
    for foldername, subfolders, filenames in os.walk(folder_path):
        for subfolder in subfolders:
            task_category = "without_r_i"
            if len(subfolder.split("_"))>1:
                if subfolder.split("_")[0] == "io":
                    task_name = subfolder.replace("io", "IO").replace("_", " ")
                elif subfolder.split("_")[0] == "cot":
                    task_name = subfolder.replace("cot", "CoT").replace("_", " ")
                elif subfolder.split("_")[0] == "tot":
                    task_name = subfolder.replace("tot", "ToT").replace("_", " ")
                elif subfolder.split("_")[0] == "got":
                    task_name = subfolder.replace("got", "GoT").replace("_", " ")
                    task_category = "with_r_i"
                elif subfolder.split("_")[0] == "dgot":
                    task_name = subfolder.replace("dgot", "DGoT").replace("_", " ")
                    task_category = "with_r_i"
            else:
                task_name = subfolder
            print("processing " + task_name)
            #if task_category == "with_r_i":
            r_1, r_2, r_L, r_i, mean_r_1, mean_r_2, mean_r_L, mean_r_i, mean_prompt_tokens, mean_completion_tokens, mean_cost = read_each_task_results([os.path.join(folder_path, subfolder)], task_category)
            data_to_write += "task_name r_1 r_2 r_L r_i mean_prompt_tokens mean_completion_tokens mean_cost\n"
            data_to_write += f"{task_name} {mean_r_1} {mean_r_2} {mean_r_L} {mean_r_i} {mean_prompt_tokens} {mean_completion_tokens} {mean_cost}\n"
            r_i_distribution_dict[task_name] = r_i
            mean_r_1_list.append(mean_r_1)
            r_1_distribution_dict[task_name] = r_1
            mean_cost_dict[task_name] = mean_cost
    file_name = os.path.join(folder_path, "results_overview.txt")
    # write in txt
    with open(file_name, "w") as file:
        file.write(data_to_write)

    return r_1_distribution_dict, r_i_distribution_dict, mean_r_1_list, mean_cost_dict

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
    # label = ax1.text(len(mean_cost_dict)+1, r1_mean_for_label_position, "the lower the better", ha='center', va='center', fontsize=12, fontweight='bold',  rotation=90, color=(127/255, 127/255, 255/255))
    ax2.set_ylabel('Mean Cost', color=(127/255, 127/255, 255/255))
    
    # 指定箱线图的箱体和中位线颜色
    boxprops = dict(facecolor='white', color='black')
    medianprops = dict(color='red')
    
    # 绘制箱线图
    ax1.boxplot(line_box_data, notch=True, patch_artist=True, boxprops=boxprops, medianprops=medianprops)
    
    # label = ax1.text(0.10, r1_mean_for_label_position, "the higher the better", ha='center', va='center', fontsize=12, fontweight='bold',  rotation=90)
    ax1.set_ylabel('Socre')

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

def draw_main_result_figure(r_1_distribution_dict, r_i_distribution_dict, mean_r_1_list, mean_cost_dict, folder_path):
    # 创建箱线图的坐标轴
    fig, ax1 = plt.subplots(figsize=(8, 6))
    # 创建柱状图的坐标轴
    ax2 = ax1.twinx()
    # 使用sorted函数和lambda表达式按值排序字典的键  
    order = ['IO', 'CoT', 'ToT', 'GoT', 'DGoT']
    # 使用sorted函数和lambda表达式按照特定顺序排序字典的键
    sorted_key = sorted(mean_cost_dict.keys(), key=lambda x: order.index(x.split(' ')[0]))
    bar_data = []
    line_box_data_r_i = []
    line_box_data_r_1 = []
    # 输出排序后的键列表  
    for key in sorted_key:
        bar_data.append(mean_cost_dict[key])
        line_box_data_r_i.append(r_i_distribution_dict[key])
        line_box_data_r_1.append(r_1_distribution_dict[key])

    r1_mean_for_label_position = sum(mean_r_1_list)/len(mean_r_1_list)
    print(r1_mean_for_label_position)

    num_categories = len(r_1_distribution_dict)
    # 绘制柱状图
    bar_width = 0.6
    x = np.arange(num_categories) *2 + bar_width
    ax2.bar(x, bar_data, width=bar_width * 2.5, alpha=0.7, color=(127/255, 127/255, 255/255), label='Mean Cost')
    # label = ax1.text(10.30, 0.66, "the lower the better", ha='center', va='center', fontsize=12, fontweight='bold',  rotation=90, color=(127/255, 127/255, 255/255))
    ax2.set_ylabel('Mean Cost', color=(127/255, 127/255, 255/255))

    # 指定箱线图的箱体和中位线颜色
    boxprops = dict(facecolor='lightyellow', color='black')
    medianprops = dict(linestyle='', linewidth=0)

    # 绘制第一组箱线图
    ax1.boxplot(line_box_data_r_i, positions=np.arange(num_categories) * 2 + 0.2, widths=0.6, vert=True, patch_artist=True, boxprops=boxprops, medianprops=medianprops, showmeans=True)

    boxprops = dict(facecolor='white', color='black')
    # 绘制第二组箱线图
    ax1.boxplot(line_box_data_r_1, positions=np.arange(num_categories) * 2 + 1, widths=0.6, vert=True, patch_artist=True, boxprops=boxprops, medianprops=medianprops, showmeans=True)

    # label = ax1.text(-0.94, 0.64, "the higher the better", ha='center', va='center', fontsize=12, fontweight='bold',  rotation=90)
    ax1.set_ylabel('Socre')

    xtick_labels = []
    # 设置x轴刻度标签
    for key in sorted_key:
        xtick_labels.append(key.split(" ")[0])

    ax1.set_xticks(np.arange(num_categories) * 2 + bar_width)
    ax1.set_xticklabels(xtick_labels)
    # 设置y轴刻度颜色
    ax2.tick_params(axis='y', colors=(127/255, 127/255, 255/255))  # 这里将y轴刻度颜色设置为蓝色
    # 显示图例
    ax2.legend()
    # 去除白边
    plt.tight_layout()

    plt.savefig(os.path.join(folder_path, 'r_1_r_i_distribution_and_mean_cost_figure.png'))
    plt.show()

def draw_double_line_box_bar_figure(r_1_distribution_dict, r_i_distribution_dict, mean_r_1_list, mean_cost_dict, folder_path):
    # 创建箱线图的坐标轴
    fig, ax1 = plt.subplots(figsize=(8, 6))
    # 创建柱状图的坐标轴
    ax2 = ax1.twinx()
    # 使用sorted函数和lambda表达式按值排序字典的键  
    sorted_key = sorted(mean_cost_dict, key=lambda k: mean_cost_dict[k], reverse=False)
    bar_data = []
    line_box_data_r_i = []
    line_box_data_r_1 = []
    # 输出排序后的键列表  
    for key in sorted_key:
        bar_data.append(mean_cost_dict[key])
        line_box_data_r_i.append(r_i_distribution_dict[key])
        line_box_data_r_1.append(r_1_distribution_dict[key])

    r1_mean_for_label_position = sum(mean_r_1_list)/len(mean_r_1_list)
    print(r1_mean_for_label_position)

    num_categories = len(r_1_distribution_dict)
    # 绘制柱状图
    bar_width = 0.6
    x = np.arange(num_categories) *2 + bar_width
    ax2.bar(x, bar_data, width=bar_width * 2.5, alpha=0.7, color=(127/255, 127/255, 255/255), label='Mean Cost')
    # label = ax1.text(8.12, 0.48, "the lower the better", ha='center', va='center', fontsize=12, fontweight='bold',  rotation=90, color=(127/255, 127/255, 255/255))
    ax2.set_ylabel('Mean Cost', color=(127/255, 127/255, 255/255))

    # 指定箱线图的箱体和中位线颜色
    boxprops = dict(facecolor='lightyellow', color='black')
    medianprops = dict(linestyle='', linewidth=0)

    # 绘制第一组箱线图
    ax1.boxplot(line_box_data_r_i, positions=np.arange(num_categories) * 2 + 0.2, widths=0.6, vert=True, patch_artist=True, boxprops=boxprops, medianprops=medianprops, showmeans=True)

    boxprops = dict(facecolor='white', color='black')
    # 绘制第二组箱线图
    ax1.boxplot(line_box_data_r_1, positions=np.arange(num_categories) * 2 + 1, widths=0.6, vert=True, patch_artist=True, boxprops=boxprops, medianprops=medianprops, showmeans=True)

    # label = ax1.text(-0.79, 0.46, "the higher the better", ha='center', va='center', fontsize=12, fontweight='bold',  rotation=90)
    ax1.set_ylabel('Socre')

    xtick_labels = []
    # 设置x轴刻度标签
    for key in sorted_key:
        xtick_labels.append("k = " + key.split(" ")[2])

    ax1.set_xticks(np.arange(num_categories) * 2 + bar_width)
    ax1.set_xticklabels(xtick_labels)
    # 设置y轴刻度颜色
    ax2.tick_params(axis='y', colors=(127/255, 127/255, 255/255))  # 这里将y轴刻度颜色设置为蓝色
    # 显示图例
    ax2.legend()
    # 去除白边
    plt.tight_layout()

    plt.savefig(os.path.join(folder_path, 'r_1_r_i_distribution_and_mean_cost_figure.png'))
    plt.show()

def cal_gumbel(mean, var, p):
    # mu = mean
    beta = np.sqrt(6*var)/np.pi
    mu = mean - 0.577215*beta
    thresh = mu - beta * math.log(-math.log(p))
    return thresh

def cal_transformation_score(folder_path):
    generate_score = []
    aggregate_score = []
    improve_score = []
    max_generate_score = []
    max_aggregate_score = []
    max_improve_score = []

    for foldername, subfolders, filenames in os.walk(folder_path):
        # 打印文件
        for filename in filenames:
            #print('文件：' + filename)
            path = folder_path + filename
        
            with open(path, 'r') as json_file:
                data = json.load(json_file)

            try:
                max_generate_score.append(max(data[1]["scores"]))
                max_aggregate_score.append(max(data[4]["scores"]))
                max_improve_score.append(max(data[7]["scores"]))
                for i in data[1]["scores"]:
                    generate_score.append(i)
                for i in data[4]["scores"]:
                    aggregate_score.append(i)
                for i in data[7]["scores"]:
                    improve_score.append(i)
                    
                avliable_num += 1
            except:
                pass
                
    mean_generate = sum(generate_score) / len(generate_score)
    mean_aggregate = sum(aggregate_score) / len(aggregate_score)
    mean_improve = sum(improve_score) / len(improve_score)
    max_mean_generate = sum(max_generate_score) / len(max_generate_score)
    max_mean_aggregate = sum(max_aggregate_score) / len(max_aggregate_score)
    max_mean_improve = sum(max_improve_score) / len(max_improve_score)
    var_max_generate = statistics.variance(max_generate_score)
    generate_gumbel_25 = cal_gumbel(max_mean_generate, var_max_generate, 0.25)
    generate_gumbel_50 = cal_gumbel(max_mean_generate, var_max_generate, 0.5)
    generate_gumbel_75 = cal_gumbel(max_mean_generate, var_max_generate, 0.75)
    var_max_aggregate = statistics.variance(max_aggregate_score)
    aggregate_gumbel_25 = cal_gumbel(max_mean_aggregate, var_max_aggregate, 0.25)
    aggregate_gumbel_50 = cal_gumbel(max_mean_aggregate, var_max_aggregate, 0.5)
    aggregate_gumbel_75 = cal_gumbel(max_mean_aggregate, var_max_aggregate, 0.75)
    var_max_improve = statistics.variance(max_improve_score)
    improve_gumbel_25 = cal_gumbel(max_mean_improve, var_max_improve, 0.25)
    improve_gumbel_50 = cal_gumbel(max_mean_improve, var_max_improve, 0.5)
    improve_gumbel_75 = cal_gumbel(max_mean_improve, var_max_improve, 0.75)

    file_name = os.path.join(folder_path.split('/')[1], folder_path.split('/')[2], folder_path.split('/')[3] + "_transformation_score_overview.txt")
    # save transformation score to .txt
    data_to_write = "mean_generate mean_aggregate mean_improve max_mean_generate max_mean_aggregate max_mean_improve\n"
    data_to_write += f"{mean_generate} {mean_aggregate} {mean_improve} {max_mean_generate} {max_mean_aggregate} {max_mean_improve}\n"
    data_to_write += "generate gumbel_25 gumbel_50 gumbel_75\n"
    data_to_write += f"{generate_gumbel_25} {generate_gumbel_50} {generate_gumbel_75}\n"
    data_to_write += "aggregate gumbel_25 gumbel_50 gumbel_75\n"
    data_to_write += f"{aggregate_gumbel_25} {aggregate_gumbel_50} {aggregate_gumbel_75}\n"
    data_to_write += "improve gumbel_25 gumbel_50 gumbel_75\n"
    data_to_write += f"{improve_gumbel_25} {improve_gumbel_50} {improve_gumbel_75}\n"

    # write in txt
    with open(file_name, "w") as file:
        file.write(data_to_write)

    return generate_score, aggregate_score, improve_score, max_generate_score, max_aggregate_score, max_improve_score, [generate_gumbel_25, generate_gumbel_50, generate_gumbel_75]

def draw_transformation_score_figure(generate_score, aggregate_score, improve_score, mode, folder_path, gumbel_thresh=None):
    sns.set_theme(style="whitegrid")

    # 自定义三类数据
    data_category1 = generate_score
    data_category2 = aggregate_score
    data_category3 = improve_score

    mean_g = sum(data_category1)/len(data_category1)
    mean_a = sum(data_category2)/len(data_category2)
    mean_i = sum(data_category3)/len(data_category3)


    # 构造数据框
    import pandas as pd
    data = pd.DataFrame({
        'Score': ['Generate'] * len(data_category1) +
                    ['Aggregate'] * len(data_category2) +
                    ['Improve'] * len(data_category3),
        'Value': data_category1 + data_category2 + data_category3
    })

    # 创建分布图
    plt.figure(figsize=(7, 4))
    sns.histplot(data=data, x='Value', stat='percent', hue='Score', multiple="layer", kde=True, palette='Set2')

    plt.ylabel('Frequency (in percentage)')
    plt.xlim(0, 1)
    if mode == "max":
        plt.xlabel('Max Score in One Transformation')
        # 获取y轴最大值
        ymin, ymax = plt.ylim()
        var_g = statistics.variance(generate_score)
        var_a = statistics.variance(aggregate_score)
        var_i = statistics.variance(improve_score)

        beta = np.sqrt(6*var_g)/np.pi  # 尺度参数
        mu = mean_g - 0.577215*beta # 位置参数

        # 生成一些数据点
        x_1 = np.linspace(gumbel_r.ppf(0.01, loc=mu, scale=beta), gumbel_r.ppf(0.99, loc=mu, scale=beta), 1000)
        # 计算概率密度函数值
        pdf_1 = gumbel_r.pdf(x_1, loc=mu, scale=beta)
        #pdf_1 = pdf_1 / max(pdf_1) * ymax * 0.8

        beta = np.sqrt(6*var_a)/np.pi  # 尺度参数
        mu = mean_a - 0.577215*beta # 位置参数
        # 生成一些数据点
        x_2 = np.linspace(gumbel_r.ppf(0.01, loc=mu, scale=beta), gumbel_r.ppf(0.99, loc=mu, scale=beta), 1000)
        # 计算概率密度函数值
        pdf_2 = gumbel_r.pdf(x_2, loc=mu, scale=beta)
        #pdf_2 = pdf_2 / max(pdf_2) * ymax * 0.8

        beta = np.sqrt(6*var_i)/np.pi  # 尺度参数
        mu = mean_i - 0.577215*beta # 位置参数
        # 生成一些数据点
        x_3 = np.linspace(gumbel_r.ppf(0.01, loc=mu, scale=beta), gumbel_r.ppf(0.99, loc=mu, scale=beta), 1000)
        # 计算概率密度函数值
        pdf_3 = gumbel_r.pdf(x_3, loc=mu, scale=beta)
        #pdf_3 = pdf_3 / max(pdf_3) * ymax * 0.8

        plt.plot(x_1, pdf_1, label='Gumbel PDF max g', linestyle='--', color=(102/255, 194/255, 165/255))
        plt.plot(x_2, pdf_2, label='Gumbel PDF max a', linestyle='--', color=(252/255, 141/255, 98/255))
        plt.plot(x_3, pdf_3, label='Gumbel PDF max i', linestyle='--', color=(141/255, 160/255, 203/255))

        plt.axvline(x=gumbel_thresh[0], color=(102/255, 194/255, 165/255), linestyle='--', linewidth=2)
        plt.text(gumbel_thresh[0], 1, 'Gumbel 25% Thresh = '+ "{:.2f}".format(gumbel_thresh[0]), horizontalalignment='center', fontsize=6, rotation='vertical', color="green")
        plt.axvline(x=gumbel_thresh[1], color=(102/255, 194/255, 165/255), linestyle='--', linewidth=2)
        plt.text(gumbel_thresh[1], 1, 'Gumbel 50% Thresh = '+ "{:.2f}".format(gumbel_thresh[1]), horizontalalignment='center', fontsize=6, rotation='vertical', color="green")
        plt.axvline(x=gumbel_thresh[2], color=(102/255, 194/255, 165/255), linestyle='--', linewidth=2)
        plt.text(gumbel_thresh[2], 1, 'Gumbel 75% Thresh = '+ "{:.2f}".format(gumbel_thresh[2]), horizontalalignment='center', fontsize=6, rotation='vertical', color="green")
        plt.title('Distribution of Three Transformation\'s Max Score')
        # 去除白边
        plt.tight_layout()
        plt.savefig(os.path.join(folder_path.split('/')[1], folder_path.split('/')[2], folder_path.split('/')[3] + '_transformation_max_score_distribution.png'))
        print("Save to " + os.path.join(folder_path.split('/')[1], folder_path.split('/')[2], folder_path.split('/')[3] + '_transformation_max_score_distribution.png'))
    else:
        plt.xlabel('Score')
        plt.axvline(x=mean_g, color=(102/255, 194/255, 165/255), linestyle='-', linewidth=2)
        plt.text(mean_g, 0.6, 'Simple Mean Thresh = '+ "{:.2f}".format(mean_g), horizontalalignment='center', fontsize=6, rotation='vertical', color="green")
        plt.axvline(x=mean_a, color=(252/255, 141/255, 98/255), linestyle='-', linewidth=2)
        plt.axvline(x=mean_i, color=(141/255, 160/255, 203/255), linestyle='--', linewidth=2)
        plt.title('Distribution of Three Transformation\'s Score')
        # 去除白边
        plt.tight_layout()
        plt.savefig(os.path.join(folder_path.split('/')[1], folder_path.split('/')[2], folder_path.split('/')[3] + '_transformation_score_distribution.png'))
        print("Save to " + os.path.join(folder_path.split('/')[1], folder_path.split('/')[2], folder_path.split('/')[3] + '_transformation_score_distribution.png'))
    # 显示图形
    plt.show()

def cal_and_draw_transformation_score(folder_path):
    generate_score, aggregate_score, improve_score, max_generate_score, max_aggregate_score, max_improve_score, gumbel_thresh = cal_transformation_score(folder_path)
    draw_transformation_score_figure(generate_score, aggregate_score, improve_score, "normal", folder_path)
    draw_transformation_score_figure(max_generate_score, max_aggregate_score, max_improve_score, "max", folder_path, gumbel_thresh)
