# author: Jayce Ning
import argparse
from utils import process_data_for_all_tasks, draw_line_box_bar_figure, draw_double_line_box_bar_figure, cal_and_draw_transformation_score

def draw_picture(task, result_folder_path):
    r_1_distribution_dict, r_i_distribution_dict, mean_r_1_list, mean_cost_dict = process_data_for_all_tasks(result_folder_path)
    if task == 'test_prompt_length' or task == 'test_nodes_num':
        draw_line_box_bar_figure(r_1_distribution_dict, mean_r_1_list, mean_cost_dict, result_folder_path)
    if task == 'draw_intro_abstract_line_box':
        draw_double_line_box_bar_figure(r_1_distribution_dict, r_i_distribution_dict, mean_r_1_list, mean_cost_dict, result_folder_path)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Description of your program')
    parser.add_argument('--result_folder_path', type=str, default=None, help='The path of the folder where the results are stored. ./results/internlm2_got_test_nodes_num_2024-03-09_08-26-48 etc.')
    parser.add_argument('--task', type=str, default='default', help='0.default, 1.test_prompt_length, 2.test_nodes_num, 3.draw_intro_abstract_line_box, 4.cal_and_draw_transformation_score')
    args = parser.parse_args()

    if args.result_folder_path is None:
        print('Please specify the result folder path.')

    else:
        if args.task == 'default' or args.task == 'test_prompt_length' or args.task == 'test_nodes_num' or args.task == 'draw_intro_abstract_line_box':
            draw_picture(args.task, args.result_folder_path)
        elif args.task == 'cal_and_draw_transformation_score':
            cal_and_draw_transformation_score(args.result_folder_path)


