from bs4 import BeautifulSoup

def read_pmc(path):
    article_title_text = ''
    abstract_text = ''
    # 打开并读取HTML文件
    with open(path, 'r', encoding='utf-8') as file:
        html_content = file.read()

    # 使用Beautiful Soup解析HTML内容
    soup = BeautifulSoup(html_content, "html.parser")

    # 查找<title-group>标签
    title_group_tag = soup.find("title-group")

    # 如果找到了<title-group>标签
    if title_group_tag:
        # 查找<article-title>标签
        article_title_tag = title_group_tag.find("article-title")
        
        # 如果找到了<article-title>标签
        if article_title_tag:
            # 提取<article-title>标签中的文本内容
            article_title_text = article_title_tag.text
            # print("Article Title:", article_title_text)
        else:
            print("No <article-title> tag found inside <title-group>.")
    else:
        print("No <title-group> tag found in the HTML content.")

    # 查找<abstract>标签
    abstract_tag = soup.find("abstract")

    # 提取<abstract>标签中的文本内容
    if abstract_tag:
        abstract_text = abstract_tag.text
        # print("Abstract:", abstract_text)
    else:
        print("Abstract tag not found in the HTML content.")

    sec_tag = soup.find_all("sec")

    sec_dict = {}
    introduction_text = ''
    # 如果找到了<sec>标签
    if sec_tag:
        #print(len(sec_tag))
        # 查找<title>标签
        for tag in sec_tag:
            title_tag = tag.find("title")
            if title_tag:
                if title_tag.text in ['Results', 'Discussion', 'Materials and Methods', 'Materials and methods', 'Methods', 
                                'Background', 'Statistical analysis', 'Conclusions', 'DISCUSSION', 'RESULTS',
                                'RESULTS', 'RESULTS', 'Conclusion', 'Statistical Analysis', 'Results and Discussion',
                                'Results and discussion', 'Methodology/Principal Findings', 'Statistics', 
                                'Statistical analysis', 'Conclusions/Significance', 'RESULTS AND DISCUSSION', 
                                'Statistical analyses', 'METHODS', 'Materials', 'Statistics', 'Results:', 
                                'Data analysis', 'Results/Discussion', 'Background', 'Methods Summary',
                                'Statistical Analyses', 'Methods:', 'Concluding Remarks', 'CONCLUSIONS',
                                'Conclusion:', 'CONCLUSION', 'Data Analysis', 'Summary', 'Findings',
                                'Material and Methods', 'Statistical analyses.', 'Patients and methods',
                                'Material and methods', 'Principal Findings', 'Constructs', 'Conclusion/Significance',
                                'Author contributions', 'METHODS SUMMARY', 'Study Design', 'Background.',
                                'Patients and Methods', 'Objective', 'Author Contributions', '3. Results',
                                '3. Results', 'Statistical Methods', 'RESEARCH DESIGN AND METHODS',
                                'Outcomes', 'Implementation', 'Purpose', 'Significance', 'Experimental design',
                                '2. Materials and Methods', 'OBJECTIVE', 'Materials.', 'Methodology',
                                '2 METHODS', 'Methodology and Principal Findings', 'Data analysis.',
                                 'Analysis', '3 RESULTS', 'Method', 'discussion', 'Concluding Remarks.']:
                    sec_dict[title_tag.text] = tag.text
                if title_tag.text in ['Introduction', 'INTRODUCTION', '1. Introduction', '1 INTRODUCTION', 'introduction']:
                    introduction_text = tag.text
    else:
        print("No <sec> tag with sec-type='intro' found in the HTML content.")
    
    return article_title_text, abstract_text, introduction_text, sec_dict

def read_pm(path):
    # 打开并读取HTML文件
    with open(path, 'r', encoding='utf-8') as file:
        html_content = file.read()

    # 使用Beautiful Soup解析HTML内容
    soup = BeautifulSoup(html_content, "html.parser")

    # 查找<title>标签
    title_tags = soup.find_all("articletitle")

    # 提取<title>标签中的文本内容
    title_text = ''
    for title_tag in title_tags:
        title_text = title_tag.text

    # 查找<abstract>标签
    abstract_tag = soup.find("abstract")

    # 提取<abstract>标签中的文本内容
    abstract_text = ''
    if abstract_tag:
        abstract_text = abstract_tag.text
        # print("Abstract:", abstract_text)
    else:
        print("Abstract tag not found in the HTML content.")

    return title_text, abstract_text