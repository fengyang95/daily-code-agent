import os
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict
from queue import Queue
from threading import Lock
import re 

import dotenv
import argparse
from tqdm import tqdm

import langchain_core.exceptions
from langchain_openai import ChatOpenAI
from langchain.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from structure import Structure

if os.path.exists('.env'):
    dotenv.load_dotenv()
template = open("template.txt", "r").read()
system = open("system.txt", "r").read()

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True, help="jsonline data file")
    parser.add_argument("--max_workers", type=int, default=1, help="Maximum number of parallel workers")
    return parser.parse_args()

def process_single_item(chain, item: Dict, language: str) -> Dict:
    """处理单个数据项"""
    try:
        response: Structure = chain.invoke({
            "language": language,
            "content": item['summary']
        })
        item['AI'] = response.model_dump()
    except langchain_core.exceptions.OutputParserException as e:
        # 尝试从错误信息中提取 JSON 字符串并修复
        error_msg = str(e)
        if "Function Structure arguments:" in error_msg:
            try:
                # 提取 JSON 字符串
                json_str = error_msg.split("Function Structure arguments:", 1)[1].strip().split('are not valid JSON')[0].strip()
                # 预处理 LaTeX 数学符号 - 使用四个反斜杠来确保正确转义
                json_str = json_str.replace('\\', '\\\\')
                # 尝试解析修复后的 JSON
                fixed_data = json.loads(json_str)
                item['AI'] = fixed_data
                return item
            except Exception as json_e:
                print(f"Failed to fix JSON for {item['id']}: {json_e} {json_str}", file=sys.stderr)
        
        # 如果修复失败，返回错误状态
        item['AI'] = {
            "tldr": "Error",
            "motivation": "Error",
            "method": "Error",
            "result": "Error",
            "conclusion": "Error",
            "topics":"Error"
        }
    return item

def create_wechat_ai_fields(item: Dict) -> Dict:
    """为WeChat文章创建兼容的AI字段结构"""
    summary = item.get('summary', '')
    
    # 将WeChat文章内容映射到AI字段结构
    ai_fields = {
        "tldr": summary[:200] + "..." if len(summary) > 200 else summary,
        "motivation": "微信公众号文章，分享AI相关技术内容",
        "method": "基于实际应用经验的技术分享",
        "result": "提供实用的AI技术见解和案例分析",
        "conclusion": "适合了解AI技术在实际场景中的应用",
        "topic": "other topic"  # WeChat文章默认归类为其他主题
    }
    
    # 根据内容关键词调整topic
    content_lower = (item.get('title', '') + ' ' + summary).lower()
    if any(keyword in content_lower for keyword in ['代码', 'code', '编程']):
        ai_fields["topic"] = "code agent"
    elif any(keyword in content_lower for keyword in ['智能体', 'agent', '代理']):
        ai_fields["topic"] = "agent analysis"
    elif any(keyword in content_lower for keyword in ['强化学习', 'rl', 'reinforcement']):
        ai_fields["topic"] = "agentic reinforcement learning"
    elif any(keyword in content_lower for keyword in ['软件工程', 'swe', '开发']):
        ai_fields["topic"] = "swe application"
    elif any(keyword in content_lower for keyword in ['基准', 'benchmark', '评测']):
        ai_fields["topic"] = "swe benchmark"
    
    return ai_fields

def filter_by_key_words(data:List[Dict],key_words:List[str])->List[Dict]:
    def split_by_special_chars(text):
        result= re.split(r'[, -().]+', text)
        return [item for item in result if item]

    def valid_text(text)->bool:
        words=split_by_special_chars(text.lower())
        if len(set(words).intersection(set(key_words)))>0:
            return True 
        else:
            return False 

    new_data=[]
    for item in data:
        if valid_text(item['summary']) is True:
            new_data.append(item)
    return new_data 
        
         
    
def process_all_items(data: List[Dict], model_name: str, language: str, max_workers: int) -> List[Dict]:
    """并行处理所有数据项"""
    llm = ChatOpenAI(model=model_name).with_structured_output(Structure, method="function_calling")
    print('Connect to:', model_name, file=sys.stderr)
    
    prompt_template = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(system),
        HumanMessagePromptTemplate.from_template(template=template)
    ])

    chain = prompt_template | llm
    
    # 使用线程池并行处理
    processed_data = [None] * len(data)  # 预分配结果列表
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有任务
        future_to_idx = {
            executor.submit(process_single_item, chain, item, language): idx
            for idx, item in enumerate(data)
        }
        
        # 使用tqdm显示进度
        for future in tqdm(
            as_completed(future_to_idx),
            total=len(data),
            desc="Processing items"
        ):
            idx = future_to_idx[future]
            try:
                result = future.result()
                processed_data[idx] = result
            except Exception as e:
                print(f"Item at index {idx} generated an exception: {e}", file=sys.stderr)
                # 保持原始数据
                processed_data[idx] = data[idx]
    
    return processed_data

def main():
    args = parse_args()
    model_name = os.environ.get("MODEL_NAME", 'deepseek-chat')
    language = os.environ.get("LANGUAGE", 'Chinese')
    key_words=os.environ.get("KEY_WORDS",'RL,Agentic,Agent,Code,LLM,SWE')

    # 检查并删除目标文件
    target_file = args.data.replace('.jsonl', f'_AI_enhanced_{language}.jsonl')
    if os.path.exists(target_file):
        os.remove(target_file)
        print(f'Removed existing file: {target_file}', file=sys.stderr)

    # 读取数据
    data = []
    with open(args.data, "r") as f:
        for line in f:
            data.append(json.loads(line))

    # 过滤WeChat文章 - WeChat文章已经是中文内容，不需要AI总结
    arxiv_data = []
    wechat_data = []
    for item in data:
        # 检查是否为WeChat文章（通过ID前缀或categories字段）
        if item.get('id', '').startswith('wechat.') or 'wechat.article' in item.get('categories', []):
            wechat_data.append(item)
        else:
            arxiv_data.append(item)
    
    print(f'Found {len(wechat_data)} WeChat articles (skipping AI enhancement)', file=sys.stderr)
    print(f'Found {len(arxiv_data)} arXiv articles (will process with AI)', file=sys.stderr)

    # 只对arXiv文章进行关键词过滤和AI增强
    key_words=[word.lower() for word in key_words.split(',')]
    arxiv_data=filter_by_key_words(arxiv_data,key_words)
    # 去重 - 只对arXiv文章去重
    seen_ids = set()
    unique_arxiv_data = []
    for item in arxiv_data:
        if item['id'] not in seen_ids:
            seen_ids.add(item['id'])
            unique_arxiv_data.append(item)

    print('Open:', args.data, file=sys.stderr)
    print(f'Processing {len(unique_arxiv_data)} unique arXiv articles with AI', file=sys.stderr)
    
    # 只对arXiv文章进行AI增强处理
    if unique_arxiv_data:
        processed_arxiv_data = process_all_items(
            unique_arxiv_data,
            model_name,
            language,
            args.max_workers
        )
    else:
        processed_arxiv_data = []
    
    # 为WeChat文章创建AI字段，使其与arXiv文章格式兼容
    wechat_data_with_ai = []
    for item in wechat_data:
        item_copy = item.copy()
        item_copy['AI'] = create_wechat_ai_fields(item)
        wechat_data_with_ai.append(item_copy)
    
    # 合并处理后的arXiv文章和带AI字段的WeChat文章
    all_processed_data = processed_arxiv_data + wechat_data_with_ai
    
    print(f'Merged {len(processed_arxiv_data)} AI-enhanced arXiv articles with {len(wechat_data_with_ai)} WeChat articles (with AI fields)', file=sys.stderr)
    
    # 保存结果 - 现在所有文章都有AI字段，可以统一处理
    with open(target_file, "w") as f:
        for item in all_processed_data:
            # 检查AI字段中的topic，过滤掉'other topic'的文章（包括WeChat文章）
            if 'AI' in item and "topic" in item['AI'] and "other topic" in item['AI']['topic']:
                continue
            f.write(json.dumps(item) + "\n")

if __name__ == "__main__":
    main()
