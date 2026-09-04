# -*- coding: utf-8 -*-
"""
Created on Mon Apr 28 21:21:01 2025

@author: Lenovo
"""
import pandas as pd
import os
import argparse
import sys


def fasta2csv(fasta_path, output_path=None):
    """
    将FASTA文件转换为CSV文件
    
    参数:
    fasta_path: 输入的FASTA文件路径
    output_path: 输出的CSV文件路径（可选）
    """
    # 检查输入文件是否存在
    if not os.path.exists(fasta_path):
        print(f"错误: 文件 '{fasta_path}' 不存在")
        return None, None
    
    # 如果未指定输出路径，使用默认文件名
    if output_path is None:
        output_path = os.path.join(os.path.dirname(fasta_path), 'test_seq.csv')
    
    with open(fasta_path, 'r') as f:
        content = f.read()
        seq = content.split('>')
    del(seq[0])

    df = pd.DataFrame(columns=['ID', 'SEQUENCE', 'SEQUENCE_space','task'])

    seq_name =[]
    for i in range(len(seq)):
        a = seq[i].split('\n')
        b = a[0].split('-')
        df.loc[i] = [b[0], a[1].strip(), " ".join(a[1].strip()),b[1]]
        seq_name.append(a[0])
    
    # 保存到指定的输出路径
    df.to_csv(output_path, index=False)
    print(f"转换完成！结果已保存到: {output_path}")
    print(f"共处理 {len(seq_name)} 条序列")
    
    return seq_name, output_path


def main():
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(
        description='将FASTA文件转换为CSV格式',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
使用示例:
  python fasta2csv.py input.fasta
  python fasta2csv.py input.fasta -o output.csv
  python fasta2csv.py -i input.fasta -o output.csv
        '''
    )
    
    # 添加参数
    parser.add_argument(
        'input',
        nargs='?',  # 使输入文件参数为可选（当使用-i选项时）
        help='输入的FASTA文件路径'
    )
    
    parser.add_argument(
        '-i', '--input',
        dest='input_file',
        help='输入的FASTA文件路径（可选）'
    )
    
    parser.add_argument(
        '-o', '--output',
        dest='output_file',
        help='输出的CSV文件路径（可选，默认为test_seq.csv）'
    )
    
    parser.add_argument(
        '-v', '--version',
        action='version',
        version='%(prog)s 1.0'
    )
    
    # 解析参数
    args = parser.parse_args()
    
    # 确定输入文件路径
    input_path = args.input_file if args.input_file else args.input
    
    # 检查是否提供了输入文件
    if not input_path:
        parser.print_help()
        print("\n错误: 需要指定输入文件")
        print("请使用以下方式之一指定输入文件:")
        print("  python fasta2csv.py input.fasta")
        print("  或")
        print("  python fasta2csv.py -i input.fasta")
        sys.exit(1)
    
    # 调用转换函数
    try:
        seq_name, csv_path = fasta2csv(input_path, args.output_file)
        if seq_name:
            print("\n前5条序列ID:")
            for i, name in enumerate(seq_name[:5]):
                print(f"  {i+1}. {name}")
            if len(seq_name) > 5:
                print(f"  ... 还有 {len(seq_name)-5} 条序列")
    except Exception as e:
        print(f"转换过程中出现错误: {str(e)}")
        sys.exit(1)


if __name__ == '__main__':
    main()