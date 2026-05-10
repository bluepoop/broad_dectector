import json
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

def load_results(json_path):
    """加载处理结果"""
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def results_to_dataframe(results):
    """将结果转换为pandas DataFrame"""
    data = []
    for well in results['wells']:
        row_data = {
            'well_id': well['well_id'],
            'row': well['row'],
            'col': well['col'],
            'r': well['rgb']['r'],
            'g': well['rgb']['g'],
            'b': well['rgb']['b'],
            'confidence': well['confidence']
        }
        data.append(row_data)

    return pd.DataFrame(data)

def visualize_rgb_heatmap(df, save_path=None):
    """可视化RGB热图"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # 创建透视表
    for idx, channel in enumerate(['r', 'g', 'b']):
        pivot_table = df.pivot(index='row', columns='col', values=channel)

        sns.heatmap(pivot_table, annot=True, cmap='viridis', ax=axes[idx],
                   cbar_kws={'label': f'{channel.upper()} value'})
        axes[idx].set_title(f'{channel.upper()} Channel Heatmap')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"热图已保存到: {save_path}")

    plt.show()

def analyze_color_distribution(df):
    """分析颜色分布"""
    print("=== 颜色分布分析 ===")
    print(f"总孔数: {len(df)}")
    print("\nRGB统计:")
    print(df[['r', 'g', 'b']].describe())

    # 计算颜色强度
    df['intensity'] = (df['r'] + df['g'] + df['b']) / 3
    df['dominant_color'] = df[['r', 'g', 'b']].idxmax(axis=1)

    print(f"\n平均颜色强度: {df['intensity'].mean():.2f}")
    print(f"颜色强度范围: {df['intensity'].min():.0f} - {df['intensity'].max():.0f}")

    print("\n主要颜色分布:")
    print(df['dominant_color'].value_counts())

    return df

def export_to_excel(df, output_path):
    """导出到Excel"""
    with pd.ExcelWriter(output_path) as writer:
        # 主要数据
        df.to_excel(writer, sheet_name='RGB_Data', index=False)

        # 统计信息
        stats_df = df[['r', 'g', 'b', 'intensity']].describe()
        stats_df.to_excel(writer, sheet_name='Statistics')

        # 按行列分组的数据
        pivot_r = df.pivot(index='row', columns='col', values='r')
        pivot_g = df.pivot(index='row', columns='col', values='g')
        pivot_b = df.pivot(index='row', columns='col', values='b')

        pivot_r.to_excel(writer, sheet_name='R_Channel_Matrix')
        pivot_g.to_excel(writer, sheet_name='G_Channel_Matrix')
        pivot_b.to_excel(writer, sheet_name='B_Channel_Matrix')

    print(f"Excel文件已保存到: {output_path}")

def batch_analyze():
    """批量分析所有结果"""
    results_dir = Path("results")
    json_files = list(results_dir.glob("*_results.json"))

    if not json_files:
        print("未找到结果文件")
        return

    all_data = []

    for json_file in json_files:
        results = load_results(json_file)
        df = results_to_dataframe(results)
        df['image_name'] = json_file.stem.replace('_results', '')
        all_data.append(df)

    # 合并所有数据
    combined_df = pd.concat(all_data, ignore_index=True)

    # 分析
    analyzed_df = analyze_color_distribution(combined_df)

    # 可视化
    visualize_rgb_heatmap(analyzed_df, 'results/rgb_heatmap.png')

    # 导出Excel
    export_to_excel(analyzed_df, 'results/rgb_analysis.xlsx')

    return analyzed_df

if __name__ == "__main__":
    batch_analyze()