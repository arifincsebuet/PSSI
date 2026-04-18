import matplotlib.pyplot as plt
import numpy as np
import os

# Create an output directory for the graphs
out_dir = "plots"
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

# Style settings for paper graphs
plt.style.use('seaborn-v0_8-whitegrid')
colors = ['#4A90E2', '#50E3C2', '#F5A623', '#D0021B', '#9013FE']

def autolabel(rects, ax, xpos='center'):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=10)

def plot_latency():
    systems = ['Elasticsearch', 'Lucene', 'Algolia', 'Typesense', 'PSSI']
    avg_latency = [420, 390, 310, 280, 190]
    p95_latency = [780, 720, 600, 540, 320]
    p99_latency = [1100, 1020, 900, 820, 480]

    x = np.arange(len(systems))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar(x - width, avg_latency, width, label='Avg (ms)', color='#4A90E2')
    rects2 = ax.bar(x, p95_latency, width, label='P95 (ms)', color='#F5A623')
    rects3 = ax.bar(x + width, p99_latency, width, label='P99 (ms)', color='#D0021B')

    ax.set_ylabel('Latency (ms)', fontsize=12)
    ax.set_title('Query Latency Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(systems, fontsize=11)
    ax.legend()

    autolabel(rects1, ax)
    autolabel(rects2, ax)
    autolabel(rects3, ax)

    fig.tight_layout()
    plt.savefig(f'{out_dir}/1_query_latency.png', dpi=300)
    plt.close()

def plot_memory():
    systems = ['Elasticsearch', 'Lucene', 'Typesense', 'PSSI']
    memory = [8.2, 7.5, 6.8, 1.5]

    fig, ax = plt.subplots(figsize=(8, 6))
    rects = ax.bar(systems, memory, color=colors[:4], width=0.6)

    ax.set_ylabel('Memory (GB)', fontsize=12)
    ax.set_title('Memory Usage Comparison', fontsize=14, fontweight='bold')
    for i, v in enumerate(memory):
        ax.text(i, v + 0.1, str(v), color='black', ha='center', fontsize=11)

    fig.tight_layout()
    plt.savefig(f'{out_dir}/2_memory_usage.png', dpi=300)
    plt.close()

def plot_network():
    systems = ['Elasticsearch', 'Algolia', 'Typesense', 'PSSI']
    network = [45, 38, 32, 12]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.bar(systems, network, color=['#4A90E2', '#50E3C2', '#F5A623', '#9013FE'], width=0.6)

    ax.set_ylabel('Avg KB per Query', fontsize=12)
    ax.set_title('Network Cost Comparison', fontsize=14, fontweight='bold')
    for i, v in enumerate(network):
        ax.text(i, v + 0.5, str(v), color='black', ha='center', fontsize=11)

    fig.tight_layout()
    plt.savefig(f'{out_dir}/3_network_cost.png', dpi=300)
    plt.close()

def plot_accuracy():
    query_types = ['Substring', 'Fuzzy', 'Semantic']
    precision = [0.94, 0.91, 0.89]
    recall = [0.91, 0.88, 0.86]
    f1 = [0.92, 0.89, 0.87]

    x = np.arange(len(query_types))
    width = 0.25

    fig, ax = plt.subplots(figsize=(9, 6))
    rects1 = ax.bar(x - width, precision, width, label='Precision@10', color='#4A90E2')
    rects2 = ax.bar(x, recall, width, label='Recall@10', color='#50E3C2')
    rects3 = ax.bar(x + width, f1, width, label='F1-score', color='#F5A623')

    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Retrieval Accuracy (PSSI)', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(query_types, fontsize=11)
    ax.set_ylim([0.7, 1.0]) # Zoom to show differences clearly
    ax.legend(loc='lower left')

    fig.tight_layout()
    plt.savefig(f'{out_dir}/4_retrieval_accuracy.png', dpi=300)
    plt.close()

def plot_privacy():
    methods = ['Plain Index', 'Encrypted Index', 'PSSI']
    leakage = [0.82, 0.35, 0.08]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.bar(methods, leakage, color=['#D0021B', '#F5A623', '#50E3C2'], width=0.5)

    ax.set_ylabel('Leakage Probability', fontsize=12)
    ax.set_title('Privacy Leakage Evaluation', fontsize=14, fontweight='bold')
    for i, v in enumerate(leakage):
        ax.text(i, v + 0.02, str(v), color='black', ha='center', fontsize=11)

    fig.tight_layout()
    plt.savefig(f'{out_dir}/5_privacy_leakage.png', dpi=300)
    plt.close()

def plot_throughput():
    systems = ['Elasticsearch', 'Typesense', 'PSSI']
    qps = [220, 260, 410]

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(systems, qps, marker='o', markersize=10, linestyle='-', linewidth=2, color='#9013FE')

    ax.set_ylabel('Queries per second (QPS)', fontsize=12)
    ax.set_title('Throughput Comparison', fontsize=14, fontweight='bold')
    ax.set_ylim([0, 500])
    
    for i, v in enumerate(qps):
        ax.text(i, v + 15, str(v), color='black', ha='center', fontsize=11)

    fig.tight_layout()
    plt.savefig(f'{out_dir}/6_throughput.png', dpi=300)
    plt.close()

def plot_component():
    labels = 'Bloom Matching', 'Semantic Matching', 'Firestore Fetch'
    sizes = [35, 40, 25]
    colors = ['#4A90E2', '#50E3C2', '#F5A623']
    explode = (0, 0, 0.1)

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
            shadow=True, startangle=140, textprops={'fontsize': 12})
    ax.axis('equal') 

    ax.set_title('Component-Level Time Breakdown', fontsize=14, fontweight='bold')
    
    fig.tight_layout()
    plt.savefig(f'{out_dir}/7_component_breakdown.png', dpi=300)
    plt.close()

if __name__ == "__main__":
    print("Generating plots...")
    plot_latency()
    plot_memory()
    plot_network()
    plot_accuracy()
    plot_privacy()
    plot_throughput()
    plot_component()
    print(f"All plots saved in directory: {out_dir}/")
