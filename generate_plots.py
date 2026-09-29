import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Configure plot style
sns.set_theme(style="whitegrid")
plt.rcParams.update({'font.size': 12, 'figure.autolayout': True})

def create_plots():
    json_path = "results/results.json"
    output_dir = "results/plots"
    os.makedirs(output_dir, exist_ok=True)

    if not os.path.exists(json_path):
        print(f"[-] Error: {json_path} not found.")
        return

    # Load JSON Data
    with open(json_path, "r") as f:
        data = json.load(f)

    # Process Data into DataFrame
    rows = []
    for item in data:
        backbone = item.get("backbone", "Unknown")
        attention = item.get("attention", "none").upper()
        depth = item.get("depth", "single")
        model_label = f"{backbone} + {attention} ({depth})"
        
        mean_acc = item.get("mean_accuracy", 0) * 100
        
        # Calculate mean metrics across folds
        fold_metrics = item.get("fold_metrics", [])
        if fold_metrics:
            mean_f1 = sum(f.get("f1", 0) for f in fold_metrics) / len(fold_metrics)
        else:
            mean_f1 = 0

        rows.append({
            "Model": model_label,
            "Backbone": backbone,
            "Attention": attention,
            "Depth": depth,
            "Accuracy": mean_acc,
            "F1_Score": mean_f1
        })

    df = pd.DataFrame(rows)

    # -------------------------------------------------------------
    # Plot 1: Top 10 Model Accuracy Comparison (Horizontal Bar Chart)
    # -------------------------------------------------------------
    plt.figure(figsize=(10, 6))
    top10 = df.sort_values(by="Accuracy", ascending=False).head(10)
    
    ax = sns.barplot(
        data=top10, 
        x="Accuracy", 
        y="Model", 
        palette="viridis"
    )
    plt.title("Top 10 Model Configurations by Mean Accuracy (%)", fontsize=14, fontweight='bold', pad=15)
    plt.xlabel("Mean Accuracy (%)", fontsize=12)
    plt.ylabel("Model Architecture", fontsize=12)
    plt.xlim(88, 95)  # Focus on high accuracy range

    # Add text labels on bars
    for p in ax.patches:
        width = p.get_width()
        ax.annotate(f"{width:.2f}%",
                    (width + 0.1, p.get_y() + p.get_height() / 2.),
                    ha='left', va='center', fontsize=10, color='black',
                    xytext=(0, 0), textcoords='offset points')

    plot1_path = os.path.join(output_dir, "top_10_accuracy.png")
    plt.savefig(plot1_path, dpi=300)
    plt.close()
    print(f"[+] Saved: {plot1_path}")

    # -------------------------------------------------------------
    # Plot 2: Impact of Attention Mechanisms across Backbones
    # -------------------------------------------------------------
    plt.figure(figsize=(11, 6))
    ax2 = sns.barplot(
        data=df,
        x="Backbone",
        y="Accuracy",
        hue="Attention",
        ci=None,
        palette="Set2"
    )
    plt.title("Impact of Attention Mechanisms Across Model Backbones", fontsize=14, fontweight='bold', pad=15)
    plt.xlabel("Backbone Architecture", fontsize=12)
    plt.ylabel("Mean Accuracy (%)", fontsize=12)
    plt.ylim(85, 95)
    plt.legend(title="Attention Module", bbox_to_anchor=(1.05, 1), loc='upper left')

    plot2_path = os.path.join(output_dir, "attention_impact_comparison.png")
    plt.savefig(plot2_path, dpi=300)
    plt.close()
    print(f"[+] Saved: {plot2_path}")

    # -------------------------------------------------------------
    # Plot 3: Accuracy vs F1-Score Trade-off Scatter Plot
    # -------------------------------------------------------------
    plt.figure(figsize=(9, 6))
    sns.scatterplot(
        data=df,
        x="Accuracy",
        y="F1_Score",
        hue="Backbone",
        style="Attention",
        s=120,
        palette="deep"
    )
    plt.title("Accuracy vs. Weighted F1-Score Trade-off", fontsize=14, fontweight='bold', pad=15)
    plt.xlabel("Mean Accuracy (%)", fontsize=12)
    plt.ylabel("Weighted F1-Score", fontsize=12)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    plot3_path = os.path.join(output_dir, "accuracy_vs_f1_scatterplot.png")
    plt.savefig(plot3_path, dpi=300)
    plt.close()
    print(f"[+] Saved: {plot3_path}")

    print("\n[✔] All graphs generated successfully in high resolution (300 DPI)!")

if __name__ == "__main__":
    create_plots()