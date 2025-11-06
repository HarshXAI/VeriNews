"""
Week 5-6: Interactive visualization notebook for comprehensive analysis
Combines all analyses into an interactive dashboard
"""

import argparse
import os
import sys
from pathlib import Path
import json

import torch
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import networkx as nx

sys.path.insert(0, str(Path(__file__).parent.parent))


def create_performance_dashboard(metrics_path):
    """Create interactive performance metrics dashboard"""
    with open(metrics_path) as f:
        metrics = json.load(f)
    
    # Training history
    if 'history' in metrics:
        history = metrics['history']
        epochs = list(range(1, len(history['train_loss']) + 1))
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Training Loss', 'F1 Score', 'Accuracy', 'Learning Progress'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Loss
        fig.add_trace(
            go.Scatter(x=epochs, y=history['train_loss'], name='Train Loss', 
                      line=dict(color='blue')),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=epochs, y=history['val_loss'], name='Val Loss',
                      line=dict(color='red', dash='dash')),
            row=1, col=1
        )
        
        # F1
        fig.add_trace(
            go.Scatter(x=epochs, y=history['train_f1'], name='Train F1',
                      line=dict(color='green')),
            row=1, col=2
        )
        fig.add_trace(
            go.Scatter(x=epochs, y=history['val_f1'], name='Val F1',
                      line=dict(color='orange', dash='dash')),
            row=1, col=2
        )
        
        # Accuracy
        fig.add_trace(
            go.Scatter(x=epochs, y=history['train_acc'], name='Train Acc',
                      line=dict(color='purple')),
            row=2, col=1
        )
        fig.add_trace(
            go.Scatter(x=epochs, y=history['val_acc'], name='Val Acc',
                      line=dict(color='brown', dash='dash')),
            row=2, col=1
        )
        
        # Combined view
        fig.add_trace(
            go.Scatter(x=epochs, y=history['val_f1'], name='Val F1',
                      fill='tonexty', line=dict(color='rgba(255,165,0,0.5)')),
            row=2, col=2
        )
        fig.add_trace(
            go.Scatter(x=epochs, y=history['val_acc'], name='Val Acc',
                      line=dict(color='rgba(75,0,130,0.5)')),
            row=2, col=2
        )
        
        fig.update_layout(height=800, title_text="Training Performance Dashboard")
        
        return fig
    
    return None


def create_attention_dashboard(attention_stats_path):
    """Create interactive attention analysis dashboard"""
    with open(attention_stats_path) as f:
        stats = json.load(f)
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Attention by Connection Type',
            'Attention Distribution',
            'Layer Comparison',
            'Edge Type Composition'
        ),
        specs=[[{"type": "bar"}, {"type": "box"}],
               [{"type": "scatter"}, {"type": "pie"}]]
    )
    
    # Extract data from all layers
    layers_data = []
    for layer_key in ['layer_1', 'layer_2', 'layer_3']:
        if layer_key in stats:
            layers_data.append(stats[layer_key])
    
    if not layers_data:
        return None
    
    # 1. Attention by type (stacked bar)
    connection_types = list(layers_data[0]['by_type'].keys())
    for i, layer in enumerate(layers_data, 1):
        means = [layer['by_type'][t]['mean'] for t in connection_types]
        fig.add_trace(
            go.Bar(name=f'Layer {i}', x=connection_types, y=means),
            row=1, col=1
        )
    
    # 2. Box plot of distributions
    for i, layer in enumerate(layers_data, 1):
        for conn_type in connection_types:
            if 'values' in layer['by_type'][conn_type]:
                vals = layer['by_type'][conn_type]['values'][:1000]  # Limit for performance
                fig.add_trace(
                    go.Box(y=vals, name=f'L{i} {conn_type}', showlegend=False),
                    row=1, col=2
                )
    
    # 3. Layer comparison scatter
    layer_means = []
    for i, layer in enumerate(layers_data, 1):
        means = [layer['by_type'][t]['mean'] for t in connection_types]
        fig.add_trace(
            go.Scatter(x=connection_types, y=means, mode='lines+markers',
                      name=f'Layer {i}', line=dict(width=3)),
            row=2, col=1
        )
    
    # 4. Edge type pie chart (from last layer)
    last_layer = layers_data[-1]
    edge_counts = {t: d['count'] for t, d in last_layer['by_type'].items()}
    fig.add_trace(
        go.Pie(labels=list(edge_counts.keys()), values=list(edge_counts.values()),
               hole=0.3),
        row=2, col=2
    )
    
    fig.update_layout(height=900, title_text="Attention Mechanism Analysis Dashboard")
    
    return fig


def create_node_importance_dashboard(node_analysis_path):
    """Create interactive node importance dashboard"""
    df = pd.read_csv(node_analysis_path)
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Node Importance vs Confidence',
            'Degree Distribution',
            'Class Distribution',
            'Top Nodes Network'
        ),
        specs=[[{"type": "scatter"}, {"type": "histogram"}],
               [{"type": "bar"}, {"type": "scatter"}]]
    )
    
    # 1. Importance vs Confidence scatter
    colors = df['true_label'].map({0: 'red', 1: 'green'})
    fig.add_trace(
        go.Scatter(
            x=df['pagerank'],
            y=df['confidence'],
            mode='markers',
            marker=dict(
                size=df['degree'] * 2,
                color=df['true_label'],
                colorscale=['red', 'green'],
                showscale=True,
                colorbar=dict(title="True Label")
            ),
            text=df['node_id'],
            hovertemplate='<b>Node %{text}</b><br>PageRank: %{x:.4f}<br>Confidence: %{y:.2f}<extra></extra>'
        ),
        row=1, col=1
    )
    
    # 2. Degree distribution
    fig.add_trace(
        go.Histogram(x=df['degree'], nbinsx=30, name='Degree'),
        row=1, col=2
    )
    
    # 3. Class distribution bar
    class_counts = df['predicted_label'].value_counts()
    fig.add_trace(
        go.Bar(x=['Fake', 'Real'], y=[class_counts.get(0, 0), class_counts.get(1, 0)],
               marker_color=['red', 'green']),
        row=2, col=1
    )
    
    # 4. Top nodes (simplified network view)
    top_nodes = df.nlargest(20, 'pagerank')
    fig.add_trace(
        go.Scatter(
            x=top_nodes['betweenness'],
            y=top_nodes['pagerank'],
            mode='markers+text',
            marker=dict(size=top_nodes['degree']*3, color=top_nodes['true_label'],
                       colorscale=['red', 'green']),
            text=top_nodes['node_id'],
            textposition='top center',
            hovertemplate='<b>Node %{text}</b><br>Betweenness: %{x:.4f}<br>PageRank: %{y:.4f}<extra></extra>'
        ),
        row=2, col=2
    )
    
    fig.update_xaxes(title_text="PageRank", row=1, col=1)
    fig.update_yaxes(title_text="Confidence", row=1, col=1)
    fig.update_xaxes(title_text="Degree", row=1, col=2)
    fig.update_xaxes(title_text="Betweenness Centrality", row=2, col=2)
    fig.update_yaxes(title_text="PageRank", row=2, col=2)
    
    fig.update_layout(height=900, title_text="Node Importance Analysis Dashboard", showlegend=False)
    
    return fig


def create_explainability_dashboard(explainability_path):
    """Create explainability insights dashboard"""
    with open(explainability_path) as f:
        data = json.load(f)
    
    explanations = data['explanations_summary']
    df = pd.DataFrame(explanations)
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Confidence Distribution',
            'Predictions by Class',
            'Neighbor Count Distribution',
            'Confidence vs Neighbors'
        ),
        specs=[[{"type": "histogram"}, {"type": "pie"}],
               [{"type": "histogram"}, {"type": "scatter"}]]
    )
    
    # 1. Confidence histogram
    fig.add_trace(
        go.Histogram(x=df['confidence'], nbinsx=20, name='Confidence'),
        row=1, col=1
    )
    
    # 2. Predictions pie
    pred_counts = df['predicted_class'].value_counts()
    fig.add_trace(
        go.Pie(labels=['Fake', 'Real'], 
               values=[pred_counts.get(0, 0), pred_counts.get(1, 0)],
               marker_colors=['red', 'green']),
        row=1, col=2
    )
    
    # 3. Neighbor count histogram
    fig.add_trace(
        go.Histogram(x=df['num_neighbors'], nbinsx=15, name='Neighbors'),
        row=2, col=1
    )
    
    # 4. Confidence vs Neighbors scatter
    fig.add_trace(
        go.Scatter(
            x=df['num_neighbors'],
            y=df['confidence'],
            mode='markers',
            marker=dict(size=8, color=df['predicted_class'], 
                       colorscale=['red', 'green']),
            hovertemplate='Node %{text}<br>Neighbors: %{x}<br>Confidence: %{y:.2f}<extra></extra>',
            text=df['node_idx']
        ),
        row=2, col=2
    )
    
    fig.update_layout(height=800, title_text="Model Explainability Dashboard", showlegend=False)
    
    return fig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiments-dir", type=str, default="experiments")
    parser.add_argument("--output-path", type=str, default="experiments/interactive_dashboard.html")
    
    args = parser.parse_args()
    
    print("="*70)
    print("📊 CREATING INTERACTIVE DASHBOARD")
    print("="*70)
    
    # Collect all available data
    dashboards = []
    
    # 1. Performance metrics
    metrics_path = os.path.join(args.experiments_dir, "models/training_metrics.json")
    if os.path.exists(metrics_path):
        print("\n  ✓ Loading training metrics...")
        fig = create_performance_dashboard(metrics_path)
        if fig:
            dashboards.append(("Training Performance", fig))
    
    # 2. Attention analysis
    attention_path = os.path.join(args.experiments_dir, "attention_analysis/attention_stats.json")
    if os.path.exists(attention_path):
        print("  ✓ Loading attention analysis...")
        fig = create_attention_dashboard(attention_path)
        if fig:
            dashboards.append(("Attention Analysis", fig))
    
    # 3. Node importance
    node_path = os.path.join(args.experiments_dir, "node_analysis/node_importance.csv")
    if os.path.exists(node_path):
        print("  ✓ Loading node importance...")
        fig = create_node_importance_dashboard(node_path)
        if fig:
            dashboards.append(("Node Importance", fig))
    
    # 4. Explainability
    explain_path = os.path.join(args.experiments_dir, "explainability/explainability_summary.json")
    if os.path.exists(explain_path):
        print("  ✓ Loading explainability analysis...")
        fig = create_explainability_dashboard(explain_path)
        if fig:
            dashboards.append(("Explainability", fig))
    
    # Create combined HTML
    print(f"\n📝 Generating interactive HTML dashboard...")
    
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Fake News Detection - Interactive Dashboard</title>
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        <style>
            body {
                font-family: Arial, sans-serif;
                margin: 0;
                padding: 20px;
                background-color: #f5f5f5;
            }
            .header {
                text-align: center;
                padding: 20px;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                border-radius: 10px;
                margin-bottom: 30px;
            }
            .dashboard {
                background: white;
                padding: 20px;
                margin-bottom: 30px;
                border-radius: 10px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            }
            .dashboard h2 {
                color: #333;
                border-bottom: 3px solid #667eea;
                padding-bottom: 10px;
            }
            .tabs {
                display: flex;
                gap: 10px;
                margin-bottom: 20px;
            }
            .tab {
                padding: 10px 20px;
                background: #e0e0e0;
                border: none;
                border-radius: 5px;
                cursor: pointer;
                font-size: 16px;
            }
            .tab.active {
                background: #667eea;
                color: white;
            }
            .tab-content {
                display: none;
            }
            .tab-content.active {
                display: block;
            }
        </style>
    </head>
    <body>
        <div class="header">
            <h1>🔍 Fake News Detection with Graph Attention Networks</h1>
            <p>Interactive Analysis Dashboard</p>
        </div>
        
        <div class="tabs">
    """
    
    for i, (title, _) in enumerate(dashboards):
        active = "active" if i == 0 else ""
        html_content += f'<button class="tab {active}" onclick="showTab({i})">{title}</button>\n'
    
    html_content += """
        </div>
    """
    
    for i, (title, fig) in enumerate(dashboards):
        active = "active" if i == 0 else ""
        html_content += f"""
        <div class="tab-content {active}" id="tab{i}">
            <div class="dashboard">
                <h2>{title}</h2>
                <div id="plot{i}"></div>
            </div>
        </div>
        """
    
    html_content += """
        <script>
            function showTab(index) {
                // Hide all tabs
                document.querySelectorAll('.tab-content').forEach(el => {
                    el.classList.remove('active');
                });
                document.querySelectorAll('.tab').forEach(el => {
                    el.classList.remove('active');
                });
                
                // Show selected tab
                document.getElementById('tab' + index).classList.add('active');
                document.querySelectorAll('.tab')[index].classList.add('active');
            }
            
            // Initialize plots
    """
    
    for i, (_, fig) in enumerate(dashboards):
        plot_json = fig.to_json()
        html_content += f"""
            Plotly.newPlot('plot{i}', {plot_json});
        """
    
    html_content += """
        </script>
    </body>
    </html>
    """
    
    # Save
    with open(args.output_path, 'w') as f:
        f.write(html_content)
    
    print(f"  ✓ Saved to: {args.output_path}")
    
    print("\n" + "="*70)
    print("✅ INTERACTIVE DASHBOARD COMPLETE!")
    print("="*70)
    print(f"\nDashboard includes {len(dashboards)} sections:")
    for title, _ in dashboards:
        print(f"  • {title}")
    
    print(f"\nOpen in browser:")
    print(f"  open {args.output_path}")


if __name__ == "__main__":
    main()
