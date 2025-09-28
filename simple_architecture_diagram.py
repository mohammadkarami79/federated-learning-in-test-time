#!/usr/bin/env python3
"""
Simple Architecture Diagram Generator
Creates a clear visual representation without complex dependencies
"""

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, Rectangle
import numpy as np

def create_simple_architecture():
    """Create a simplified but comprehensive architecture diagram"""
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Colors
    colors = {
        'client': '#E3F2FD',    # Light blue
        'server': '#F3E5F5',    # Light purple  
        'mae': '#E8F5E8',       # Light green
        'diffusion': '#FFF3E0', # Light orange
        'pfeddef': '#FCE4EC',   # Light pink
        'attack': '#FFEBEE'     # Light red
    }
    
    # Title
    ax.text(7, 9.5, 'Multi-Layered Federated Defense Framework', 
            fontsize=16, fontweight='bold', ha='center')
    
    # Input
    input_rect = Rectangle((1, 8), 2, 1, facecolor=colors['attack'], 
                          edgecolor='red', linewidth=2)
    ax.add_patch(input_rect)
    ax.text(2, 8.5, 'Adversarial\nInput x', fontsize=10, fontweight='bold', ha='center')
    
    # Defense Layer 1: MAE
    mae_rect = Rectangle((4, 8), 2.5, 1, facecolor=colors['mae'], 
                        edgecolor='green', linewidth=2)
    ax.add_patch(mae_rect)
    ax.text(5.25, 8.7, 'MAE Detection', fontsize=11, fontweight='bold', ha='center')
    ax.text(5.25, 8.3, 's_det = f_mae(x)', fontsize=9, ha='center', style='italic')
    
    # Defense Layer 2: Diffusion
    diff_rect = Rectangle((7.5, 8), 2.5, 1, facecolor=colors['diffusion'], 
                         edgecolor='orange', linewidth=2)
    ax.add_patch(diff_rect)
    ax.text(8.75, 8.7, 'DiffPure', fontsize=11, fontweight='bold', ha='center')
    ax.text(8.75, 8.3, 'x_pure = DDIM(x)', fontsize=9, ha='center', style='italic')
    
    # Defense Layer 3: pFedDef
    pfed_rect = Rectangle((11, 8), 2.5, 1, facecolor=colors['pfeddef'], 
                         edgecolor='purple', linewidth=2)
    ax.add_patch(pfed_rect)
    ax.text(12.25, 8.7, 'pFedDef', fontsize=11, fontweight='bold', ha='center')
    ax.text(12.25, 8.3, 'ŷ = Σ αₖ·fₖ(x)', fontsize=9, ha='center', style='italic')
    
    # Federated Learning Layer
    server_rect = Rectangle((5, 5.5), 4, 1.5, facecolor=colors['server'], 
                           edgecolor='purple', linewidth=2)
    ax.add_patch(server_rect)
    ax.text(7, 6.7, 'Federated Server', fontsize=12, fontweight='bold', ha='center')
    ax.text(7, 6.3, 'Robust Aggregation', fontsize=10, ha='center')
    ax.text(7, 5.9, 'θ_global = Σ wᵢ·θᵢ', fontsize=9, ha='center', style='italic')
    
    # Clients
    client_positions = [(1, 3.5), (5, 3.5), (9, 3.5)]
    for i, (x, y) in enumerate(client_positions):
        client_rect = Rectangle((x, y), 3, 1.5, facecolor=colors['client'], 
                               edgecolor='blue', linewidth=1.5)
        ax.add_patch(client_rect)
        ax.text(x + 1.5, y + 1.1, f'Client {i+1}', fontsize=10, fontweight='bold', ha='center')
        ax.text(x + 1.5, y + 0.7, f'Local Dataset D_{i+1}', fontsize=8, ha='center')
        ax.text(x + 1.5, y + 0.3, 'pFedDef Training', fontsize=8, ha='center')
    
    # Pre-trained Models
    model_y = 1.5
    
    # Diffusion Model
    diff_model = Rectangle((1, model_y), 2.5, 1, facecolor=colors['diffusion'], 
                          edgecolor='orange', linewidth=1)
    ax.add_patch(diff_model)
    ax.text(2.25, model_y + 0.7, 'Diffusion Model', fontsize=9, fontweight='bold', ha='center')
    ax.text(2.25, model_y + 0.3, 'U-Net Architecture', fontsize=8, ha='center')
    
    # MAE Model  
    mae_model = Rectangle((4.5, model_y), 2.5, 1, facecolor=colors['mae'], 
                         edgecolor='green', linewidth=1)
    ax.add_patch(mae_model)
    ax.text(5.75, model_y + 0.7, 'MAE Detector', fontsize=9, fontweight='bold', ha='center')
    ax.text(5.75, model_y + 0.3, 'ViT Encoder-Decoder', fontsize=8, ha='center')
    
    # Global Model
    global_model = Rectangle((8, model_y), 2.5, 1, facecolor=colors['pfeddef'], 
                            edgecolor='purple', linewidth=1)
    ax.add_patch(global_model)
    ax.text(9.25, model_y + 0.7, 'Global Model', fontsize=9, fontweight='bold', ha='center')
    ax.text(9.25, model_y + 0.3, 'Aggregated pFedDef', fontsize=8, ha='center')
    
    # Evaluation
    eval_model = Rectangle((11, model_y), 2.5, 1, facecolor='lightgray', 
                          edgecolor='black', linewidth=1)
    ax.add_patch(eval_model)
    ax.text(12.25, model_y + 0.7, 'Evaluation', fontsize=9, fontweight='bold', ha='center')
    ax.text(12.25, model_y + 0.3, 'Metrics & Analysis', fontsize=8, ha='center')
    
    # Arrows - simplified
    # Input flow
    ax.annotate('', xy=(3.9, 8.5), xytext=(3.1, 8.5),
                arrowprops=dict(arrowstyle='->', lw=2, color='black'))
    ax.annotate('', xy=(7.4, 8.5), xytext=(6.6, 8.5),
                arrowprops=dict(arrowstyle='->', lw=2, color='black'))
    ax.annotate('', xy=(10.9, 8.5), xytext=(10.1, 8.5),
                arrowprops=dict(arrowstyle='->', lw=2, color='black'))
    
    # Output
    ax.text(13, 7.5, 'Robust\nPrediction', fontsize=10, fontweight='bold', ha='center')
    ax.annotate('', xy=(12.8, 7.8), xytext=(12.5, 8.2),
                arrowprops=dict(arrowstyle='->', lw=2, color='purple'))
    
    # Federated arrows
    for i, (x, y) in enumerate(client_positions):
        # Client to server
        ax.annotate('', xy=(6.8, 5.5), xytext=(x + 1.5, y + 1.5),
                    arrowprops=dict(arrowstyle='->', lw=1.5, color='blue', alpha=0.7))
        # Server to client  
        ax.annotate('', xy=(x + 1.5, y + 1.5), xytext=(7.2, 5.5),
                    arrowprops=dict(arrowstyle='->', lw=1.5, color='purple', alpha=0.7))
    
    # Model connections
    ax.annotate('', xy=(5.75, 7.9), xytext=(5.75, 2.6),
                arrowprops=dict(arrowstyle='->', lw=1, color='gray', alpha=0.6))
    ax.annotate('', xy=(8.75, 7.9), xytext=(2.25, 2.6),
                arrowprops=dict(arrowstyle='->', lw=1, color='gray', alpha=0.6))
    
    # Performance box
    perf_rect = Rectangle((0.5, 6), 3, 2, facecolor='#F0F8FF', 
                         edgecolor='navy', linewidth=1.5)
    ax.add_patch(perf_rect)
    ax.text(2, 7.6, 'Performance Targets', fontsize=10, fontweight='bold', ha='center')
    ax.text(2, 7.3, '• Clean Acc: >95%', fontsize=8, ha='center')
    ax.text(2, 7.0, '• Robust Acc: >85%', fontsize=8, ha='center')
    ax.text(2, 6.7, '• Detection AUC: >0.95', fontsize=8, ha='center')
    ax.text(2, 6.4, '• Comm. Cost: <2x', fontsize=8, ha='center')
    ax.text(2, 6.1, '• Memory: 6GB GPU', fontsize=8, ha='center')
    
    # Dataset indicators
    datasets = ['CIFAR-10', 'CIFAR-100', 'MNIST', 'BR35H']
    for i, dataset in enumerate(datasets):
        dataset_rect = Rectangle((11 + i * 0.7, 0.2), 0.6, 0.4, 
                                facecolor='lightblue', edgecolor='blue', linewidth=0.5)
        ax.add_patch(dataset_rect)
        ax.text(11.3 + i * 0.7, 0.4, dataset, fontsize=7, ha='center', fontweight='bold')
    
    ax.text(12.5, 0.8, 'Supported Datasets', fontsize=9, ha='center', fontweight='bold')
    
    plt.tight_layout()
    return fig

def create_flow_diagram():
    """Create a simplified flow diagram"""
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 8)
    ax.axis('off')
    
    # Title
    ax.text(6, 7.5, 'Test-Time Defense Flow', fontsize=16, fontweight='bold', ha='center')
    
    # Colors
    colors = {
        'input': '#FFEBEE',
        'decision': '#FFF3E0',
        'process': '#E8F5E8',
        'output': '#E3F2FD'
    }
    
    # Flow steps
    steps = [
        (1.5, 6, 'Input x', colors['input']),
        (4, 6, 'MAE\nDetection', colors['process']),
        (6.5, 6, 'Decision\ns_det > τ?', colors['decision']),
        (9, 6, 'DiffPure\nPurification', colors['process']),
        (11, 6, 'Clean\nPath', colors['process']),
        (6.5, 3, 'pFedDef\nEnsemble', colors['process']),
        (6.5, 1, 'Final\nPrediction', colors['output'])
    ]
    
    # Draw steps
    for x, y, text, color in steps:
        if 'Decision' in text:
            # Diamond shape for decision
            diamond = patches.RegularPolygon((x, y), 4, radius=0.8,
                                           facecolor=color, edgecolor='orange')
            ax.add_patch(diamond)
        else:
            # Rectangle for process
            rect = Rectangle((x-0.7, y-0.4), 1.4, 0.8, facecolor=color,
                           edgecolor='black', linewidth=1.5)
            ax.add_patch(rect)
        ax.text(x, y, text, fontsize=9, fontweight='bold', ha='center', va='center')
    
    # Arrows
    arrows = [
        ((2.2, 6), (3.3, 6)),      # Input to MAE
        ((4.7, 6), (5.8, 6)),      # MAE to Decision
        ((7.3, 6), (8.3, 6)),      # Decision to DiffPure (Yes)
        ((7.3, 6), (10.3, 6)),     # Decision to Clean (No)
        ((9, 5.6), (7.2, 3.4)),    # DiffPure to Ensemble
        ((11, 5.6), (7.2, 3.4)),   # Clean to Ensemble
        ((6.5, 2.6), (6.5, 1.4))   # Ensemble to Output
    ]
    
    for start, end in arrows:
        ax.annotate('', xy=end, xytext=start,
                    arrowprops=dict(arrowstyle='->', lw=1.5, color='black'))
    
    # Labels on arrows
    ax.text(8.5, 6.3, 'Yes', fontsize=8, color='red', fontweight='bold')
    ax.text(9.5, 6.3, 'No', fontsize=8, color='green', fontweight='bold')
    
    # Performance metrics
    ax.text(1, 4.5, 'Performance:', fontsize=10, fontweight='bold')
    ax.text(1, 4.2, '• Detection: <10ms', fontsize=8)
    ax.text(1, 3.9, '• Purification: <200ms', fontsize=8)
    ax.text(1, 3.6, '• Total: <250ms', fontsize=8)
    ax.text(1, 3.3, '• Memory: <2GB', fontsize=8)
    
    plt.tight_layout()
    return fig

if __name__ == "__main__":
    try:
        print("🔄 Creating simplified architecture diagram...")
        
        # Create main architecture
        fig1 = create_simple_architecture()
        fig1.savefig('architecture_overview.png', dpi=300, bbox_inches='tight',
                     facecolor='white', edgecolor='none')
        fig1.savefig('architecture_overview.pdf', bbox_inches='tight',
                     facecolor='white', edgecolor='none')
        print("✅ Architecture overview saved")
        plt.close(fig1)
        
        # Create flow diagram
        fig2 = create_flow_diagram()
        fig2.savefig('defense_flow_diagram.png', dpi=300, bbox_inches='tight',
                     facecolor='white', edgecolor='none')
        fig2.savefig('defense_flow_diagram.pdf', bbox_inches='tight',
                     facecolor='white', edgecolor='none')
        print("✅ Defense flow diagram saved")
        plt.close(fig2)
        
        print("\n📊 All diagrams created successfully!")
        print("🔹 architecture_overview.png/pdf - System architecture")
        print("🔹 defense_flow_diagram.png/pdf - Defense pipeline flow")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
