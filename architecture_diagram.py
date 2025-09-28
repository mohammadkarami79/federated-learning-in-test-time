#!/usr/bin/env python3
"""
Architecture Diagram Generator for Multi-Layered Federated Defense Framework
Creates a comprehensive visual representation of the system architecture
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, Circle, Rectangle, Arrow
import numpy as np

# Set style for professional publication
plt.style.use('default')
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 10
plt.rcParams['figure.figsize'] = (16, 12)
plt.rcParams['figure.dpi'] = 300

def create_architecture_diagram():
    fig, ax = plt.subplots(1, 1, figsize=(16, 12))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 12)
    ax.axis('off')
    
    # Color scheme - professional and distinct
    colors = {
        'client': '#E3F2FD',           # Light blue
        'server': '#F3E5F5',          # Light purple
        'mae': '#E8F5E8',             # Light green
        'diffusion': '#FFF3E0',       # Light orange
        'pfeddef': '#FCE4EC',         # Light pink
        'attack': '#FFEBEE',          # Light red
        'flow': '#424242',            # Dark gray
        'highlight': '#1976D2'        # Blue
    }
    
    # Title
    ax.text(8, 11.5, 'Multi-Layered Federated Defense Framework Architecture', 
            fontsize=16, fontweight='bold', ha='center')
    
    # ========================= ATTACK LAYER =========================
    # Adversarial Input
    attack_box = FancyBboxPatch((0.5, 9.5), 3, 1.5, boxstyle="round,pad=0.1", 
                               facecolor=colors['attack'], edgecolor='red', linewidth=2)
    ax.add_patch(attack_box)
    ax.text(2, 10.7, 'Adversarial Input', fontsize=11, fontweight='bold', ha='center')
    ax.text(2, 10.3, 'x_adv = x + δ', fontsize=9, ha='center', style='italic')
    ax.text(2, 10.0, '||δ||∞ ≤ ε', fontsize=9, ha='center', style='italic')
    ax.text(2, 9.7, 'PGD, C&W, AutoAttack', fontsize=8, ha='center')
    
    # ========================= DEFENSE PIPELINE =========================
    
    # Layer 1: MAE Detection
    mae_box = FancyBboxPatch((5, 9), 2.5, 2.5, boxstyle="round,pad=0.1",
                            facecolor=colors['mae'], edgecolor='green', linewidth=2)
    ax.add_patch(mae_box)
    ax.text(6.25, 11.2, 'Layer 1: MAE Detection', fontsize=11, fontweight='bold', ha='center')
    ax.text(6.25, 10.8, 'Vision Transformer', fontsize=9, ha='center')
    ax.text(6.25, 10.5, 'Patch Size: 4×4/16×16', fontsize=8, ha='center')
    ax.text(6.25, 10.2, 'Mask Ratio: 0.75', fontsize=8, ha='center')
    ax.text(6.25, 9.9, 'Multi-Scale Detection', fontsize=8, ha='center')
    ax.text(6.25, 9.6, 's_det = f_mae(x)', fontsize=8, ha='center', style='italic')
    ax.text(6.25, 9.3, 'Adaptive Threshold', fontsize=8, ha='center')
    
    # Layer 2: Diffusion Purification
    diff_box = FancyBboxPatch((8.5, 9), 2.5, 2.5, boxstyle="round,pad=0.1",
                             facecolor=colors['diffusion'], edgecolor='orange', linewidth=2)
    ax.add_patch(diff_box)
    ax.text(9.75, 11.2, 'Layer 2: DiffPure', fontsize=11, fontweight='bold', ha='center')
    ax.text(9.75, 10.8, 'U-Net Diffusion', fontsize=9, ha='center')
    ax.text(9.75, 10.5, 'Hidden: 256 channels', fontsize=8, ha='center')
    ax.text(9.75, 10.2, 'DDIM Sampling', fontsize=8, ha='center')
    ax.text(9.75, 9.9, 'Timesteps: 1-50', fontsize=8, ha='center')
    ax.text(9.75, 9.6, 'x_pure = DDIM(x, t)', fontsize=8, ha='center', style='italic')
    ax.text(9.75, 9.3, 'Conditional on s_det', fontsize=8, ha='center')
    
    # Layer 3: pFedDef Ensemble
    pfed_box = FancyBboxPatch((12, 9), 3, 2.5, boxstyle="round,pad=0.1",
                             facecolor=colors['pfeddef'], edgecolor='purple', linewidth=2)
    ax.add_patch(pfed_box)
    ax.text(13.5, 11.2, 'Layer 3: pFedDef', fontsize=11, fontweight='bold', ha='center')
    ax.text(13.5, 10.8, 'Mixture of Experts (K=3)', fontsize=9, ha='center')
    ax.text(13.5, 10.5, 'ResNet-18 Backbone', fontsize=8, ha='center')
    ax.text(13.5, 10.2, 'Attention Mechanism', fontsize=8, ha='center')
    ax.text(13.5, 9.9, 'ŷ = Σ αₖ·fₖ(x_pure)', fontsize=8, ha='center', style='italic')
    ax.text(13.5, 9.6, 'Dynamic Weighting', fontsize=8, ha='center')
    ax.text(13.5, 9.3, 'Meta-Learning', fontsize=8, ha='center')
    
    # ========================= FEDERATED LEARNING LAYER =========================
    
    # Client Side
    client_y = 6.5
    for i in range(3):
        x_pos = 1 + i * 4.5
        
        # Client box
        client_box = FancyBboxPatch((x_pos, client_y), 3.5, 1.8, boxstyle="round,pad=0.1",
                                   facecolor=colors['client'], edgecolor='blue', linewidth=1.5)
        ax.add_patch(client_box)
        ax.text(x_pos + 1.75, client_y + 1.5, f'Client {i+1}', fontsize=10, fontweight='bold', ha='center')
        
        # Local components
        ax.text(x_pos + 1.75, client_y + 1.2, 'Local Dataset D_i', fontsize=8, ha='center')
        ax.text(x_pos + 1.75, client_y + 1.0, 'pFedDef Training', fontsize=8, ha='center')
        ax.text(x_pos + 1.75, client_y + 0.8, f'θᵢ = {{θᵢ⁽¹⁾, θᵢ⁽²⁾, θᵢ⁽³⁾}}', fontsize=7, ha='center', style='italic')
        ax.text(x_pos + 1.75, client_y + 0.6, 'Local Epochs: 5-8', fontsize=8, ha='center')
        ax.text(x_pos + 1.75, client_y + 0.4, 'SGD + Momentum', fontsize=8, ha='center')
        ax.text(x_pos + 1.75, client_y + 0.1, 'Defense Integration', fontsize=8, ha='center')
    
    # Server
    server_box = FancyBboxPatch((5.5, 4), 5, 1.8, boxstyle="round,pad=0.1",
                               facecolor=colors['server'], edgecolor='purple', linewidth=2)
    ax.add_patch(server_box)
    ax.text(8, 5.5, 'Federated Server', fontsize=12, fontweight='bold', ha='center')
    ax.text(8, 5.1, 'Robust Aggregation: FedAvg + Byzantine Tolerance', fontsize=9, ha='center')
    ax.text(8, 4.8, 'θ_global = Σ (nᵢ/n) · θᵢ', fontsize=9, ha='center', style='italic')
    ax.text(8, 4.5, 'Global Model Distribution', fontsize=9, ha='center')
    ax.text(8, 4.2, 'Communication Rounds: 15-25', fontsize=8, ha='center')
    
    # ========================= MODEL COMPONENTS =========================
    
    # Pre-trained Models
    models_y = 2
    
    # Diffusion Model
    diff_model = FancyBboxPatch((1, models_y), 3, 1.5, boxstyle="round,pad=0.1",
                               facecolor=colors['diffusion'], edgecolor='orange', linewidth=1.5)
    ax.add_patch(diff_model)
    ax.text(2.5, models_y + 1.1, 'Diffusion Model', fontsize=10, fontweight='bold', ha='center')
    ax.text(2.5, models_y + 0.8, 'U-Net Architecture', fontsize=8, ha='center')
    ax.text(2.5, models_y + 0.6, 'Training: 50-100 epochs', fontsize=8, ha='center')
    ax.text(2.5, models_y + 0.4, 'Dataset-specific', fontsize=8, ha='center')
    ax.text(2.5, models_y + 0.1, 'Checkpoint: diffuser_*.pt', fontsize=7, ha='center')
    
    # MAE Model
    mae_model = FancyBboxPatch((5, models_y), 3, 1.5, boxstyle="round,pad=0.1",
                              facecolor=colors['mae'], edgecolor='green', linewidth=1.5)
    ax.add_patch(mae_model)
    ax.text(6.5, models_y + 1.1, 'MAE Detector', fontsize=10, fontweight='bold', ha='center')
    ax.text(6.5, models_y + 0.8, 'ViT Encoder-Decoder', fontsize=8, ha='center')
    ax.text(6.5, models_y + 0.6, 'Training: 30-50 epochs', fontsize=8, ha='center')
    ax.text(6.5, models_y + 0.4, 'Reconstruction Loss', fontsize=8, ha='center')
    ax.text(6.5, models_y + 0.1, 'Checkpoint: mae_*.pt', fontsize=7, ha='center')
    
    # Global Model
    global_model = FancyBboxPatch((9, models_y), 3, 1.5, boxstyle="round,pad=0.1",
                                 facecolor=colors['pfeddef'], edgecolor='purple', linewidth=1.5)
    ax.add_patch(global_model)
    ax.text(10.5, models_y + 1.1, 'Global pFedDef', fontsize=10, fontweight='bold', ha='center')
    ax.text(10.5, models_y + 0.8, 'Aggregated Parameters', fontsize=8, ha='center')
    ax.text(10.5, models_y + 0.6, 'Expert Ensemble', fontsize=8, ha='center')
    ax.text(10.5, models_y + 0.4, 'Attention Weights', fontsize=8, ha='center')
    ax.text(10.5, models_y + 0.1, 'federated_*.pt', fontsize=7, ha='center')
    
    # Evaluation
    eval_box = FancyBboxPatch((12.5, models_y), 3, 1.5, boxstyle="round,pad=0.1",
                             facecolor='#F5F5F5', edgecolor='black', linewidth=1.5)
    ax.add_patch(eval_box)
    ax.text(14, models_y + 1.1, 'Evaluation Metrics', fontsize=10, fontweight='bold', ha='center')
    ax.text(14, models_y + 0.8, 'Clean Accuracy', fontsize=8, ha='center')
    ax.text(14, models_y + 0.6, 'Adversarial Accuracy', fontsize=8, ha='center')
    ax.text(14, models_y + 0.4, 'Detection Rate', fontsize=8, ha='center')
    ax.text(14, models_y + 0.1, 'Communication Cost', fontsize=8, ha='center')
    
    # ========================= FLOW ARROWS =========================
    
    # Input flow
    ax.arrow(3.5, 10.2, 1.3, 0, head_width=0.1, head_length=0.1, fc=colors['flow'], ec=colors['flow'])
    ax.text(4.25, 10.5, 'Input', fontsize=8, ha='center')
    
    # Between defense layers
    ax.arrow(7.5, 10.2, 0.8, 0, head_width=0.1, head_length=0.1, fc=colors['flow'], ec=colors['flow'])
    ax.text(7.9, 10.5, 'if detected', fontsize=7, ha='center')
    
    ax.arrow(11, 10.2, 0.8, 0, head_width=0.1, head_length=0.1, fc=colors['flow'], ec=colors['flow'])
    ax.text(11.4, 10.5, 'purified', fontsize=7, ha='center')
    
    # From defense to output
    ax.arrow(13.5, 9, 0, -0.5, head_width=0.1, head_length=0.1, fc=colors['flow'], ec=colors['flow'])
    ax.text(13.8, 8.3, 'ŷ', fontsize=10, ha='center', fontweight='bold')
    
    # Federated learning flow
    # Clients to server
    for i in range(3):
        x_start = 2.75 + i * 4.5
        ax.arrow(x_start, 6.5, (8 - x_start) * 0.7, -1.2, head_width=0.1, head_length=0.1, 
                fc=colors['flow'], ec=colors['flow'], alpha=0.7)
    
    # Server to clients
    for i in range(3):
        x_end = 2.75 + i * 4.5
        ax.arrow(8, 5.8, (x_end - 8) * 0.7, 1.2, head_width=0.1, head_length=0.1, 
                fc=colors['flow'], ec=colors['flow'], alpha=0.7)
    
    ax.text(8, 6.8, 'Model\nUpdates', fontsize=8, ha='center', fontweight='bold')
    
    # Pre-trained models to defense layers
    ax.arrow(6.5, 3.5, 0, 5.3, head_width=0.1, head_length=0.1, fc='gray', ec='gray', alpha=0.6)
    ax.arrow(2.5, 3.5, 7, 5.3, head_width=0.1, head_length=0.1, fc='gray', ec='gray', alpha=0.6)
    
    # ========================= LEGEND =========================
    
    legend_x = 0.5
    legend_y = 0.5
    
    # Legend box
    legend_box = Rectangle((legend_x, legend_y), 15, 1, facecolor='white', 
                          edgecolor='black', linewidth=1, alpha=0.9)
    ax.add_patch(legend_box)
    
    ax.text(legend_x + 0.2, legend_y + 0.7, 'Legend:', fontsize=10, fontweight='bold')
    
    # Legend items
    legend_items = [
        ('Defense Layers', colors['mae'], 1.5),
        ('Federated Components', colors['client'], 4),
        ('Pre-trained Models', colors['diffusion'], 6.5),
        ('Data Flow', colors['flow'], 9),
        ('Evaluation', '#F5F5F5', 11.5),
        ('Adversarial Input', colors['attack'], 14)
    ]
    
    for item, color, x_offset in legend_items:
        legend_rect = Rectangle((legend_x + x_offset, legend_y + 0.3), 0.3, 0.2, 
                               facecolor=color, edgecolor='black', linewidth=0.5)
        ax.add_patch(legend_rect)
        ax.text(legend_x + x_offset + 0.4, legend_y + 0.4, item, fontsize=8)
    
    # ========================= DATASET INDICATORS =========================
    
    # Dataset boxes
    datasets = ['CIFAR-10', 'CIFAR-100', 'MNIST', 'BR35H']
    for i, dataset in enumerate(datasets):
        dataset_box = FancyBboxPatch((13 + i * 0.7, 0.1), 0.6, 0.3, boxstyle="round,pad=0.02",
                                    facecolor='lightblue', edgecolor='blue', linewidth=0.5)
        ax.add_patch(dataset_box)
        ax.text(13.3 + i * 0.7, 0.25, dataset, fontsize=6, ha='center', fontweight='bold')
    
    ax.text(14.5, 0.5, 'Supported Datasets', fontsize=8, ha='center', fontweight='bold')
    
    # ========================= PERFORMANCE INDICATORS =========================
    
    # Performance metrics box
    perf_box = FancyBboxPatch((0.5, 7.5), 3.5, 1.5, boxstyle="round,pad=0.1",
                             facecolor='#F0F8FF', edgecolor='navy', linewidth=1.5)
    ax.add_patch(perf_box)
    ax.text(2.25, 8.7, 'Expected Performance', fontsize=10, fontweight='bold', ha='center')
    ax.text(2.25, 8.4, 'Clean Acc: >95%', fontsize=8, ha='center')
    ax.text(2.25, 8.2, 'Robust Acc: >85%', fontsize=8, ha='center')
    ax.text(2.25, 8.0, 'Detection AUC: >0.95', fontsize=8, ha='center')
    ax.text(2.25, 7.8, 'Comm. Cost: <2x FedAvg', fontsize=8, ha='center')
    ax.text(2.25, 7.6, 'Memory: 6GB GPU', fontsize=8, ha='center')
    
    plt.tight_layout()
    return fig

def create_detailed_flow_diagram():
    """Create a detailed flow diagram showing the test-time defense process"""
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Title
    ax.text(7, 9.5, 'Test-Time Defense Flow Diagram', 
            fontsize=16, fontweight='bold', ha='center')
    
    # Colors
    colors = {
        'input': '#FFEBEE',
        'decision': '#FFF3E0', 
        'process': '#E8F5E8',
        'output': '#E3F2FD'
    }
    
    # Step 1: Input
    input_box = FancyBboxPatch((1, 8), 2, 1, boxstyle="round,pad=0.1",
                              facecolor=colors['input'], edgecolor='red', linewidth=2)
    ax.add_patch(input_box)
    ax.text(2, 8.5, 'Test Input x', fontsize=11, fontweight='bold', ha='center')
    ax.text(2, 8.2, '(potentially adversarial)', fontsize=8, ha='center')
    
    # Step 2: MAE Detection
    mae_box = FancyBboxPatch((5, 8), 2.5, 1, boxstyle="round,pad=0.1",
                            facecolor=colors['process'], edgecolor='green', linewidth=2)
    ax.add_patch(mae_box)
    ax.text(6.25, 8.5, 'MAE Detection', fontsize=11, fontweight='bold', ha='center')
    ax.text(6.25, 8.2, 's_det = MAE_score(x)', fontsize=9, ha='center', style='italic')
    
    # Decision Diamond
    decision_diamond = patches.RegularPolygon((9, 8.5), 4, radius=0.6, 
                                            facecolor=colors['decision'], edgecolor='orange')
    ax.add_patch(decision_diamond)
    ax.text(9, 8.5, 's_det > τ?', fontsize=10, fontweight='bold', ha='center')
    
    # Path 1: No purification (Clean)
    clean_box = FancyBboxPatch((11, 8), 2, 1, boxstyle="round,pad=0.1",
                              facecolor=colors['process'], edgecolor='blue', linewidth=1)
    ax.add_patch(clean_box)
    ax.text(12, 8.5, 'No Purification', fontsize=10, ha='center')
    ax.text(12, 8.2, 'x_clean = x', fontsize=9, ha='center', style='italic')
    
    # Path 2: Purification (Adversarial)
    purify_box = FancyBboxPatch((7, 6), 3, 1, boxstyle="round,pad=0.1",
                               facecolor=colors['process'], edgecolor='orange', linewidth=2)
    ax.add_patch(purify_box)
    ax.text(8.5, 6.5, 'DiffPure Purification', fontsize=11, fontweight='bold', ha='center')
    ax.text(8.5, 6.2, 'x_pure = DDIM(x, t_adapt)', fontsize=9, ha='center', style='italic')
    
    # Adaptive timestep calculation
    timestep_box = FancyBboxPatch((4, 4.5), 3, 1, boxstyle="round,pad=0.1",
                                 facecolor=colors['decision'], edgecolor='orange', linewidth=1)
    ax.add_patch(timestep_box)
    ax.text(5.5, 5, 'Adaptive Timestep', fontsize=10, fontweight='bold', ha='center')
    ax.text(5.5, 4.7, 't = f(s_det, confidence)', fontsize=9, ha='center', style='italic')
    
    # Convergence point
    conv_box = FancyBboxPatch((9, 3), 3, 1, boxstyle="round,pad=0.1",
                             facecolor=colors['process'], edgecolor='purple', linewidth=2)
    ax.add_patch(conv_box)
    ax.text(10.5, 3.5, 'pFedDef Ensemble', fontsize=11, fontweight='bold', ha='center')
    ax.text(10.5, 3.2, 'ŷ = Σ αₖ · fₖ(x_proc)', fontsize=9, ha='center', style='italic')
    
    # Final output
    output_box = FancyBboxPatch((9, 1), 3, 1, boxstyle="round,pad=0.1",
                               facecolor=colors['output'], edgecolor='blue', linewidth=2)
    ax.add_patch(output_box)
    ax.text(10.5, 1.5, 'Final Prediction', fontsize=11, fontweight='bold', ha='center')
    ax.text(10.5, 1.2, 'ŷ + confidence', fontsize=9, ha='center', style='italic')
    
    # Arrows
    # Input to MAE
    ax.arrow(3, 8.5, 1.8, 0, head_width=0.1, head_length=0.1, fc='black', ec='black')
    
    # MAE to Decision
    ax.arrow(7.5, 8.5, 1, 0, head_width=0.1, head_length=0.1, fc='black', ec='black')
    
    # Decision to Clean (No)
    ax.arrow(9.5, 8.8, 1.3, -0.1, head_width=0.1, head_length=0.1, fc='green', ec='green')
    ax.text(10.2, 9, 'No', fontsize=8, color='green', fontweight='bold')
    
    # Decision to Purification (Yes)
    ax.arrow(8.8, 8, -0.2, -1.3, head_width=0.1, head_length=0.1, fc='red', ec='red')
    ax.text(8.3, 7.2, 'Yes', fontsize=8, color='red', fontweight='bold')
    
    # Purification to timestep
    ax.arrow(7.5, 6.2, -1.8, -1, head_width=0.1, head_length=0.1, fc='orange', ec='orange')
    
    # Timestep back to purification
    ax.arrow(6.5, 5.2, 1.8, 1, head_width=0.1, head_length=0.1, fc='orange', ec='orange')
    
    # Both paths to ensemble
    ax.arrow(12, 8, -1.2, -4.3, head_width=0.1, head_length=0.1, fc='blue', ec='blue')
    ax.arrow(8.5, 6, 1.5, -2.3, head_width=0.1, head_length=0.1, fc='orange', ec='orange')
    
    # Ensemble to output
    ax.arrow(10.5, 3, 0, -1.8, head_width=0.1, head_length=0.1, fc='purple', ec='purple')
    
    # Performance annotations
    ax.text(1, 7, 'Performance Metrics:', fontsize=10, fontweight='bold')
    ax.text(1, 6.7, '• Detection Time: <10ms', fontsize=8)
    ax.text(1, 6.4, '• Purification Time: <200ms', fontsize=8)
    ax.text(1, 6.1, '• Total Latency: <250ms', fontsize=8)
    ax.text(1, 5.8, '• Memory Usage: <2GB', fontsize=8)
    ax.text(1, 5.5, '• Accuracy Preservation: >98%', fontsize=8)
    
    plt.tight_layout()
    return fig

if __name__ == "__main__":
    try:
        # Create architecture diagram
        print("🔄 Creating architecture overview diagram...")
        fig1 = create_architecture_diagram()
        fig1.savefig('architecture_overview.png', dpi=300, bbox_inches='tight', 
                     facecolor='white', edgecolor='none')
        fig1.savefig('architecture_overview.pdf', bbox_inches='tight', 
                     facecolor='white', edgecolor='none')
        print("✅ Architecture overview diagram saved as PNG and PDF")
        plt.close(fig1)
        
        # Create flow diagram
        print("🔄 Creating defense flow diagram...")
        fig2 = create_detailed_flow_diagram()
        fig2.savefig('defense_flow_diagram.png', dpi=300, bbox_inches='tight',
                     facecolor='white', edgecolor='none')
        fig2.savefig('defense_flow_diagram.pdf', bbox_inches='tight',
                     facecolor='white', edgecolor='none')
        print("✅ Defense flow diagram saved as PNG and PDF")
        plt.close(fig2)
        
        print("\n📊 Diagrams created successfully!")
        print("🔹 architecture_overview.png - Complete system architecture")
        print("🔹 defense_flow_diagram.png - Test-time defense flow")
        print("🔹 Both available in PNG and PDF formats")
        
    except Exception as e:
        print(f"❌ Error creating diagrams: {e}")
        import traceback
        traceback.print_exc()
