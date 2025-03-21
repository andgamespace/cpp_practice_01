#!/usr/bin/env python3
"""
Generate architecture diagram for the trading environment.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

def create_architecture_diagram():
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Set background color
    ax.set_facecolor('#f5f5f5')
    
    # Define colors
    colors = {
        'python': '#3776AB',
        'cpp': '#00599C',
        'react': '#61DAFB',
        'pytorch': '#EE4C2C',
        'arrow': '#D8D8D8',
        'websocket': '#4B8BBE',
        'taskflow': '#FF6F61',
        'arrow_color': '#555555'
    }
    
    # Define component positions
    components = {
        'python_api': {'pos': (0.2, 0.9), 'width': 0.2, 'height': 0.1, 'color': colors['python'], 'label': 'Python API'},
        'cpp_backend': {'pos': (0.5, 0.9), 'width': 0.2, 'height': 0.1, 'color': colors['cpp'], 'label': 'C++ Backend'},
        'react_frontend': {'pos': (0.8, 0.9), 'width': 0.2, 'height': 0.1, 'color': colors['react'], 'label': 'React Frontend'},
        
        'gymnasium': {'pos': (0.2, 0.75), 'width': 0.2, 'height': 0.1, 'color': colors['python'], 'label': 'Gymnasium Interface'},
        'dataloader': {'pos': (0.5, 0.75), 'width': 0.2, 'height': 0.1, 'color': colors['cpp'], 'label': 'DataLoader'},
        'websocket_client': {'pos': (0.8, 0.75), 'width': 0.2, 'height': 0.1, 'color': colors['react'], 'label': 'WebSocket Client'},
        
        'pytorch': {'pos': (0.2, 0.6), 'width': 0.2, 'height': 0.1, 'color': colors['pytorch'], 'label': 'PyTorch Models'},
        'backtest_engine': {'pos': (0.5, 0.6), 'width': 0.2, 'height': 0.1, 'color': colors['cpp'], 'label': 'BacktestEngine'},
        'charts': {'pos': (0.8, 0.6), 'width': 0.2, 'height': 0.1, 'color': colors['react'], 'label': 'Chart Components'},
        
        'drl': {'pos': (0.2, 0.45), 'width': 0.2, 'height': 0.1, 'color': colors['python'], 'label': 'DRL Algorithms'},
        'indicators': {'pos': (0.5, 0.45), 'width': 0.2, 'height': 0.1, 'color': colors['cpp'], 'label': 'TechnicalIndicators'},
        'dashboard': {'pos': (0.8, 0.45), 'width': 0.2, 'height': 0.1, 'color': colors['react'], 'label': 'Dashboard UI'},
        
        'taskflow': {'pos': (0.5, 0.3), 'width': 0.2, 'height': 0.1, 'color': colors['taskflow'], 'label': 'Taskflow Concurrency'},
        'websocket_server': {'pos': (0.5, 0.15), 'width': 0.2, 'height': 0.1, 'color': colors['websocket'], 'label': 'WebSocket Server'},
    }
    
    # Draw components
    for name, comp in components.items():
        rect = patches.Rectangle(
            comp['pos'], comp['width'], comp['height'],
            linewidth=1, edgecolor='black', facecolor=comp['color'], alpha=0.7,
            label=comp['label']
        )
        ax.add_patch(rect)
        
        # Add label
        ax.text(
            comp['pos'][0] + comp['width']/2,
            comp['pos'][1] + comp['height']/2,
            comp['label'],
            ha='center', va='center',
            color='white', fontweight='bold'
        )
    
    # Define connections
    connections = [
        ('python_api', 'gymnasium'),
        ('python_api', 'cpp_backend'),
        ('gymnasium', 'pytorch'),
        ('pytorch', 'drl'),
        
        ('cpp_backend', 'dataloader'),
        ('cpp_backend', 'backtest_engine'),
        ('dataloader', 'backtest_engine'),
        ('backtest_engine', 'indicators'),
        ('backtest_engine', 'taskflow'),
        ('indicators', 'taskflow'),
        ('backtest_engine', 'websocket_server'),
        
        ('react_frontend', 'websocket_client'),
        ('websocket_client', 'charts'),
        ('charts', 'dashboard'),
        
        ('websocket_server', 'websocket_client'),
    ]
    
    # Draw connections
    for start, end in connections:
        start_comp = components[start]
        end_comp = components[end]
        
        # Calculate start and end points
        if start_comp['pos'][1] > end_comp['pos'][1]:  # Vertical connection
            start_x = start_comp['pos'][0] + start_comp['width']/2
            start_y = start_comp['pos'][1]
            end_x = end_comp['pos'][0] + end_comp['width']/2
            end_y = end_comp['pos'][1] + end_comp['height']
        else:  # Horizontal connection
            start_x = start_comp['pos'][0] + start_comp['width']
            start_y = start_comp['pos'][1] + start_comp['height']/2
            end_x = end_comp['pos'][0]
            end_y = end_comp['pos'][1] + end_comp['height']/2
        
        # Draw arrow
        ax.arrow(
            start_x, start_y, end_x - start_x, end_y - start_y,
            head_width=0.01, head_length=0.01, fc=colors['arrow_color'], ec=colors['arrow_color'],
            length_includes_head=True, linewidth=1.5
        )
    
    # Set axis limits
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    
    # Remove axis ticks and labels
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    
    # Add title
    ax.set_title('Trading Environment Architecture', fontsize=16, fontweight='bold')
    
    # Add legend for component types
    legend_elements = [
        patches.Patch(facecolor=colors['python'], edgecolor='black', alpha=0.7, label='Python Components'),
        patches.Patch(facecolor=colors['cpp'], edgecolor='black', alpha=0.7, label='C++ Components'),
        patches.Patch(facecolor=colors['react'], edgecolor='black', alpha=0.7, label='React Components'),
        patches.Patch(facecolor=colors['pytorch'], edgecolor='black', alpha=0.7, label='PyTorch Integration'),
        patches.Patch(facecolor=colors['taskflow'], edgecolor='black', alpha=0.7, label='Concurrency'),
        patches.Patch(facecolor=colors['websocket'], edgecolor='black', alpha=0.7, label='Communication')
    ]
    ax.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=3)
    
    # Add data flow description
    data_flow = """
    Data Flow:
    1. Python Interface calls C++ Backend through PyBind11
    2. C++ Backend processes data using Taskflow for concurrency
    3. BacktestEngine calculates technical indicators and runs simulations
    4. Results are sent to WebSocket Server for real-time updates
    5. React Frontend displays data through Chart Components
    6. DRL Algorithms interact with the environment through Gymnasium
    """
    ax.text(0.02, -0.15, data_flow, ha='left', va='top', fontsize=10, transform=ax.transAxes)
    
    # Save figure
    plt.tight_layout()
    plt.savefig('architecture.png', dpi=300, bbox_inches='tight')
    print("Architecture diagram saved as 'architecture.png'")

if __name__ == "__main__":
    create_architecture_diagram()