#!/usr/bin/env python3
"""
Create comprehensive summary visualization of flatfield correction results.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import os

def load_test_results():
    """Load all test results from JSON files."""
    results = {}
    base_dir = "/data/home/cmarc/SWIR_Projects/Flatfield/Images"
    
    for pos in ['pos1', 'pos2', 'pos3', 'pos4']:
        json_file = os.path.join(base_dir, f"updated_test_results_{pos}.json")
        if os.path.exists(json_file):
            with open(json_file, 'r') as f:
                results[pos] = json.load(f)
        else:
            print(f"Warning: {json_file} not found")
    
    return results

def create_performance_summary(results):
    """Create performance summary plots."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Data extraction
    positions = []
    wavelengths = []
    methods = []
    mean_improvements = []
    success_rates = []
    n_images = []
    
    wavelength_map = {
        'pos1': '1.57μm',
        'pos2': '1.55μm', 
        'pos3': '1.38μm',
        'pos4': 'Open'
    }
    
    for pos, data in results.items():
        if 'summary' in data:
            positions.append(pos)
            wavelengths.append(wavelength_map.get(pos, 'Unknown'))
            methods.append(data['correction_method'])
            mean_improvements.append(data['summary']['mean_improvement'])
            success_rates.append(data['summary']['success_rate'])
            n_images.append(data['n_images_tested'])
    
    # Plot 1: Mean Improvement by Position
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    bars = axes[0, 0].bar(positions, mean_improvements, color=colors, alpha=0.7)
    axes[0, 0].set_title('Mean Uniformity Improvement by Position', fontweight='bold', fontsize=12)
    axes[0, 0].set_ylabel('Improvement (%)')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, improvement in zip(bars, mean_improvements):
        height = bar.get_height()
        axes[0, 0].text(bar.get_x() + bar.get_width()/2., height + 0.02,
                       f'{improvement:.2f}%', ha='center', va='bottom', fontweight='bold')
    
    # Add wavelength labels
    for i, (pos, wavelength) in enumerate(zip(positions, wavelengths)):
        axes[0, 0].text(i, -0.15, wavelength, ha='center', va='top', 
                       transform=axes[0, 0].get_xaxis_transform(), fontsize=10)
    
    # Plot 2: Method Distribution
    method_counts = {}
    for method in methods:
        method_counts[method] = method_counts.get(method, 0) + 1
    
    method_colors = {'multiply': '#2ca02c', 'divide': '#d62728'}
    wedges, texts, autotexts = axes[0, 1].pie(method_counts.values(), 
                                             labels=method_counts.keys(),
                                             autopct='%1.0f%%',
                                             colors=[method_colors.get(k, '#gray') for k in method_counts.keys()],
                                             startangle=90)
    axes[0, 1].set_title('Correction Method Distribution', fontweight='bold', fontsize=12)
    
    # Plot 3: Individual Image Improvements
    all_improvements = []
    all_positions_expanded = []
    
    for pos, data in results.items():
        if 'uniformity_improvements' in data:
            improvements = data['uniformity_improvements']
            all_improvements.extend(improvements)
            all_positions_expanded.extend([pos] * len(improvements))
    
    # Box plot of improvements by position
    improvement_data = []
    pos_labels = []
    for pos in positions:
        if pos in results and 'uniformity_improvements' in results[pos]:
            improvement_data.append(results[pos]['uniformity_improvements'])
            pos_labels.append(f"{pos}\n({wavelength_map[pos]})")
    
    if improvement_data:
        bp = axes[1, 0].boxplot(improvement_data, labels=pos_labels, patch_artist=True)
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        axes[1, 0].set_title('Distribution of Individual Image Improvements', fontweight='bold', fontsize=12)
        axes[1, 0].set_ylabel('Improvement (%)')
        axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Performance Rating Summary
    ratings = []
    for pos, data in results.items():
        if 'summary' in data:
            improvement = data['summary']['mean_improvement']
            if improvement > 1.5:
                ratings.append('✅ GOOD')
            elif improvement > 0.5:
                ratings.append('⚠️ MODERATE')
            else:
                ratings.append('❌ POOR')
    
    rating_counts = {}
    for rating in ratings:
        rating_counts[rating] = rating_counts.get(rating, 0) + 1
    
    rating_colors = {'✅ GOOD': '#2ca02c', '⚠️ MODERATE': '#ff7f0e', '❌ POOR': '#d62728'}
    if rating_counts:
        wedges, texts, autotexts = axes[1, 1].pie(rating_counts.values(),
                                                 labels=rating_counts.keys(),
                                                 autopct='%1.0f%%',
                                                 colors=[rating_colors.get(k, '#gray') for k in rating_counts.keys()],
                                                 startangle=90)
        axes[1, 1].set_title('Overall Performance Rating', fontweight='bold', fontsize=12)
    
    plt.suptitle('SWIR Flatfield Correction - Position-Optimized Performance Summary', 
                 fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    # Save the plot
    output_path = "/data/home/cmarc/SWIR_Projects/Flatfield/Images/flatfield_performance_summary.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"✓ Performance summary saved to: {output_path}")

def create_detailed_comparison_table(results):
    """Create a detailed comparison table."""
    print("\n" + "="*100)
    print("DETAILED FLATFIELD CORRECTION PERFORMANCE ANALYSIS")
    print("="*100)
    
    wavelength_map = {
        'pos1': '1.57μm',
        'pos2': '1.55μm', 
        'pos3': '1.38μm',
        'pos4': 'Open'
    }
    
    header = f"{'Position':<10} {'Wavelength':<12} {'Method':<10} {'Mean Imp':<10} {'Median Imp':<12} {'Std Dev':<10} {'Range':<15} {'Images':<8} {'Rating':<12}"
    print(header)
    print("-" * len(header))
    
    for pos in ['pos1', 'pos2', 'pos3', 'pos4']:
        if pos in results and 'summary' in results[pos]:
            data = results[pos]
            summary = data['summary']
            
            wavelength = wavelength_map.get(pos, 'Unknown')
            method = data['correction_method']
            mean_imp = summary['mean_improvement']
            median_imp = summary.get('median_improvement', 'N/A')
            std_imp = summary.get('std_improvement', 'N/A')
            min_imp = summary.get('min_improvement', 'N/A')
            max_imp = summary.get('max_improvement', 'N/A')
            n_images = data['n_images_tested']
            
            # Determine rating
            if mean_imp > 1.5:
                rating = '✅ GOOD'
            elif mean_imp > 0.5:
                rating = '⚠️ MODERATE'
            else:
                rating = '❌ POOR'
            
            range_str = f"{min_imp:.2f} to {max_imp:.2f}" if isinstance(min_imp, (int, float)) else "N/A"
            median_str = f"{median_imp:.2f}%" if isinstance(median_imp, (int, float)) else "N/A"
            std_str = f"{std_imp:.2f}" if isinstance(std_imp, (int, float)) else "N/A"
            
            row = f"{pos:<10} {wavelength:<12} {method:<10} {mean_imp:<10.2f} {median_str:<12} {std_str:<10} {range_str:<15} {n_images:<8} {rating:<12}"
            print(row)
    
    print("\n" + "="*50)
    print("KEY FINDINGS:")
    print("="*50)
    
    # Find best and worst performers
    best_improvement = -999
    worst_improvement = 999
    best_pos = None
    worst_pos = None
    
    for pos, data in results.items():
        if 'summary' in data:
            improvement = data['summary']['mean_improvement']
            if improvement > best_improvement:
                best_improvement = improvement
                best_pos = pos
            if improvement < worst_improvement:
                worst_improvement = improvement
                worst_pos = pos
    
    if best_pos:
        print(f"🏆 BEST PERFORMER: {best_pos} ({wavelength_map.get(best_pos, 'Unknown')}) with {best_improvement:.2f}% improvement")
    
    if worst_pos:
        print(f"⚠️  LEAST EFFECTIVE: {worst_pos} ({wavelength_map.get(worst_pos, 'Unknown')}) with {worst_improvement:.2f}% improvement")

    # Calculate overall statistics
    all_improvements = []
    for pos, data in results.items():
        if 'uniformity_improvements' in data:
            all_improvements.extend(data['uniformity_improvements'])
    
    if all_improvements:
        overall_mean = np.mean(all_improvements)
        overall_median = np.median(all_improvements)
        overall_std = np.std(all_improvements)
        
        print(f"\n📊 OVERALL STATISTICS:")
        print(f"   Total images processed: {len(all_improvements)}")
        print(f"   Overall mean improvement: {overall_mean:.2f}%")
        print(f"   Overall median improvement: {overall_median:.2f}%")
        print(f"   Overall standard deviation: {overall_std:.2f}%")
        
        success_count = sum(1 for imp in all_improvements if imp > 0)
        success_rate = success_count / len(all_improvements) * 100
        print(f"   Success rate (positive improvement): {success_rate:.1f}%")

def main():
    """Main execution function."""
    print("Loading flatfield correction test results...")
    
    # Load all test results
    results = load_test_results()
    
    if not results:
        print("Error: No test results found!")
        return
    
    print(f"Loaded results for {len(results)} positions: {list(results.keys())}")
    
    # Create performance summary visualization
    print("\nCreating performance summary visualization...")
    create_performance_summary(results)
    
    # Create detailed comparison table
    create_detailed_comparison_table(results)
    
    print("\n✅ Summary analysis completed!")

if __name__ == "__main__":
    main()
