import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Read the CSV file
df = pd.read_csv('/Users/luogan/Code/sglang/test_new_test/router_test_20250807_174029.csv')

# Get unique hosts
unique_hosts = df['host'].unique()
print(f"Unique hosts found: {unique_hosts}")

# Create lists to store the proportions
num_requests = []
host_proportions = {host: [] for host in unique_hosts}

# Calculate cumulative proportions for each row
for i in range(1, len(df) + 1):
    df_subset = df.iloc[:i]
    host_counts = df_subset['host'].value_counts()
    
    num_requests.append(i)
    
    for host in unique_hosts:
        if host in host_counts:
            proportion = host_counts[host] / i
        else:
            proportion = 0
        host_proportions[host].append(proportion)

# Create the plot
plt.figure(figsize=(12, 8))

# Define colors for different hosts
colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']
host_colors = {host: colors[i % len(colors)] for i, host in enumerate(unique_hosts)}

# Plot lines for each host
for host in unique_hosts:
    plt.plot(num_requests, host_proportions[host], 
             label=host.replace('http://localhost:', 'Port '), 
             linewidth=2, 
             color=host_colors[host],
             alpha=0.8)

# Customize the plot
plt.xlabel('Number of Requests', fontsize=12)
plt.ylabel('Proportion of Requests', fontsize=12)
plt.title('Distribution of Requests Across Different Hosts', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3, linestyle='--')
plt.legend(loc='best', fontsize=10)

# Format y-axis as percentage
plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0%}'.format(y)))

# Add some statistics as text
total_requests = len(df)
final_proportions = {host: host_proportions[host][-1] for host in unique_hosts}
stats_text = f"Total Requests: {total_requests}\n"
stats_text += "Final Distribution:\n"
for host, prop in final_proportions.items():
    stats_text += f"  {host.replace('http://localhost:', 'Port ')}: {prop:.1%}\n"

plt.text(0.02, 0.98, stats_text, 
         transform=plt.gca().transAxes,
         verticalalignment='top',
         fontsize=9,
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig('/Users/luogan/Code/sglang/host_distribution_plot.png', dpi=300, bbox_inches='tight')
plt.show()

print(f"\nPlot saved as 'host_distribution_plot.png'")
print(f"\nFinal distribution:")
for host, prop in final_proportions.items():
    count = int(prop * total_requests)
    print(f"  {host}: {count} requests ({prop:.1%})")