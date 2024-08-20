import numpy as np
import laspy as lp
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from collections import Counter

las = lp.read(input("Path : "))
t = las.gps_time
t = [i + 1e9 for i in t]

# Count the number of points for each unique timestamp
timestamp_counts = Counter(t)

# Sort timestamps for plotting
sorted_timestamps = sorted(timestamp_counts.keys())
counts = [timestamp_counts[timestamp] for timestamp in sorted_timestamps]

"""# Plotting
plt.figure(figsize=(10, 6))
plt.plot(sorted_timestamps, counts, marker='o', linestyle='-', color='b')
plt.xlabel('Timestamp (GPS Time)')
plt.ylabel('Number of Points')
plt.title('Number of Points at Each Timestamp in LAS File')
plt.grid(True)
plt.show()
"""
#print(counts)
print(len(sorted_timestamps))