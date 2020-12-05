from pathlib import Path
import os
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "sans-serif"

fontsize = 16.0
big_fontsize = 19.0
fig_width = 6
golden_ratio = 1.618
fig_height = fig_width/golden_ratio

plot_dir = os.path.join(Path(__file__).parent.parent.absolute(), "plots")