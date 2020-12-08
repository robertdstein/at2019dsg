from pathlib import Path
import os
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "sans-serif"

fontsize = 5.
big_fontsize = 7.0
ed_big_fontsize = 12.0


# Nature Astronomy 2-column Figures should be ~188 mm high and 180 mm wide

inch_to_mm = 25.4

fig_width = 180. / inch_to_mm
fig_height = 185./ inch_to_mm
small_fig_width = 88. / inch_to_mm
small_fig_height = 130./ inch_to_mm

marker_size=3

dpi = 300

ed_fig_width = 6
golden_ratio = 1.618
ed_fig_height = ed_fig_width/golden_ratio

plot_dir = os.path.join(Path(__file__).parent.parent.absolute(), "plots")