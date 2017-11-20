"""
GLM fitting in second level fMRI
================================

Full step-by-step example of fitting a GLM to perform a second level analysis
in experimental data and visualizing the results.

More specifically:

1. A sequence of subject fMRI button press contrasts is downloaded.
2. a mask of the useful brain volume is computed
3. A one-sample t-test is applied to the brain maps

(as fixed effects, then contrast estimation)

Author : Martin Perez-Guevara: 2016
"""

import pandas as pd
from nilearn import plotting
from scipy.stats import norm
import matplotlib.pyplot as plt

from nilearn.datasets import fetch_localizer_contrasts
from nistats.second_level_model import SecondLevelModel

#########################################################################
# Fetch dataset
# --------------
# We download a list of left vs right button press contrasts from a
# localizer dataset.
n_subjects = 16
data = fetch_localizer_contrasts(["left vs right button press"], n_subjects,
                                 get_tmaps=True)

###########################################################################
# Display subject t_maps
# ----------------------
# We plot a grid with all the subjects t-maps thresholded at t = 2 for
# simple visualization purposes. The button press effect is visible among
# all subjects
from nilearn.image import crop_img
subjects = [subject_data[0] for subject_data in data['ext_vars']]

# fig, axes = plt.subplots(nrows=2, ncols=8, figsize=(16, 6))
for cidx, tmap in enumerate(data['tmaps']):
    plotting.plot_glass_brain(crop_img(tmap), colorbar=False, threshold=2.0,
                              # axes=axes[int(cidx / 8), int(cidx % 8)],
                              plot_abs=False, display_mode='z',
                              annotate=False)
    plt.savefig('/tmp/img_%02d.png' % cidx)
# plt.subplots_adjust(left=.01, right=.99, bottom=.01, hspace=.01, wspace=.01)
# fig.suptitle('subjects t_map left-right button press')
stop
plt.tight_layout(pad=0, rect=[-.05, -.05, 1.05, 1.05], h_pad=0, w_pad=0)
# fig.savefig('individuals.png', bbox_inches='tight')
plt.show()
"""

############################################################################
# Estimate second level model
# ---------------------------
# We define the input maps and the design matrix for the second level model
# and fit it.
second_level_input = data['cmaps']
design_matrix = pd.DataFrame([1] * len(second_level_input),
                             columns=['contrast'])

second_level_model = SecondLevelModel(smoothing_fwhm=8.0, n_jobs=-2)
second_level_model = second_level_model.fit(second_level_input,
                                            design_matrix=design_matrix)

#########################################################################
# Estimate uncorrected effects
# --------------------------------
# To estimate the contrast is very simple. We can just provide the column
# name of the design matrix.
z_map = second_level_model.compute_contrast(output_type='z_score')

#########################################################################
# We threshold the contrast at uncorrected p < 0.001 and plot
p_val = 0.001
z_th = norm.isf(p_val)
display = plotting.plot_glass_brain(z_map, threshold=z_th, colorbar=True,
                                    plot_abs=False, display_mode='z')

plt.savefig('result_onesample.png')
plotting.show()
"""

"""
#########################################################################
# Estimate FWE corrected effects with permutation test
# ----------------------------------------------------
# We use a similar function that also allows to tune the permutation
# parameters. For this example we will only use 1000 permutations. But we
# recommend to use 10000.
z_map = second_level_model.compute_contrast_permutations(
    output_type='cor_z_score', n_perm=1000)

#########################################################################
# We threshold the contrast at corrected p < 0.001 and plot
p_val = 0.001
z_th = norm.isf(p_val)
display = plotting.plot_glass_brain(z_map, threshold=z_th, colorbar=True,
                                    plot_abs=False, display_mode='z')
plotting.show()
"""