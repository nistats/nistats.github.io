"""Studying firts-level-model details in a trials-and-error fashion
================================================================

In this tutorial, we study the parametrization of the first-level
model used for fMRI data analysis and clarify their impact on the
results of the analysis.

We use an exploratory approach, in which we incrementally include some
new features in the analysis and look at the outcome, i.e. the
resulting brain maps.

Readers without prior experience in fMRI data analysis should first
run the plot_sing_subject_single_run tutorial to get a bit more
familiar with the base concepts, and only then run thi script.

To run this example, you must launch IPython via ``ipython
--matplotlib`` in a terminal, or use ``jupyter-notebook``.

.. contents:: **Contents**
    :local:
    :depth: 1

"""

import numpy as np
import pandas as pd
from nilearn import plotting

from nistats import datasets

###############################################################################
# Retrieving the data
# -------------------
#
# We use a so-called localizer dataset, which consists in a 5-minutes
# acquisition of a fast event-related dataset.

data = datasets.fetch_localizer_first_level()
t_r = 2.4
paradigm_file = data.paradigm
events= pd.read_csv(paradigm_file, sep=' ', header=None, index_col=None)
events.columns = ['session', 'trial_type', 'onset']
fmri_img = data.epi_img

###############################################################################
# Running a basic model
# ---------------------

from nistats.first_level_model import FirstLevelModel
first_level_model = FirstLevelModel(t_r)
first_level_model = first_level_model.fit(fmri_img, events=events)
design_matrix = first_level_model.design_matrices_[0]

from nistats.reporting import plot_design_matrix
plot_design_matrix(design_matrix)

#########################################################################
# Specify the contrasts.
# 
# For this, let's create a function that, given the deisgn matrix,
# generates the corresponding contrasts.
# This will be useful

def make_localizer_contrasts(design matrix):
    """ returns a dictionary of four contasts, given the design matrix"""
    # first generate canonical contrasts 
    contrast_matrix = np.eye(design_matrix.shape[1])
    contrasts = dict([(column, contrast_matrix[i])
                      for i, column in enumerate(design_matrix.columns)])
    # Add more complex contrasts
    contrasts["audio"] = contrasts["clicDaudio"] + contrasts["clicGaudio"] +\
                         contrasts["calculaudio"] + contrasts["phraseaudio"]
    contrasts["video"] = contrasts["clicDvideo"] + contrasts["clicGvideo"] + \
                         contrasts["calculvideo"] + contrasts["phrasevideo"]
    contrasts["computation"] = contrasts["calculaudio"] + contrasts["calculvideo"]
    contrasts["sentences"] = contrasts["phraseaudio"] + contrasts["phrasevideo"]

    #########################################################################
    # Short list of more relevant contrasts
    contrasts = {
        "left-right": (contrasts["clicGaudio"] + contrasts["clicGvideo"]
                       - contrasts["clicDaudio"] - contrasts["clicDvideo"]),
        "H-V": contrasts["damier_H"] - contrasts["damier_V"],
        "audio-video": contrasts["audio"] - contrasts["video"],
        "computation-sentences": (contrasts["computation"] -
                                  contrasts["sentences"]),
    }
    return contrasts

contrasts = make_localizer_contrasts(design_matrix)

#########################################################################
# contrast estimation

fig = plt.figure(figsize=(8, 2))
for index, (contrast_id, contrast_val) in enumerate(contrasts.items()):
    ax = plt.subplot(1, len(contrasts), 1 + index)
    z_map = first_level_model.compute_contrast(
        contrast_val, output_type='z_score')
    plotting.plot_stat_map(
        z_map, display_mode='z', threshold=3.0, title=contrast_id, axes=ax,
        cut_coords=1)

plotting.show()

#########################################################################
# Null drift model

first_level_model = FirstLevelModel(t_r, drift_model=None)
first_level_model = first_level_model.fit(fmri_img, events=events)
design_matrix = first_level_model.design_matrices_[0]
plot_design_matrix(design_matrix)
contrasts = make_localizer_contrasts(design_matrix)
fig = plt.figure(figsize=(8, 2))
for index, (contrast_id, contrast_val) in enumerate(contrasts.items()):
    ax = plt.subplot(1, len(contrasts), 1 + index)
    z_map = first_level_model.compute_contrast(
        contrast_val, output_type='z_score')
    plotting.plot_stat_map(
        z_map, display_mode='z', threshold=3.0, title=contrast_id, axes=ax,
        cut_coords=1)

plotting.show()
