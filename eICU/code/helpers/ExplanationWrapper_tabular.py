import torch
import numpy as np
import gc
import quantus
import random
import copy
from typing import Any, Callable, Sequence, Tuple, Union, Optional
from quantus.helpers.utils import expand_indices


# Plotting specifics.
from matplotlib.patches import Circle, RegularPolygon
from matplotlib.path import Path
from matplotlib.projections.polar import PolarAxes
from matplotlib.projections import register_projection
from matplotlib.spines import Spine
from matplotlib.transforms import Affine2D

from captum.attr import (
    IntegratedGradients,
    DeepLift,
    KernelShap,
    LRP,
    Lime,
    visualization)
    
from captum._utils.models.linear_model import SkLearnLinearRegression, SkLearnLasso
from quantus.helpers.utils import expand_indices

""" Wrappers around Captum Explanations """


def explainer_wrapper(**kwargs):
    """Wrapper for explainer functions."""
    if kwargs["method"] == "Lime":
        return lime_explainer(**kwargs)
    elif kwargs["method"] == "KernelShap":
        return kernelshap_explainer(**kwargs)
    elif kwargs["method"] == "LRP":
        return lrp_explainer(**kwargs)
    elif kwargs["method"] == "DeepLIFT":
        return deeplift_explainer(**kwargs)
    elif kwargs["method"] == "IntegratedGradients":
        return intgrad_explainer(**kwargs)
    elif kwargs["method"] == "Random":
        return random_explainer(**kwargs)
    else:
        raise ValueError("Pick an explaination function that exists.")


def lime_explainer(
    model, inputs, abs=False, normalise=False, *args, **kwargs
) -> np.array:
    """Wrapper aorund captum's LIME implementation."""

    gc.collect()
    torch.cuda.empty_cache()

    # Set model in evaluate mode.
    model.to(kwargs.get("device", None))
    model.eval()
    
    if not isinstance(inputs, torch.Tensor):
        inputs = (
            torch.Tensor(inputs)
        )

    inputs = inputs.to(kwargs.get("device", None))
    baseline = kwargs.get("baseline", torch.zeros_like(inputs)).to(kwargs.get("device", None))

    assert (
        len(np.shape(inputs)) == 3
    ), "Inputs should be shaped (nr_samples, nr_channels, nr_features) e.g., (16, 1, 20)."

    explanation = (
        Lime(model,interpretable_model = SkLearnLasso(alpha=0.001))
        .attribute(
            inputs=inputs,
            baselines=baseline,
            n_samples=kwargs.get("n_samples",20),
        )
        .sum(axis=1)
        .cpu()
        .data
    )

    gc.collect()
    torch.cuda.empty_cache()

    if normalise:
        explanation = quantus.normalise_func.normalise_by_negative(explanation)

    if isinstance(explanation, torch.Tensor):
        if explanation.requires_grad:
            explanation = explanation.cpu().detach().numpy()
        explanation = explanation.cpu().numpy()
    
    if(explanation == 0).all():
        explanation = np.random.normal(0,0.000001,explanation.shape)     # to avoid numerical issues in case Lime attributs all zeros
    
    return explanation

##########################################################################

def kernelshap_explainer(
    model, inputs, abs=False, normalise=False, *args, **kwargs
) -> np.array:
    """Wrapper aorund captum's KernelShap implementation."""

    gc.collect()
    torch.cuda.empty_cache()

    # Set model in evaluate mode.
    model.to(kwargs.get("device", None))
    model.eval()
    
    if not isinstance(inputs, torch.Tensor):
        inputs = (
            torch.Tensor(inputs)
        )

    inputs = inputs.to(kwargs.get("device", None))
    baseline = kwargs.get("baseline", torch.zeros_like(inputs)).to(kwargs.get("device", None))

    assert (
        len(np.shape(inputs)) == 3
    ), "Inputs should be shaped (nr_samples, nr_channels, nr_features) e.g., (16, 1, 20)."

    explanation = (
        KernelShap(model)
        .attribute(
            inputs=inputs,
            baselines=baseline,
            n_samples=kwargs.get("n_samples",30),
        )
        .sum(axis=1)
        .cpu()
        .data
    )

    gc.collect()
    torch.cuda.empty_cache()

    if normalise:
        explanation = quantus.normalise_func.normalise_by_negative(explanation)

    if isinstance(explanation, torch.Tensor):
        if explanation.requires_grad:
            return explanation.cpu().detach().numpy()
        return explanation.cpu().numpy()

    return explanation

##########################################################################




def lrp_explainer(
    model, inputs, abs=False, normalise=False, *args, **kwargs
) -> np.array:
    """Wrapper aorund captum's LRP implementation."""

    gc.collect()
    torch.cuda.empty_cache()

    # Set model in evaluate mode.
    model.to(kwargs.get("device", None))
    model.eval()
    
    if not isinstance(inputs, torch.Tensor):
        inputs = (
            torch.Tensor(inputs)
        )

    inputs = inputs.to(kwargs.get("device", None))

    assert (
        len(np.shape(inputs)) == 3
    ), "Inputs should be shaped (nr_samples, nr_channels, nr_features) e.g., (16, 1, 20)."

    explanation = (
        LRP(model)
        .attribute(
            inputs=inputs,
        )
        .sum(axis=1)
        .cpu()
        .data
    )

    gc.collect()
    torch.cuda.empty_cache()

    if normalise:
        explanation = quantus.normalise_func.normalise_by_negative(explanation)

    if isinstance(explanation, torch.Tensor):
        if explanation.requires_grad:
            return explanation.cpu().detach().numpy()
        return explanation.cpu().numpy()

    return explanation

##########################################################################


def deeplift_explainer(
    model, inputs, abs=False, normalise=False, *args, **kwargs
) -> np.array:
    """Wrapper aorund captum's DeepLIFT implementation."""

    gc.collect()
    torch.cuda.empty_cache()

    # Set model in evaluate mode.
    model.to(kwargs.get("device", None))
    model.eval()
    
    if not isinstance(inputs, torch.Tensor):
        inputs = (
            torch.Tensor(inputs)
        )
    
    inputs = inputs.to(kwargs.get("device", None))
    baseline = kwargs.get("baseline", torch.zeros_like(inputs)).to(kwargs.get("device", None))

    assert (
        len(np.shape(inputs)) == 3
    ), "Inputs should be shaped (nr_samples, nr_channels, nr_features) e.g., (16, 1, 20)."

    explanation = (
        DeepLift(model)
        .attribute(
            inputs=inputs,
            baselines = baseline,
        )
        .sum(axis=1)
        .cpu()
        .data
    )

    gc.collect()
    torch.cuda.empty_cache()

    if normalise:
        explanation = quantus.normalise_func.normalise_by_negative(explanation)

    if isinstance(explanation, torch.Tensor):
        if explanation.requires_grad:
            return explanation.cpu().detach().numpy()
        return explanation.cpu().numpy()

    return explanation

##########################################################################



def intgrad_explainer(
    model, inputs, abs=False, normalise=False, *args, **kwargs
) -> np.array:
    """Wrapper aorund captum's Integrated Gradients implementation."""

    gc.collect()
    torch.cuda.empty_cache()

    # Set model in evaluate mode.
    model.to(kwargs.get("device", None))
    model.eval()
    
    
    if not isinstance(inputs, torch.Tensor):
        inputs = (
            torch.Tensor(inputs)
        )

    inputs = inputs.to(kwargs.get("device", None))
    baseline = kwargs.get("baseline", torch.zeros_like(inputs)).to(kwargs.get("device", None))

    assert (
        len(np.shape(inputs)) == 3
    ), "Inputs should be shaped (nr_samples, nr_channels, nr_features) e.g., (16, 1, 20)."


    explanation = (
        IntegratedGradients(model)
        .attribute(
            inputs=inputs,
            baselines=baseline,
            n_steps=kwargs.get("n_steps",30),
            method="riemann_trapezoid",
        )
        .sum(axis=1)
        .cpu()
        .data
    )

    gc.collect()
    torch.cuda.empty_cache()

    if normalise:
        explanation = quantus.normalise_func.normalise_by_negative(explanation)

    if isinstance(explanation, torch.Tensor):
        if explanation.requires_grad:
            return explanation.cpu().detach().numpy()
        return explanation.cpu().numpy()

    return explanation


##########################################################################

def random_explainer(
    model, inputs, abs=False, normalise=False, *args, **kwargs
) -> np.array:
    """Wrapper aorund captum's LRP implementation."""

    gc.collect()
    torch.cuda.empty_cache()

    # Set model in evaluate mode.
    model.to(kwargs.get("device", None))
    model.eval()
    
    if not isinstance(inputs, torch.Tensor):
        inputs = (
            torch.Tensor(inputs)
        )

    inputs = inputs.to(kwargs.get("device", None))

    assert (
        len(np.shape(inputs)) == 3
    ), "Inputs should be shaped (nr_samples, nr_channels, nr_features) e.g., (16, 1, 20)."

    explanation = (
        LRP(model)
        .attribute(
            inputs=inputs,
        )
        .sum(axis=1)
        .cpu()
        .data
    )

    explanation = np.random.randn(*explanation.shape)
    gc.collect()
    torch.cuda.empty_cache()

    if normalise:
        explanation = quantus.normalise_func.normalise_by_negative(explanation)

    if isinstance(explanation, torch.Tensor):
        if explanation.requires_grad:
            return explanation.cpu().detach().numpy()
        return explanation.cpu().numpy()

    return explanation





   
    
def custom_perturbation_func(
    arr: np.array,
    indices: Tuple[slice, ...],  # Alt. Union[int, Sequence[int], Tuple[np.array]],
    indexed_axes: Sequence[int],
    **kwargs,
) -> np.array:
    """
    Application for categorical data.
    Allows sampling of data points that have a similar category values as the instance of interest.
    ----------
    arr: np.ndarray
         Array to be perturbed.
    indices: int, sequence, tuple
        Array-like, with a subset shape of arr.
    indexed_axes: sequence
        The dimensions of arr that are indexed.
        These need to be consecutive, and either include the first or last dimension of array.      
    kwargs: optional
        Keyword arguments.
    Returns
    -------
    arr_perturbed: np.ndarray
         The array which some of its indices have been perturbed.
    """
    
    perturb_std = kwargs.get('perturb_std',0.1)
    # Determines the size of sample range. Should be within [0,1].
    # 0.1 means one samples values that lie within the -0.1/+0.1 'quantile' around the original value.
    # The higher, the larger the possible distance from the original value
    
    use_baseline = kwargs.get('use_baseline',False)
    #indicates whether perturbed values should be sampled from entire data domain or only from specific baseline

    sets = kwargs.get('sets')
    len_sets = kwargs.get('len_sets')
    baseline = kwargs.get('baseline')
    
    
    indices = expand_indices(arr, indices, indexed_axes)

    orig_idx_in_set = np.asarray([np.where(sets[i]==arr[0][i]) for i in range(20)]).reshape(20)
    new_idx0 = orig_idx_in_set.copy() 

    sample_widths = np.ceil(perturb_std*len_sets).astype(int)
    move_steps = [] # obtain number of steps we move away from original index
    
    for i in range(20):
      r = [s for s in range(-sample_widths[i],sample_widths[i] + 1,1)]
      move_steps.append(random.choice(r))

    new_idx0 = new_idx0 + move_steps
    new_idx = np.clip(new_idx0,0,len_sets-1)  #ensure valid index
    all_perturbed = np.asarray([sets[i][new_idx[i]] for i in range(20)]).reshape((1,20))


    arr_perturbed = copy.copy(arr)
    
    if use_baseline:
        baseline = baseline.reshape((1,20))
        arr_perturbed[indices] = baseline[indices]
    else:
        arr_perturbed[indices] = all_perturbed[indices]
    

    return arr_perturbed
    
    

def radar_factory(num_vars, frame='circle'):
    """Create a radar chart with `num_vars` axes.

    This function creates a RadarAxes projection and registers it.

    Parameters
    ----------
    num_vars : int
        Number of variables for radar chart.
    frame : {'circle' | 'polygon'}
        Shape of frame surrounding axes.
    """
    # calculate evenly-spaced axis angles
    theta = np.linspace(0, 2*np.pi, num_vars, endpoint=False)

    class RadarAxes(PolarAxes):

        name = 'radar'

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            # rotate plot such that the first axis is at the top
            self.set_theta_zero_location('N')

        def fill(self, *args, closed=True, **kwargs):
            """Override fill so that line is closed by default."""
            return super().fill(closed=closed, *args, **kwargs)

        def plot(self, *args, **kwargs):
            """Override plot so that line is closed by default."""
            lines = super().plot(*args, **kwargs)
            for line in lines:
                self._close_line(line)

        def _close_line(self, line):
            x, y = line.get_data()
            # FIXME: markers at x[0], y[0] get doubled-up
            if x[0] != x[-1]:
                x = np.concatenate((x, [x[0]]))
                y = np.concatenate((y, [y[0]]))
                line.set_data(x, y)

        def set_varlabels(self, labels, angles=None):
            self.set_thetagrids(angles=np.degrees(theta), labels=labels)

        def _gen_axes_patch(self):
            # The Axes patch must be centered at (0.5, 0.5) and of radius 0.5
            # in axes coordinates.
            if frame == 'circle':
                return Circle((0.5, 0.5), 0.5)
            elif frame == 'polygon':
                return RegularPolygon((0.5, 0.5), num_vars,
                                      radius=.5, edgecolor="k")
            else:
                raise ValueError("unknown value for 'frame': %s" % frame)

        def draw(self, renderer):
            """ Draw. If frame is polygon, make gridlines polygon-shaped."""
            if frame == 'polygon':
                gridlines = self.yaxis.get_gridlines()
                for gl in gridlines:
                    gl.get_path()._interpolation_steps = num_vars
            super().draw(renderer)


        def _gen_axes_spines(self):
            if frame == 'circle':
                return super()._gen_axes_spines()
            elif frame == 'polygon':
                # spine_type must be 'left'/'right'/'top'/'bottom'/'circle'.
                spine = Spine(axes=self,
                              spine_type='circle',
                              path=Path.unit_regular_polygon(num_vars))
                # unit_regular_polygon gives a polygon of radius 1 centered at
                # (0, 0) but we want a polygon of radius 0.5 centered at (0.5,
                # 0.5) in axes coordinates.
                spine.set_transform(Affine2D().scale(.5).translate(.5, .5)
                                    + self.transAxes)

                return {'polar': spine}
            else:
                raise ValueError("unknown value for 'frame': %s" % frame)

    register_projection(RadarAxes)
    return theta
    
    
    
    
