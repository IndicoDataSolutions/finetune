import glob
import os

import numpy as np
import pytest
import tqdl

from finetune import SequenceLabeler


def test_model_loads_and_preds(save_model_dir):
    path = save_model_dir / "ner_backwards_compatibility.jl"
    tqdl.download(
        "https://s3.amazonaws.com/bendropbox/ner_backwards_compatibility.jl",
        str(path),
    )
    model = SequenceLabeler.load(path)
    preds = model.predict(
        [
            "This is a test sequence about Microsoft Corp, Apple Inc and Indico Data Solutions. "
            * 10
        ]
    )
    assert len(preds[0]) == 30
    acceptable_preds = set(["Microsoft Corp", "Apple Inc", "Indico Data Solutions"])
    for pred in preds[0]:
        assert pred["text"] in acceptable_preds


def test_roberta_collapse_whitespace_old_default(get_untrained_sequence_labeler):
    model = get_untrained_sequence_labeler(collapse_whitespace=True, version="0.8.6")
    assert model.config.collapse_whitespace == True
    model.fit(["test"], [[]])
    model.save("./test_model.jl")
    model = SequenceLabeler.load("./test_model.jl")
    assert model.config.collapse_whitespace == False


def test_roberta_default_no_change(get_untrained_sequence_labeler):
    model = get_untrained_sequence_labeler(collapse_whitespace=True)
    assert model.config.collapse_whitespace == True
    model.fit(["test"], [[]])
    model.save("./test_model.jl")
    model = SequenceLabeler.load("./test_model.jl")
    assert model.config.collapse_whitespace == True


# TODO: eventually clean this up and push these files to s3.
BUNDLES = glob.glob(os.path.join("/Finetune/tests/backwards_compat_bundles/*.jl"))


def handle_length_mismatch(a, b, atol=0, rtol=0):
    if len(b) > len(a):
        # Swap a and b so that a is the longer list.
        a, b = b, a
    for i, ai in enumerate(a):
        if i >= len(b):
            raise ValueError(
                f"Length mismatch - First differing element: {ai} does not have a corresponding element in b"
            )
        try:
            nested_assert_allclose(ai, b[i], atol=atol, rtol=rtol)
        except AssertionError:
            raise ValueError(
                f"Length mismatch - First differing element: {ai} and {b[i]}"
            )
    raise ValueError(
        "The lists are not the same length.. and also there is a problem with handle length mismatch function. "
    )


def nested_assert_allclose(a, b, atol=0, rtol=0):
    assert type(a) == type(b)
    if isinstance(a, list):
        if len(a) != len(b):
            handle_length_mismatch(a, b, atol=atol, rtol=rtol)
        for a_i, b_i in zip(a, b):
            nested_assert_allclose(a_i, b_i, atol=atol, rtol=rtol)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        try:
            for k in a.keys():
                nested_assert_allclose(a[k], b[k], atol=atol, rtol=rtol)
        except AssertionError:
            raise ValueError(f"Dict mismatch: {a} != {b}")
    elif isinstance(a, (np.ndarray, float)):
        # This might be too leniant. But going to do a first pass to make sure nothing is horrendously wrong.
        # and go from there.
        # TODO; Make this less leniant.
        np.testing.assert_allclose(a, b, atol=atol, rtol=rtol)
    else:
        assert a == b


@pytest.mark.skip(
    reason="Skipping backwards compat tests. They are expensive but should be run manually as and when required."
)
@pytest.mark.parametrize("bundle_path", BUNDLES)
def test_backwards_compat_extreme(bundle_path):
    model = SequenceLabeler.load(bundle_path, key="model")
    texts = SequenceLabeler.load(bundle_path, key="texts")
    contexts = SequenceLabeler.load(bundle_path, key="contexts")

    expected_flat_features = SequenceLabeler.load(bundle_path, key="flat_features")
    expected_sequence_features = SequenceLabeler.load(
        bundle_path, key="sequence_features"
    )
    expected_preds = SequenceLabeler.load(bundle_path, key="preds")
    expected_probs = SequenceLabeler.load(bundle_path, key="probs")

    try:
        flat_features = model.featurize(texts, context=contexts)
    except:
        if expected_flat_features is not None:
            raise
        flat_features = None
    if expected_flat_features is not None:
        # Allow 1% relative difference and 1e-4 absolute difference. Difficult to know what we actually need here
        # To be successful, but as long as the preds are the same, this is mostly just for us to build confidence.
        nested_assert_allclose(
            flat_features,
            expected_flat_features,
            atol=5e-2 if model.config.float_16_predict else 1e-4,
            rtol=1e-2,
        )

    try:
        sequence_features = model.featurize_sequence(texts, context=contexts)
    except:
        if expected_sequence_features is not None:
            raise
        sequence_features = None
    if expected_sequence_features is not None:
        nested_assert_allclose(
            sequence_features,
            expected_sequence_features,
            atol=5e-2 if model.config.float_16_predict else 1e-4,
            rtol=1e-2,
        )

    try:
        preds = model.predict(texts, context=contexts)
    except:
        if expected_preds is not None:
            raise
        preds = None
    if expected_preds is not None:
        # Slightly looser atol on here, but as long as the preds are the same nobody is going to care about
        # 1% change in probas.
        nested_assert_allclose(
            preds,
            expected_preds,
            atol=5e-2 if model.config.float_16_predict else 1e-2,
        )

    try:
        probs = model.predict_proba(texts, context=contexts)
    except:
        if expected_probs is not None:
            raise
        probs = None
    if expected_probs is not None:
        nested_assert_allclose(
            probs,
            expected_probs,
            atol=5e-2 if model.config.float_16_predict else 1e-2,
        )
