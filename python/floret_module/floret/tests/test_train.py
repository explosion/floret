import pytest
from pathlib import Path
import floret
import sys
from numpy.testing import assert_almost_equal


@pytest.mark.parametrize(
    "mode,value", [("fasttext", 0.0033344), ("floret", -0.00057555)]
)
def test_train_unsupervised_fasttext(mode, value):
    data_path = Path(__file__).parent / "data.txt"
    model = floret.train_unsupervised(
        str(data_path),
        model="cbow",
        mode=mode,
        hashCount=2,
        bucket=100,
        minn=3,
        maxn=6,
        minCount=1,
        thread=1,
    )

    the_zero = model.get_word_vector("the")[0]

    # this seems unexpected
    if sys.platform.startswith("linux"):
        assert_almost_equal(the_zero, value)

    model.save_model(f"test_model_{mode}.bin")
    model.save_vectors(f"test_model_{mode}.vectors")
    model.save_floret_vectors(f"test_model_{mode}.floret")

    model2 = floret.load_model(f"test_model_{mode}.bin")
    assert model.get_word_vector("the")[0] == model2.get_word_vector("the")[0]


@pytest.mark.parametrize(
    "mode,value", [("fasttext", 0.0033344)]
)
def test_train_unsupervised_fasttext_compare(mode, value):
    fasttext = pytest.importorskip("fasttext")
    data_path = Path(__file__).parent / "data.txt"
    model = floret.train_unsupervised(
        str(data_path),
        model="cbow",
        mode=mode,
        hashCount=2,
        bucket=100,
        minn=3,
        maxn=6,
        minCount=1,
        thread=1,
    )

    the_zero = model.get_word_vector("the")[0]

    # this seems unexpected
    if sys.platform.startswith("linux"):
        assert_almost_equal(the_zero, value)
    # compare floret fasttext mode to original fasttext module
    if mode == "fasttext":
        fasttext_model = fasttext.train_unsupervised(
            str(data_path),
            model="cbow",
            bucket=100,
            minn=3,
            maxn=6,
            minCount=1,
            thread=1,
        )
        assert_almost_equal(fasttext_model.get_word_vector("the")[0], the_zero)
