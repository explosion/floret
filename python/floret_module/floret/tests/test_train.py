import pytest
from pathlib import Path
import fasttext
import floret
from numpy.testing import assert_almost_equal


@pytest.mark.parametrize(
    "mode,value", [("fasttext", 0.0033344), ("floret", -0.00057555)]
)
def test_train_unsupervised_fasttext(mode, value):
    data_path = Path(__file__).parent / "test_data" / "data.txt"
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

    assert_almost_equal(model.get_word_vector("the")[0], value)

    # compare floret fasttext mode to original fasttext module
    if mode == "fasttext":
        fasttext_model = fasttext.train_unsupervised(
            "test_data/data.txt",
            model="cbow",
            bucket=100,
            minn=3,
            maxn=6,
            minCount=1,
            thread=1,
        )
        assert_almost_equal(fasttext_model.get_word_vector("the")[0], value)
