<a href="https://explosion.ai"><img src="https://explosion.ai/assets/img/logo.svg" width="125" height="125" align="right" /></a>

# floret: fastText + Bloom embeddings for compact, full-coverage vectors with spaCy

floret is an extended version of [fastText](https://fasttext.cc) that can
produce word representations for any word from a compact vector table. It
combines:

- fastText's subwords to provide embeddings for any word
- Bloom embeddings ("hashing trick") for a compact vector table

## Installation

```bash
pip install floret
```

## Usage

Train floret vectors using the options:

- `mode`: `"floret"`, storing both words and subwords in the same compact hash
  table
- `hashCount`: store each entry in 1-4 rows in the hash table (recommended:
  `2`)
- `bucket`: in combination with `hashCount>1`, the size of the hash table can
  be greatly reduced (recommended: `25000`--`100000`, reduced from the fastText
  default of `2000000`)
- `minn`: min length of char ngram (default: `3`)
- `maxn`: max length of char ngram (default: `6`)

```python
import floret

# train vectors
model = floret.train_unsupervised(
    "data.txt",
    model="cbow",
    mode="floret",
    hashCount=2,
    bucket=50000,
    minn=3,
    maxn=6,
)

# query vector
model.get_word_vector("broccoli")

# save full model
model.save_model("vectors.bin")

# export standard word-only vector table
model.save_vectors("vectors.vec")

# export floret vector table
model.save_floret_vectors("vectors.floret")
```

**Note:** with the default setting `mode="fasttext"`, `floret` trains original
fastText vectors.

## Use floret vectors in spaCy

Import floret vectors into spaCy v3.2+:

```bash
spacy init vectors --mode floret vectors.floret spacy_vectors_model
```

## Notes

`floret` contains all features of the original [`fasttext`
module](https://pypi.org/project/fasttest). See the [`fasttext`
docs](https://fasttext.cc/docs/en/python-module.html) for more information.

The `fasttext` and `floret` binary formats saved with
`model.save_model("model.bin")` are not compatible.
