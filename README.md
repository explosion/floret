<a href="https://explosion.ai"><img src="https://explosion.ai/assets/img/logo.svg" width="125" height="125" align="right" /></a>

# floret: fastText + Bloom embeddings for compact, full-coverage vectors with spaCy

floret is an extended version of [fastText](https://fasttext.cc) that can
produce word representations for any word from a compact vector table. It
combines:

- fastText's subwords to provide embeddings for any word
- Bloom embeddings ("hashing trick") for a compact vector table

## Install floret

### Build floret from source

```bash
git clone https://github.com/explosion/floret
cd floret
make
```

This produces the main binary `floret`.

### Install for python

Install the python wrapper with `pip`:

```bash
pip install floret
```

Or install from source in developer mode:

```bash
git clone https://github.com/explosion/floret
cd floret
pip install -r requirements.txt
pip install --no-build-isolation --editable .
```

See the [python docs](python/README.md).

## Usage

`floret` adds two additional command line options to `fasttext`:

```
  -hashOnly           both word and char ngrams hashed only in buckets [false]
  -hashCount          with hashOnly: number of hashes (1-4) per word / subword [1]
```

With `-hashOnly`, the word entries are stored in the same table as the subword
embeddings (buckets), reducing the size of the saved vector data.

With `-hashCount 2`, each entry is stored as the sum of 2 rows in the internal
subwords hash table. `floret` supports 1-4 hashes per entry in the embeddings
table. By storing an entry in the embedding table as the sum of more than one
row, it is possible to greatly reduce the number of rows in the table with a
relatively small effect on the performance, both in terms of accuracy and
speed.

Here's how to train CBOW embeddings with subwords as 4-grams and 5-grams, 2
hashes per entry, and a compact table of 50K entries rather than the default of
2M entries.

```bash
floret cbow -dim 300 -minn 4 -maxn 5 -hashOnly -hashCount 2 -bucket 50000 \
-input input.txt -output vectors
```

With the `-hashOnly` option, floret will save an additional vector table with
the file ending `.floret`. The format is very similar to `.vec` with a header
line followed by one line per vector. The word tokens are replaced with the
index of the row and the header is extended to contain all the relevant
training settings needed to load this table in spaCy.

To import this vector table in [spaCy](https://spacy.io) v3.2+:

```bash
spacy init vectors --mode floret vectors.floret spacy_vectors_dir
```

## How floret works

In its original implementation, fastText stores words and subwords in two
separate tables. The word table contains one entry per word in the vocabulary
(typically ~1M entries) and the subwords are stored a separate fixed-size table
by hashing each subword into one row in the table (default 2M entries). A
relatively large table is used to reduce the number of collisions between
subwords. However, for 1M words + 2M subwords with 300-dimensional vectors of
32-bit floats, you'd need around 3GB to store the resulting data, which is
prohibitive for many use cases.

In addition, many libraries that import vectors only support the word table 
(`.vec`), which limits the coverage to words above a certain frequency in the
training data. For languages with rich morphology, even a large vector table
may not provide good coverage for words seen during training and you are still
likely to encounter words that were not seen at all during training.

In order to store word and subword vectors in a more compact format, we turn to
an algorithm that's been used by [spaCy](https://spacy.io) all along: Bloom
embeddings. Bloom embeddings (also called the "hashing trick", or known as
[`HashEmbed`](https://thinc.ai/docs/api-layers#hashembed) within spaCy's ML
library [thinc](https://thinc.ai)) can be used to store distinct
representations in a compact table by hashing each entry into multiple rows in
the table. By representing each entry as the sum of multiple rows, where it's
unlikely that two entries will collide on multiple hashes, most entries will
end up with a distinct representation.

With the settings `-minn 4 -maxn 5 -hashOnly -hashCount 2`, the embedding for
the word `apple` is stored internally as the sum of 2 hashed rows for each of
the word, 4-grams and 5-ngrams. The word is padded with the BOW and EOW
characters `<` and `>`, creating the following word and subword entries:

```
<apple>
<app
appl
pple
ple>
<appl
apple
pple>
```

For compatibility with spaCy,
[MurmurHash](https://github.com/aappleby/smhasher) is used to hash the word and
char ngram strings. The final embedding for `apple` is then the sum of two rows
(`-hashCount 2`) per word and char ngram above.

With `-hashOnly`, `floret` will save an additional vector table with the
ending `.floret` alongside the usual `.bin` and `.vec` files. The format is
very similar to `.vec` with a header line followed by one line per entry in the
vector table with the row index rather than a word token at the beginning of
each line. The header is extended to contain all the training settings required
to use this table in another application or library like spaCy.

The header contains the space-separated settings:

```none
bucket dim minn maxn hashCount hashSeed BOW EOW
```

A demo `.floret` table with `-bucket 10 -dim 10 -minn 2 -maxn3 -hashCount 2`:

```none
10 10 2 3 2 2166136261 < >
0 -2.2611 3.9302 2.6676 -11.233 0.093715 -10.52 -9.6463 -0.11853 2.101 -0.10145
1 -3.12 -1.7981 10.7 -6.171 4.4527 10.967 9.073 6.2056 -6.1199 -2.0402
2 9.5689 5.6721 -8.4832 -1.2249 2.1871 -3.0264 -2.391 -5.3308 -3.2847 -4.0382
3 3.6268 4.2759 -1.7007 1.5002 5.5266 1.8716 -12.063 0.26314 2.7645 2.4929
4 -11.683 -7.7068 2.1102 2.214 7.2202 0.69799 3.2173 -5.382 -2.0838 5.0314
5 -4.3024 8.0241 2.0714 -1.0174 -0.28369 1.7622 7.8797 -1.7795 6.7541 5.6703
6 8.3574 -5.225 8.6529 8.5605 -8.9465 3.767 -5.4636 -1.4635 -0.98947 -0.58025
7 -10.01 3.3894 -4.4487 1.1669 -11.904 6.5158 4.3681 0.79913 -6.9131 -8.687
8 -5.4576 7.1019 -8.8259 1.7189 4.955 -8.9157 -3.8905 -0.60086 -2.1233 5.892
9 8.0678 -4.4142 3.6236 4.5889 -2.7611 2.4455 0.67096 -4.2822 2.0875 4.6274
```

This table can be imported into a spaCy pipeline using `spacy init vectors` in
spaCy v3.2+ with the option `--floret-vectors`:

```bash
spacy init vectors --mode floret vectors.floret spacy_vectors_dir
```

## Notes

The fastText and floret binary formats (`.bin`) are not compatible, so it's
important to load a `.bin` file with the same program used to train it.

See the [fastText documentation](https://fasttext.cc) for details about all
other commands and options. `floret` supports all existing `fasttext` commands
and does not modify any `fasttext` defaults.

The original fastText README is provided below for reference.

---

# fastText README

[fastText](https://fasttext.cc/) is a library for efficient learning of word representations and sentence classification.

## Table of contents

- [Resources](#resources)
  - [Models](#models)
  - [Supplementary data](#supplementary-data)
  - [FAQ](#faq)
  - [Cheatsheet](#cheatsheet)
- [Requirements](#requirements)
- [Building fastText](#building-fasttext)
  - [Getting the source code](#getting-the-source-code)
  - [Building fastText using make (preferred)](#building-fasttext-using-make-preferred)
  - [Building fastText using cmake](#building-fasttext-using-cmake)
  - [Building fastText for Python](#building-fasttext-for-python)
- [Example use cases](#example-use-cases)
  - [Word representation learning](#word-representation-learning)
  - [Obtaining word vectors for out-of-vocabulary words](#obtaining-word-vectors-for-out-of-vocabulary-words)
  - [Text classification](#text-classification)
- [Full documentation](#full-documentation)
- [References](#references)
  - [Enriching Word Vectors with Subword Information](#enriching-word-vectors-with-subword-information)
  - [Bag of Tricks for Efficient Text Classification](#bag-of-tricks-for-efficient-text-classification)
  - [FastText.zip: Compressing text classification models](#fasttextzip-compressing-text-classification-models)

## Resources

### Models

- Recent state-of-the-art [English word vectors](https://fasttext.cc/docs/en/english-vectors.html).
- Word vectors for [157 languages trained on Wikipedia and Crawl](https://fasttext.cc/docs/en/crawl-vectors.html).
- Models for [language identification](https://fasttext.cc/docs/en/language-identification.html#content) and [various supervised tasks](https://fasttext.cc/docs/en/supervised-models.html#content).

### Supplementary data

- The preprocessed [YFCC100M data](https://fasttext.cc/docs/en/dataset.html#content) used in [2].

### FAQ

You can find [answers to frequently asked questions](https://fasttext.cc/docs/en/faqs.html#content) on our [website](https://fasttext.cc/).

### Cheatsheet

We also provide a [cheatsheet](https://fasttext.cc/docs/en/cheatsheet.html#content) full of useful one-liners.

## Requirements

We are continuously building and testing our library, CLI and Python bindings under various docker images using [circleci](https://circleci.com/).

Generally, **fastText** builds on modern Mac OS and Linux distributions.
Since it uses some C++11 features, it requires a compiler with good C++11 support.
These include :

- (g++-4.7.2 or newer) or (clang-3.3 or newer)

Compilation is carried out using a Makefile, so you will need to have a working **make**.
If you want to use **cmake** you need at least version 2.8.9.

One of the oldest distributions we successfully built and tested the CLI under is [Debian jessie](https://www.debian.org/releases/jessie/).

For the word-similarity evaluation script you will need:

- Python 2.6 or newer
- NumPy & SciPy

For the python bindings (see the subdirectory python) you will need:

- Python version 2.7 or >=3.4
- NumPy & SciPy
- [pybind11](https://github.com/pybind/pybind11)

One of the oldest distributions we successfully built and tested the Python bindings under is [Debian jessie](https://www.debian.org/releases/jessie/).

If these requirements make it impossible for you to use fastText, please open an issue and we will try to accommodate you.

## Building fastText

We discuss building the latest stable version of fastText.

### Getting the source code

You can find our [latest stable release](https://github.com/facebookresearch/fastText/releases/latest) in the usual place.

There is also the master branch that contains all of our most recent work, but comes along with all the usual caveats of an unstable branch. You might want to use this if you are a developer or power-user.

### Building fastText using make (preferred)

```
$ wget https://github.com/facebookresearch/fastText/archive/v0.9.2.zip
$ unzip v0.9.2.zip
$ cd fastText-0.9.2
$ make
```

This will produce object files for all the classes as well as the main binary `fasttext`.
If you do not plan on using the default system-wide compiler, update the two macros defined at the beginning of the Makefile (CC and INCLUDES).

### Building fastText using cmake

For now this is not part of a release, so you will need to clone the master branch.

```
$ git clone https://github.com/facebookresearch/fastText.git
$ cd fastText
$ mkdir build && cd build && cmake ..
$ make && make install
```

This will create the fasttext binary and also all relevant libraries (shared, static, PIC).

### Building fastText for Python

For now this is not part of a release, so you will need to clone the master branch.

```
$ git clone https://github.com/facebookresearch/fastText.git
$ cd fastText
$ pip install .
```

For further information and introduction see python/README.md

## Example use cases

This library has two main use cases: word representation learning and text classification.
These were described in the two papers [1](#enriching-word-vectors-with-subword-information) and [2](#bag-of-tricks-for-efficient-text-classification).

### Word representation learning

In order to learn word vectors, as described in [1](#enriching-word-vectors-with-subword-information), do:

```
$ ./fasttext skipgram -input data.txt -output model
```

where `data.txt` is a training file containing `UTF-8` encoded text.
By default the word vectors will take into account character n-grams from 3 to 6 characters.
At the end of optimization the program will save two files: `model.bin` and `model.vec`.
`model.vec` is a text file containing the word vectors, one per line.
`model.bin` is a binary file containing the parameters of the model along with the dictionary and all hyper parameters.
The binary file can be used later to compute word vectors or to restart the optimization.

### Obtaining word vectors for out-of-vocabulary words

The previously trained model can be used to compute word vectors for out-of-vocabulary words.
Provided you have a text file `queries.txt` containing words for which you want to compute vectors, use the following command:

```
$ ./fasttext print-word-vectors model.bin < queries.txt
```

This will output word vectors to the standard output, one vector per line.
This can also be used with pipes:

```
$ cat queries.txt | ./fasttext print-word-vectors model.bin
```

See the provided scripts for an example. For instance, running:

```
$ ./word-vector-example.sh
```

will compile the code, download data, compute word vectors and evaluate them on the rare words similarity dataset RW [Thang et al. 2013].

### Text classification

This library can also be used to train supervised text classifiers, for instance for sentiment analysis.
In order to train a text classifier using the method described in [2](#bag-of-tricks-for-efficient-text-classification), use:

```
$ ./fasttext supervised -input train.txt -output model
```

where `train.txt` is a text file containing a training sentence per line along with the labels.
By default, we assume that labels are words that are prefixed by the string `__label__`.
This will output two files: `model.bin` and `model.vec`.
Once the model was trained, you can evaluate it by computing the precision and recall at k (P@k and R@k) on a test set using:

```
$ ./fasttext test model.bin test.txt k
```

The argument `k` is optional, and is equal to `1` by default.

In order to obtain the k most likely labels for a piece of text, use:

```
$ ./fasttext predict model.bin test.txt k
```

or use `predict-prob` to also get the probability for each label

```
$ ./fasttext predict-prob model.bin test.txt k
```

where `test.txt` contains a piece of text to classify per line.
Doing so will print to the standard output the k most likely labels for each line.
The argument `k` is optional, and equal to `1` by default.
See `classification-example.sh` for an example use case.
In order to reproduce results from the paper [2](#bag-of-tricks-for-efficient-text-classification), run `classification-results.sh`, this will download all the datasets and reproduce the results from Table 1.

If you want to compute vector representations of sentences or paragraphs, please use:

```
$ ./fasttext print-sentence-vectors model.bin < text.txt
```

This assumes that the `text.txt` file contains the paragraphs that you want to get vectors for.
The program will output one vector representation per line in the file.

You can also quantize a supervised model to reduce its memory usage with the following command:

```
$ ./fasttext quantize -output model
```

This will create a `.ftz` file with a smaller memory footprint. All the standard functionality, like `test` or `predict` work the same way on the quantized models:

```
$ ./fasttext test model.ftz test.txt
```

The quantization procedure follows the steps described in [3](#fasttextzip-compressing-text-classification-models). You can
run the script `quantization-example.sh` for an example.

## Full documentation

Invoke a command without arguments to list available arguments and their default values:

```
$ ./fasttext supervised
Empty input or output path.

The following arguments are mandatory:
  -input              training file path
  -output             output file path

The following arguments are optional:
  -verbose            verbosity level [2]

The following arguments for the dictionary are optional:
  -minCount           minimal number of word occurrences [1]
  -minCountLabel      minimal number of label occurrences [0]
  -wordNgrams         max length of word ngram [1]
  -bucket             number of buckets [2000000]
  -minn               min length of char ngram [0]
  -maxn               max length of char ngram [0]
  -t                  sampling threshold [0.0001]
  -label              labels prefix [__label__]

The following arguments for training are optional:
  -lr                 learning rate [0.1]
  -lrUpdateRate       change the rate of updates for the learning rate [100]
  -dim                size of word vectors [100]
  -ws                 size of the context window [5]
  -epoch              number of epochs [5]
  -neg                number of negatives sampled [5]
  -loss               loss function {ns, hs, softmax} [softmax]
  -thread             number of threads [12]
  -pretrainedVectors  pretrained word vectors for supervised learning []
  -saveOutput         whether output params should be saved [0]

The following arguments for quantization are optional:
  -cutoff             number of words and ngrams to retain [0]
  -retrain            finetune embeddings if a cutoff is applied [0]
  -qnorm              quantizing the norm separately [0]
  -qout               quantizing the classifier [0]
  -dsub               size of each sub-vector [2]
```

Defaults may vary by mode. (Word-representation modes `skipgram` and `cbow` use a default `-minCount` of 5.)

## References

Please cite [1](#enriching-word-vectors-with-subword-information) if using this code for learning word representations or [2](#bag-of-tricks-for-efficient-text-classification) if using for text classification.

### Enriching Word Vectors with Subword Information

[1] P. Bojanowski\*, E. Grave\*, A. Joulin, T. Mikolov, [_Enriching Word Vectors with Subword Information_](https://arxiv.org/abs/1607.04606)

```
@article{bojanowski2017enriching,
  title={Enriching Word Vectors with Subword Information},
  author={Bojanowski, Piotr and Grave, Edouard and Joulin, Armand and Mikolov, Tomas},
  journal={Transactions of the Association for Computational Linguistics},
  volume={5},
  year={2017},
  issn={2307-387X},
  pages={135--146}
}
```

### Bag of Tricks for Efficient Text Classification

[2] A. Joulin, E. Grave, P. Bojanowski, T. Mikolov, [_Bag of Tricks for Efficient Text Classification_](https://arxiv.org/abs/1607.01759)

```
@InProceedings{joulin2017bag,
  title={Bag of Tricks for Efficient Text Classification},
  author={Joulin, Armand and Grave, Edouard and Bojanowski, Piotr and Mikolov, Tomas},
  booktitle={Proceedings of the 15th Conference of the European Chapter of the Association for Computational Linguistics: Volume 2, Short Papers},
  month={April},
  year={2017},
  publisher={Association for Computational Linguistics},
  pages={427--431},
}
```

### FastText.zip: Compressing text classification models

[3] A. Joulin, E. Grave, P. Bojanowski, M. Douze, H. Jégou, T. Mikolov, [_FastText.zip: Compressing text classification models_](https://arxiv.org/abs/1612.03651)

```
@article{joulin2016fasttext,
  title={FastText.zip: Compressing text classification models},
  author={Joulin, Armand and Grave, Edouard and Bojanowski, Piotr and Douze, Matthijs and J{\'e}gou, H{\'e}rve and Mikolov, Tomas},
  journal={arXiv preprint arXiv:1612.03651},
  year={2016}
}
```

(\* These authors contributed equally.)
