PyRPCA
======

Robust PCA in Python. Methods are from the http://perception.csl.illinois.edu/matrix-rank/sample_code.html and papers therein.

Requirement
===========
 * scipy
 * numpy
 * propack(optional)
 * scikit-learn

Scripts
=======
 * `test_robustpca.py` test whether the algorithms included can recovery the synthetic data successfully. Use `nosetest test_robustpca.py`
 * `plot_benchmark.py` plot the benchmarks with synthetic data generated with different parameters. Use `python2 plot_benchmark.py`
 * `background_subtraction.py` generate the result using the escalator dataset. Use `python2 background_subtraction.py`. This will generate the `.mat` files with respect to each algorithms and can be directly readable from matlab. Furthermore, `background_subtraction_visualize.py` could be used to generate a video. The temporary image files are located in `/tmp/robust_pca_tmp/` which should be created first.
 * `topic_extraction.py` extracts the keywords from the 20newsgroup dataset. It will generate two files, one is `origin.txt` and another is `keyword.txt`. The keyword and the original text on the same line is one-one mapped.

Aknowledgement
==============
Special thanks for the follow two resources and their authors.
 * http://perception.csl.illinois.edu/matrix-rank/sample_code.html
 * www.stanford.edu/~peleato/math301_slides.pdf‎