# Report

In order to more efficiently analyze the results of exploring the
search space, Google's program was tweaked to return the accuracy for
the testing set after running.  It was then run inside a series of
nested Bash `for ... in` loops, with the parameters passed in via
command-line arguments using Tensorflow's built-in
`tf.app.flags.DEFINE_foo` feature, where `foo` is a type (*e.g.*,
`float`, `integer`, or `string`).

Parameters will be listed in (max step, batch size, length of hidden
vector 1, length of hidden vector 2) order.


## Naive sampling of entire search space

All 48 combinations of the following were tried:


| MS   | BS   | H1  | H2 |
|------|------|-----|----|
| 2000 | 100  | 32  | 8  |
| 4000 | 1000 | 64  | 16 |
|      |      | 128 | 32 |
|      |      |     | 64 |


In general, the larger values performed better than the smaller ones,
although for H1 the effect was absent.  The difference between the
different parameter combinations was not especially pronounced: the
best run had 92.34% correct, while the worst was 86.95%, in spite of
the not insignificant differences in training time.


The highest accuracy value came from (4000, 1000, 128, 64), ever so
slightly edging out (by a difference of 0.0004, or two images) the run
with a tenth the batch size, which took about a third of the time to
run.  Thus, when choosing a batch size, since matching an extra few
images is generally not worth the extra fortyish seconds the model
would take to train (as compared to 18 for the smaller batch size
(BS=1000 takes about a minute)).


One trend made clear by sorting the runs by their accuracy numbers is
that the larger maximum step was almost universally superior to the
smaller, to the point that the top 21 scores were all MS=4k.  Number
22 was (2000, 1000, 128, 64), representing the best (and most
computationally-expensive) of the MS=2k runs.


## Targeted search of different MS values

All 24 combinations of the following were tried:


| MS     | BS  | H1  | H2 |
|--------|-----|-----|----|
| 2000   | 100 | 64  | 16 |
| 4000   |     | 128 | 32 |
| 8000   |     |     | 64 |
| 16000  |     |     |    |


The results were clear: the larger MS values always performed better
than their smaller equivalents, with the best score being 95.92%
correct.  Interestingly, for a given MS value, larger H1 values tended
to perform better than smaller.  Previously, these two values had not
shown any particular correlation.  For a given MS and H1 pair, for
smaller MS values larger H2 values were better, while for large MS
values smaller H2 values were better.  Since the MS value is how long
the network is allowed to train, and since the longer a network has
had to train the longer 


Here is a graphical representation of the lengths of each vector/layer
for the most accurate network of the 24.  Each `x` represents ten
elements, rounded.  Since this is the MNIST dataset, the input layer
has 784 elements, and the output 10.

```
xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
xxxxxxxxxxxxx
xx
x
```

This may have something to do with it.  If nothing else, the graphic
gives a sense of scale to an often-abstracted problem.


## Impact of layer sizing

Finally, to better understand the impact of layer sizing on
reasonably-good MS values, a final set of values was tested:


| MS   | BS  | H1  | H2 |
|------|-----|-----|----|
| 4000 | 100 | 16  | 8  |
| 8000 |     | 32  | 16 |
|      |     | 64  | 32 |
|      |     | 128 | 64 |
|      |     | 256 |    |


All 40 combinations were tried.  The best combination was (8000, 100,
256, 16) with a score of 94.5%, followed by H2 = (32, 64, 8), in that
order.  The worst score used H1=16, H2=8.  Clearly, in this case, more
total neurons correlates with higher scores, although H1 was weighted
more heavily than H2.


# Summary

The maximum step size mattered most.

Batch size was barely relevant.

H1 and H2 had no significant, consistent correlations, although in
general larger values for H1 and smaller values for H2 worked better
in some cases.


# Prompt

For this assignment, explore MNIST training with a neural network with
2 hidden layers, as described here:

https://www.tensorflow.org/versions/r0.10/tutorials/mnist/tf/index.html

In the code: fully_connected_feed.py, note lines 36-40, which define
max_size, batch_step and number of neurons in hidden-layers 1 and 2.

Explore some of these parameters and report on your observations
(e.g. %correct vs training time).

Submit your report here (on Blackboard).

Please note: working collaboratively is acceptable at the conceptual
level and for learning about TensorFlow, but each individual should
submit their own unique exploration and report

