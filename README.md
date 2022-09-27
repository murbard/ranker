# ranker

Score items using an ELO point system, by presenting pairs of items
to the user which are expected to decrease the entropy of the posterior
distribution over the score.

One approach uses HMC sampling, the other is a variational Bayes
approach where we ignore the covariance structure. The intuition
in the latter is that, at the beginning where scores are loose,
there isn't much to do aside from asking the user to rank a bunch
of things, and when scores are better known it becomes about
finding items close to each other with large distributions.

It would be worth considering comparing this fancy stuff to the heuristic:
"pick the two items with the most distribution overlap and compare them"
