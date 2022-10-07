NOTE: this may be wrong, saving for reference but something is fishy

If we're going to approximate maybe we just ought to use
https://threeplusone.com/pubs/on_logistic_normal.pdf

Note that we only need gradients after all

# Approximation for integral

We seek to approximate

$$I(a,b) = \int_{-\infty}^{\infty} \log (1+e^{-a x + b}) \frac{1}{\sqrt{2 \pi}} e^{-\frac{1}{2}x^2}$$

To do so, we first take the second
derivative of $\log(1+e^{-a x + b})$ with respect to $b$. This gives us the PDF of a logistic distribution with location $b/a$ and scale $1/a$.

We approximate that logistic distribution with a normal distribution with mean $b/a$ and standard deviation $\frac{\pi}{a\sqrt{3}}$.

The integral product is then easily
computed. We integrate back twice
with respect to $b$, and find the
constant by making a series expansion
around $a = 0$ and $b = 0$

This gives us the approximation


$$I(a,b) \simeq b - \sqrt{\frac{\pi}{6}} + e^{-\frac{3 b^2}{2 (3 a^2 + \pi^2)}}\sqrt{\frac{3a^2+\pi^2}{6 \pi}}-\frac{1}{2}\mathrm{erfc}\left(\sqrt{\frac{3 b /2}{3 a^2 + \pi^2}}\right)$$


Useful for variational Bayes method.

The gradients have a simpler form:

$$\frac{\partial I(a,b)}{\partial a} = a \frac{e^{-\frac{3 b^2}{2(3 a^2 + \pi^2)}}}{\sqrt{(3a^2+\pi^2)\frac{2\pi}{3}}}$$

and

$$\frac{\partial I(a,b)}{\partial b} = 
\frac{1}{2}\left(1+\textrm{erf}\left(b\middle/\sqrt{\frac{2}{3}(3a^2+\pi^2)}\right)\right)$$

But overall it's better to just
have a lookup table.
