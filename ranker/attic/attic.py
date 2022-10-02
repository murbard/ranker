# dead code

## Using scipy optimize

   class Memoizer():

        def __init__(self, vbayes):
            self.gh = None
            self.vbayes = vbayes
            self.n = vbayes.n

        def adjust_params(self, params):
            params[self.n:] = np.exp(params[self.n:])
            return

        def adjust_gh(self, gh, params):
            self.gh.h[self.n:,self.n:] *= params[self.n:].reshape(-1,1).dot(params[self.n:].reshape(1,-1))
            self.gh.g[self.n:] *= params[self.n:]

        def f(self, params):
            self.adjust_params(params)
            if np.any(params != self.vbayes.params) or self.gh is None:
                self.vbayes.params = params
                self.gh = self.vbayes.gradient(obs)
                self.adjust_gh(self.gh, params)

            return self.gh.val

        def grad(self, params):
            self.adjust_params(params)
            if np.any(params != self.vbayes.params):
                self.vbayes.params = params
                self.gh = self.vbayes.gradient(obs)
                self.adjust_gh(self.gh, params)
            return self.gh.g

        def hess(self, params):
            self.adjust_params(params)
            if np.any(params != self.vbayes.params):
                self.vbayes.params = params
                self.gh = self.vbayes.gradient(obs)
                self.adjust_gh(self.gh, params)
            return self.gh.h

    memo = Memoizer(v)
    params = v.params.copy()
    params[v.n:] = np.log(params[v.n:])
    minimize(memo.f, params, jac=memo.grad, hess=memo.hess, method='trust-ncg', options={'maxiter': 100000, 'disp': True})



# test gradients
#
#

    # ε = 1e-8
    # np.core.arrayprint._line_width = 200

    # max_so_far = 0.0
    # for p,func in enumerate([lambda v, _: v.__gradient_invgamma_entropy__(),
    # lambda  v, _: v.__gradient_normal_entropy__(),
    # lambda  v, _: v.__gradient_gamma_cross_entropy__(),
    # lambda  v, _:  v.__gradient_normal_cross_entropy__(),
    # lambda  v, obs:  v.__gradient_observations__(obs)]):
    #     gh = func(v, obs)
    #     for i in range(0, 2 * v.n + 2):
    #         v.params[i] += ε
    #         gh2 = func(v, obs)
    #         v.params[i] -= ε
    #         mx = np.max(np.abs(((gh2.g - gh.g) / ε)[i:] -  gh.h[i,i:]))
    #         if mx > max_so_far:
    #             max_so_far = mx
    #             best = (p, i, np.argmax(np.abs(((gh2.g - gh.g) / ε)[i:] -  gh.h[i,i:])))
    #             if mx > 1e-6:
    #                 print('plouf')

    #         print(i, ((gh2.g - gh.g) / ε)[i:] -  gh.h[i,i:],'\n\n')
    # print(max_so_far, best)
    #
    ## BEGIN TEST
            # gh1 = self.__gradient_observations__(obs)
            # ε = np.random.randn(2 * self.n + 2) * 1e-5
            # ε[1:] = 0.0 # let's see if mu works

            # self.params += ε
            # gh2 = self.__gradient_observations__(obs)
            # self.params -= ε

            # first_order = gh1.g.dot(ε)
            # second_order = ε.dot(gh1.h - 0.5 * np.diag(np.diag(gh1.h))).dot(ε)
            # actual = gh2.val - gh1.val
            # if np.abs(actual) > 1e-20 and np.abs((actual - first_order)) > 1e-20:
            #     error = np.abs(actual - first_order) / np.abs(actual)
            #     print(f"first order relative error : {error}")
            #     second_order_error = np.abs((actual - first_order) - second_order) / np.abs(actual - first_order)
            #     print(f"second order relative error : {second_order_error}")

            # END TEST


            # Newton step