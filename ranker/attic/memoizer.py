
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
    #minimize(memo.f, params, jac=memo.grad, hess=memo.hess, method='trust-ncg', options={'maxiter': 100000, 'disp': True})