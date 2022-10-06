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