import jax
import numpy as np

import jax.random as jrandom
import jax.numpy as jnp
from functools import partial

def build_func_dict(obs, ts, ls2, y0, noise_sigma, func, product=True):
    #normal = lambda x, mu, s2: jnp.exp(jnp.sum(-0.5 * jnp.log(2*jnp.pi*s2) - 0.5 * jnp.square(x-mu)/s2))
    normal = lambda x, mu, s2: jnp.exp(jnp.sum(-0.5 * jnp.log(2*jnp.pi*s2) - 0.5 * jnp.square(x-mu)/s2 + 0.5 * jnp.log(2*jnp.pi*ls2)))
    lognormal = lambda x, mu, s2: jnp.sum(-0.5 * jnp.log(2*jnp.pi*s2) - 0.5 * jnp.square(x-mu)/s2)
    kernel = normal if product else normal0

    def bayes_posterior(params):
        ys = func(y0, ts, params)
        #logprior = jnp.sum(-0.5 * jnp.square(params) - 0.5 * jnp.log(2*jnp.pi))
        loglik = jnp.sum(-0.5 * jnp.square((ys - obs) / noise_sigma))# - 0.5 * jnp.log(2*jnp.pi) - jnp.log(noise_sigma))
        return loglik# + logprior
        
    def mmd_posterior(params):
        logprior = lognormal(params, 0.0, 1.0)
        s2 = jnp.square(noise_sigma)
        tj0 = func(y0, ts, params)
        
        def calculate_V(tj):
            kxy = jax.vmap(kernel, (0, 0, None))(obs, tj, s2 + ls2)
            return jnp.sum(kxy)
    
        def calculate_W(tj1, tj2):
            kxy = jax.vmap(kernel, (0, 0, None))(tj2, tj1, 2 * s2 + ls2)
            return jnp.sum(kxy)
            
        Vs = calculate_V(tj0)
        #Ws = calculate_W(tj0, tj0)
        return Vs# - 0.5 * Ws

    def particle_updates_marginalized(params_set, params_set2=None, sum=True):
        n_particles = params_set.shape[0]
        if params_set2 is None:
            params_set2 = params_set
        params_set2 = jax.lax.stop_gradient(params_set2)
        trajs1 = jax.vmap(func, (None, None, 0))(y0, ts, params_set)
        trajs2 = jax.vmap(func, (None, None, 0))(y0, ts, params_set2)
        
        s2 = jnp.square(noise_sigma)
        
        def calculate_V(tj):
            kxy = jax.vmap(kernel, (0, 0, None))(obs, tj, s2 + ls2)
            return jnp.sum(kxy)
            
        def calculate_W(tj1, tj2):
            kxy = jax.vmap(kernel, (0, 0, None))(tj2, tj1, 2 * s2 + ls2)
            return jnp.sum(kxy)
            
        Vs = jax.vmap(calculate_V)(trajs1)
        Ws = jax.vmap(jax.vmap(calculate_W, (None, 0)), (0, None))(trajs1, trajs2)
        Ws = Ws.at[jnp.diag_indices_from(Ws)].set(0.0)
        Ws = jnp.sum(Ws, 1) / (n_particles - 1)
        if sum:
            return jnp.sum(Vs - Ws)
        return Vs - Ws

    func_dict = {'bayes': jax.jit(bayes_posterior), 'mmd-bayes': jax.jit(mmd_posterior), \
                 'pcuq': jax.jit(partial(particle_updates_marginalized, sum=True))}
    
    return func_dict


def mala(func, n_iters, dt0, params_init, reg_const, key, multi=False):
    g0 = jax.value_and_grad(func, 0)
    gfunc = jax.vmap(g0, 0) if multi else jax.value_and_grad(func)
    
    params = params_init
    trace = []
    n_particles = params_init.shape[0] if multi else 1

    def logratio(pold, pnew, dt):
        lold, gold = g0(pold)
        lnew, gnew = g0(pnew)
        lold -= 0.5 * reg_const * jnp.sum(jnp.square(pold))
        lnew -= 0.5 * reg_const * jnp.sum(jnp.square(pnew))
        gold -= reg_const * pold
        gnew -= reg_const * pnew
        
        t1 = lnew - lold
        q1 = -0.25 * jnp.sum(jnp.square(pold - pnew - dt * gnew))
        q2 = -0.25 * jnp.sum(jnp.square(pnew - pold - dt * gold))
        return t1 / reg_const + (q1 - q2) / (reg_const * dt)

    logratio = jax.vmap(logratio, (0, 0, None)) if multi else logratio

    hist = []
    window = 10
    for i in range(n_iters):
        rng, key = jrandom.split(key)
        ll, g = gfunc(params)
        
        params_new = params + dt0 * g - dt0 * reg_const * params + jnp.sqrt(2 * reg_const * dt0) * jrandom.normal(rng, params.shape)
        rng, key = jrandom.split(key)

        lr = logratio(params, params_new, dt0)
        
        if multi:
            flag = jnp.where(jnp.log(jrandom.uniform(rng, lr.shape)) <= lr)[0]
            params = params.at[flag, :].set(params_new[flag, :])
            hist.append(flag.shape[0] / n_particles)
        else:
            flag = (jnp.log(jrandom.uniform(rng, lr.shape)) <= lr)
            params = params_new if flag else params
            hist.append(flag+0.0)
            
        trace.append(params.copy())
        
        hist = hist[-window:]
        if len(hist) >= window and jnp.mean(jnp.array(hist)) < 0.2 and (i+1)%window == 0 and i >= 50:
            dt0 = dt0 * 0.75#max(dt0*0.75, 1e-8)
            #print(dt0)
        elif len(hist) >= window and jnp.mean(jnp.array(hist)) > 0.8 and (i+1)%window == 0 and i >= 50:
            dt0 *= 1.5
            #print(dt0)

        if (i+1) % 1000 == 0:
            print('{}: {}'.format(i+1, jnp.mean(jnp.array(hist))))
            
    return trace

def langevin(func, n_iters, dt0, params_init, reg_const, key, multi=False):
    gfunc = jax.value_and_grad(func, 0)
    gfunc = jax.vmap(gfunc, 0) if multi else gfunc
    
    params = params_init
    trace = []
    
    for i in range(n_iters):
        rng, key = jrandom.split(key)
        ll, g = gfunc(params)
        
        params_new = params + dt0 * g - dt0 * reg_const * params + jnp.sqrt(2 * reg_const * dt0) * jrandom.normal(rng, params.shape)
        params = params_new
        
        trace.append(params.copy())
        
        if (i+1) % 1000 == 0:
            print('{}: {}'.format(i+1, ll))
            
    return trace