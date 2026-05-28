import jax

from discodj import DiscoDJ


def disco_sim(res=256, res_pm=None, n_steps=100, cuda_scatter_gather=False):
    """
    This is a jitted DISCO-DJ run, where everything is done within a jitted function
    """
    res_pm = res_pm or res
    a_ini = 0.02

    dj = DiscoDJ(dim=3, res=res, boxsize=100.0)
    dj = dj.with_timetables()

    white_noise_field = dj.get_ngenic_noise(123)
    pkstate = dj.with_linear_ps()
    ics = dj.with_ics(pkstate, white_noise_field)
    lpt_state = dj.with_lpt(ics, n_order=1)
    sim_ini = dj.with_lpt_ics(lpt_state, n_order=1, a_ini=a_ini)

    X, P, a = dj.run_nbody(sim_ini, a_end=1.0, n_steps=n_steps, res_pm=res_pm, stepper="bullfrog",
                           cuda_scatter_gather=cuda_scatter_gather)

    return dj, X, P, a


disco_sim.jit = jax.jit(disco_sim, static_argnames=("res", "res_pm", "n_steps", "cuda_scatter_gather"))

if __name__ == '__main__':
    # Run only the supported (no-CUDA-scatter-gather) path to avoid
    # triggering the library assertion in `acc.py`.
    out_no_cuda = disco_sim.jit(cuda_scatter_gather=False)
    print("Finished no-CUDA run")
