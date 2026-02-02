'''prometheus.'''

# use single float32 precision by default.
import jax
jax.config.update('jax_enable_x64', False)
