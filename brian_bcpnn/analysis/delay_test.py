from brian2 import *

eqs = """
dV/dt = 1/second : 1
"""
model = NeuronGroup(2, model=eqs, threshold='V>=20', reset='V=0')


synapse = Synapses(source=model, target=model, on_pre='V_post+=5')

synapse.connect('i!=j')

synapse.delay = 'rand()*ms'

print(synapse.delay)