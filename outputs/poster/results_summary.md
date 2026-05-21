# Mie RL Poster Results

Dataset: 700 analytical Mie simulations; 7 materials x 100 radii; wavelengths 300-900 nm in 10 nm steps; homogeneous spheres in air.
RL setup: dueling Double DQN, 9D normalized state, 5 actions (radius down/up, material previous/next, stop), scattering-anchored global peaks, weighted error = 0.80 peak wavelength + 0.10 scattering amplitude + 0.10 absorption amplitude.

| Query | DQN solved | DQN mean error | DQN best error | DQN best design | Brute-force best error | Brute-force design |
| --- | ---: | ---: | ---: | --- | ---: | --- |
| 648 nm target | 89.5% | 0.0748 | 0.0450 | Cu, r=374.7 nm | 0.0239 | Cu, r=205.2 nm |
| 700 nm target | 44.5% | 0.1341 | 0.0140 | Si, r=223.7 nm | 0.0140 | Si, r=223.7 nm |
