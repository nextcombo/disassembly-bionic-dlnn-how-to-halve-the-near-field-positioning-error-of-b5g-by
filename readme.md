# Disassembly | Bionic DLNN: How to Halve the Near-Field Positioning Error of B5G by Halving? With Core Code and Simulation Results [Python Code Included]

Disassembly | Biomimetic DLNN: How to Halve the Near-Field Positioning Error by More Than Half? With Core Code and Simulation Results [ISAC Communication Perception Integrated Special Section] Friends who work on near-field positioning are likely to encounter this contradiction: as the antenna array gets larger, it should be more accurate in positioning. However, when running actual simulations, the error actually becomes much larger than expected.
The problem lies not in the hardware but in your understanding of "near-field."
From using 64 antennas to 256 and 511 antennas, the Rayleigh distance is pushed to kilometers, and all targets fall within the near-field region. Still relying on the assumption of plane wave for building channel models and designing positioning algorithms based on far-field, the error naturally cannot be reduced.
This article has organized all the pitfalls I have encountered in this direction: from why near-field is difficult, to how the biomimetic tree dendritic liquid neural network (DLNN) can reduce errors, with each step providing core code snippets and simulation results. After reading, you will at least avoid 80% of futile attempts.
I. Problem Background: The More Antennas You Have, The Harder It Is to Position
Let's calculate a simple calculation first. The Rayleigh distance is defined as:
[d_R = \frac{2D^2}{\lambda}]
Where (D) is the array aperture, and (\lambda) is the wavelength. The simulation configuration in this work is an uniform linear array (ULA) with 511 antennas, operating at 28 GHz, and half-wavelength spacing:
Array aperture: (D = 510 * \lambda/2 = 255\lambda)
Rayleigh distance: (d_R = 2 * (255\lambda)^2 / \lambda ≈ 1392) meters
This means that targets within 1392 meters are all in the near-field region. And our positioning target distance is only 8 to 35 meters, which is definitely deep near-field.
The core change in near-field is that the wavefront changes from plane wave to spherical wave:
Under the far-field assumption, the arrival angles (AoA) of different antennas are the same, and the spatial features only contain angle information;  
In the near-field spherical wave scenario, each antenna sees a different degree of wavefront curvature, coupled with distance and angle. It is necessary to jointly estimate in the two-dimensional space of distance-angle, and the traditional angle spectrum alone is not sufficient.  
This is the fundamental difficulty in near-field positioning: the far-field model directly fails, and the accuracy of traditional direction of arrival estimation algorithms drops dramatically.  
II. Industry Pain Points: What happens when we move classic algorithms to near-field scenarios? It's simple—it constructs spatial spectra based on the plane wave assumption for the steering vectors, systematically introducing model mismatch errors in near-field. We have compared under the same configuration (see section below), and MUSIC's positioning error rapidly deteriorates under low transmitter power, and every round of positioning requires exhaustive search over the entire grid, resulting in significant computational overhead.  
Looking at deep learning approaches. A common practice is to feed the observation vector directly into CNN/RNN/Transformer:  
CNN: Proficient in extracting spatial local features, but insensitive to temporal dynamic information, while the observation itself evolves over time steps in near-field positioning;  
Standard RNN/LSTM: Capable of modeling temporal dynamics, but with a "hardcoded" memory mechanism, lacking generalization ability against different signal-to-noise ratios and distances;  
Transformer: Effective but with large parameters, high inference latency, and resources that are insufficient for communication nodes with limited capabilities.  
In summary, the pain points are either too inaccurate or too heavy, lacking a "both accurate and lightweight" solution.  
III. Core Concept: Why is it "Dendrite + Fluid"? This concept comes from two biological/brain-like inspirations:
1. Biomimetic dendritic structure (Dendritic Structure). Biological neurons are not simple single pathways, but through multiple dendritic branches receiving signals from different sources in parallel and then weighting and integrating at the soma. Inspiration by this, DLNN allows each neuron to have multiple parallel dendritic branches, combined with a dynamic gate mechanism - equivalent to assigning an "attention weight" to each dendrite, allowing the network to decide which branch to trust at different moments. This greatly enhances the model's ability to capture complex spatiotemporal features.
2. Leakage integral of liquid neural networks (Liquid Neural Network) leakage integral. The state of a liquid is not instantaneous but accumulates over time: the current state is the "leakage retention" of the previous state plus the "integral update" of new inputs. This leakage-based integral mechanism endows the network with robust temporal memory, naturally adaptive to different signal-to-noise ratios and dynamic environments.
IV. Dissecting the algorithm architecture: The core code of DLNN is very simple in the entire network forward process, about 60 lines in PyTorch implementation. Below is the complete core fragment with comments (the complete project can be found in the end of the article):
class DLNN(nn.Module):  def __init__(self, input_size=511, liquid_size=23, output_size=2,         num_dendrites=4, leaky_rate_init=0.3,         dendrite_act='relu', gate_act='sigmoid',         dropout_rate=0.1, use_layernorm=True):    super().__init__()    self.liquid_size  = liquid_size    self.num_dendrites = num_dendrites    # Learnable leak rate (through sigmoid Constraints 0~1 Between)    self.leaky_rate = nn.Parameter(torch.tensor(leaky_rate_init))    self.dend_act = getattr(torch, dendrite_act)  # Dendrite activation    self.gate_act = getattr(torch, gate_act)    # Door-controlled activation    # Input branch / Recursive branch / Door-controlled weight    self.W_in   = nn.Parameter(torch.randn(liquid_size, num_dendrites, input_size))    self.W_liquid = nn.Parameter(torch.randn(liquid_size, num_dendrites, liquid_size))    self.W_gate  = nn.Parameter(torch.randn(liquid_size, num_dendrites))    self.b_liquid = nn.Parameter(torch.zeros(liquid_size, 1))    self.W_out  = nn.Parameter(torch.randn(output_size, liquid_size))    self.b_out  = nn.Parameter(torch.zeros(output_size, 1))    self.ln  = nn.LayerNorm([liquid_size, 1])    self.dropout = nn.Dropout(dropout_rate)    # Parameter initialization    nn.init.xavier_uniform_(self.W_in, gain=nn.init.calculate_gain(dendrite_act))    nn.init.xavier_uniform_(self.W_liquid, gain=nn.init.calculate_gain(dendrite_act))    nn.init.xavier_uniform_(self.W_out, gain=1.0)  def forward(self, x):    # x: (batch, seq_len, input_dim)    batch, seq_len, _ = x.size()    liquid_state = torch.zeros(batch, self.liquid_size, 1, device=x.device)    for t in range(seq_len):     # Iterative along time steps      x_t = x[:, t, :]      # 1) Input branch: Each dendrite processes input in parallel      in_br = self.dend_act(torch.einsum("ldi,bi->bld", self.W_in, x_t))      # 2) Recursive branch: Processing of dendrites from the previous moment's liquid state      prev = liquid_state.squeeze(-1)      rec_br = self.dend_act(torch.einsum("ldh,bh->bld", self.W_liquid, prev))      # 3) Dynamic Gate Control: Assign weights to each dendrite and weighted aggregation      gate = self.gate_act(self.W_gate).unsqueeze(0)      merged_in = (in_br * gate).sum(-1, keepdim=True)      merged_rec = (rec_br * gate).sum(-1, keepdim=True)      # 4) Liquid input + Normalization + Randomly discard      liquid_in = merged_in + merged_rec + self.b_liquid.unsqueeze(0)      liquid_in = self.ln(liquid_in)      liquid_in = self.dropout(liquid_in)      # 5) Leakage integral: state = (1-α)·Old state + α·tanh(New input)      alpha = torch.sigmoid(self.leaky_rate)      liquid_state = (1 - alpha) * liquid_state + alpha * torch.tanh(liquid_in)    out = torch.matmul(self.W_out, liquid_state) + self.b_out    return out.squeeze(-1)\nSeveral design points worth noting (also the key points I learned from my mistakes).):   
1  
The gate is "each dendrite a weight" (W_gate: [liquid_size, num_dendrites]), rather than a global scalar - this ensures that different neurons can learn different fusion strategies.
2  
The leakage rate is a learnable parameter, not a fixed constant, and after sigmoid constraints, it is decided by the network itself to "remember how much and forget how much."
3  
Time-step iteration structure: The number of time steps in the input sequence determines how many times the state update is iterated, which is precisely the fundamental reason why LNN can model sequential data.
4  
LayerNorm + Dropout: Liquid layer computations can easily accumulate numerical instability under long sequences, and adding normalization significantly stabilizes convergence.
V. Data and Simulation Settings: A Summary of Simulation Configurations (Directly Reproducible):
Parameters
Value
Number of antennas (M)
511 (ULA, half-wavelength spacing)
Frequency (f)
28 GHz  
Location area
Distance: 8 to 35 meters, angle (\pi/4) to (3\pi/4).
Channel model
Near-field LoS channel + free space path loss
Training loss
HuberLoss  
Optimizer
Adam (lr=0.001, weight_decay=1e-5)  
Learning rate scheduling
ExponentialLR (gamma=0.98)  
Early cessation
patience=10  
Key points in data generation: Each element of the near-field LoS channel is a spherical wave phase
h = np.exp(-1j * (2 * np.pi / (3e8 / f)) * np.linalg.norm(ULA_BS.element_coordinate.T - target_coordinate, ord=2, axis=-1))
Note: Here, the use is of the true Euclidean distance from each antenna to the target (spherical wave), rather than the projected distance under far-field approximation. Then, perform an angular domain transformation on the channel matrix, taking the projection modulus as the observation vector fed into the network—this step essentially "expands" the near-field spatial features into a format that the network can digest.
6. Simulation Results Comparison
Under conditions of 40 dBm (10 W) and 10 dBm (10 mW), the positioning accuracy (RMSE, in meters) is compared with existing deep models:
Method
40 dBm RMSE
10 dBm RMSE
Latency
DLNN (This paper) 0.1706 0.2642 0.0580 s Other deep models
0.2836
0.2986
Several key conclusions:
1.
The advantage is significant under high signal-to-noise ratio: The error drops from 0.2836 m to 0.1706 m when using 40 dBm, a reduction of about 40%;
2.
It is more stable under low signal-to-noise ratio: When other models have errors approaching 0.3 m, the DLNN still maintains an error of 0.2642 m—thanks to the time-series memory of the liquid layer's adaptive suppression of noise;
3.
The latency is only 0.058 seconds: With small parameter quantity and lightweight structure, it is very suitable for real-time positioning and deployment in resource-constrained communication nodes.
In summary: DLNN achieves better performance in terms of "accuracy, speed, energy consumption" across three dimensions.
1.
Near-field channels must be modeled as spherical waves, and datasets generated by approximating "pseudo-near-field" with plane wave far-field cannot train models that work directly in real near-field scenarios;
2.
Do not skip angle domain transformation: Directly feeding raw antenna domain observations into the network leads to severe feature coupling and slow convergence; doing angle domain transformation before projection significantly improves results;
3.
Don't randomly adjust initial values for leakage rate: Too large an initial value can lead the network to favor "transient" behavior and lose temporal memory; between 0.2 and 0.4 is a relatively stable range;
4.
Add normalization after the liquid layer: As the state norm in time steps gradually expands during iterations, without adding LayerNorm, the loss tends to oscillate later on;
5.
Bidirectional Data Augmentation: By reversing the observation sequence and stacking it with the original observations, a dual-channel input is created equivalent to allowing the model to simultaneously see both positive and negative temporal features. This enhances both precision and robustness.

References:
[1] Q. Zhu, Y. Wang, T. Ohtsuki, S. Mao, D. Niyato, and G. Gui, "Bio-Inspired Dendritic Liquid Neural Network for Robust Near-Field Positioning in B5G/6G," IEEE Wireless Communications Letters, Early Access, Oct. 2025, doi: 10.1109/LWC.2025.3627310.
[2] R. Hasani, M. Lechner, A. Amini, D. Rus, and R. Grosu, "Liquid Time-constant Networks," in Proc. AAAI, 2021. (Original work on liquid neural networks)
Next Upcoming Post: [ISAC Column] Dissecting Spherical Wave Beamforming in Near-Field Communication: From Phase Compensation to Focus Gain, How Much Benefit Can the Same Approach Transfer to the Communications Side? Stay tuned.
📰 Historical Topics: ISAC Column | Drone Signal Detection | RIS Anti-Interference | MIMO-ISAR Imaging.

## Images

![Disassembly | Biomimetic DLNN: How to Halve the Near-Field Positioning Error of B5G by More Than Half? With Core Code and Simulation Results [Python Code Included]](01.jpg)
![Disassembly | Biomimetic DLNN: How to Halve the Near-Field Positioning Error of B5G by More Than Half? With Core Code and Simulation Results [Python Code Included]](02.jpg)
![Disassembly | Biomimetic DLNN: How to Halve the Near-Field Positioning Error of B5G by More Than Half? With Core Code and Simulation Results [Python Code Included]](03.jpg)
![Disassembly | Biomimetic DLNN: How to Halve the Near-Field Positioning Error of B5G by More Than Half? With Core Code and Simulation Results [Python Code Included]](04.jpg)

<!-- PAYMENT_BLOCK -->

## Get this dataset

- Payment link (Stripe): [Buy now](https://buy.stripe.com/3cs8yP7sY87d0vu9AB)
- Payment QR code: ![Payment QR code](pay.png)

<!-- /PAYMENT_BLOCK -->
