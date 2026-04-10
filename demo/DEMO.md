Neurobotics OSS 'Droid' Demo

What you are seeing below in the demo video* is the Neurobotics OSS 'Droid' Repository cloned from GitHub and setup inside a VSCode IDE workspace. The source code window at the top contains the CMAC Cerebellum Node python code under debug.

This is a full simulation, including computing the values of the chemical neurotransmitters including Dopamine and Serotonin. Here is an example of that output embedded in the output of the right terminal pane:

​

'neuromod': {'DA': 0.5, 'NE': 0.2, '5HT': 0.5, 'ACh': 0.5}

You can also see what region of the body is getting a simulated 'poke' and how hard that poke is:

{'zone_id': 'arm_R_lower', 'pressure_N': 0.83, 'is_nociceptive': False}

​

The middle pane shows the cortical activation at the top level, ready for associative processing and the higher functional areas of the brain.

​

The three terminal windows below that are running the following telemetry and neuro-probe tooling scripts, all of which we have released with our OSS drop under AGPL-3.0.

​

The terminal windows are running, from right to left:

​

Right terminal: Running the telemetry script for the ZMQ implementation of the Spinal Cord tracts.

​

Middle terminal: A running real-time log of what the Somatosensory Cortex S1 ( Brodmann Areas 1, 2, 3 ) has received.

​

Left terminal: The equivalent of a neural probe inserted into the Thalamus and simulating the ascending spinocortical pathway relay. When the Return key is hit to start the Thalamic Stimulator Probe, the scrolling in the other terminals is showing the output of the cortical areas involved.

​

There you have it. A complete demonstration of one of the ascending tracts as implemented in the Bubo architecture and freely available to the research and academic communities. Commercial uses require strict licensing, see documentation at GitHub for more details and complete source code.

https://github.com/neurobotics-sci/neurobotics-biological-substrate
