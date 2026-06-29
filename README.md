# SBALF OSS Edition

This represents the released codebase for the Bubo Mark XVI Hardware version.

Current Neurobotics LLC Research & Development work is completely withheld,
except for a couple of hints for the astute reader *wink*.

## Included

* Tier 1 Reflex Systems
* Tier 2 Homeostatic Systems
* Tier 3 Limbic Systems      - Multimodal Long Term Memory IP Held
* Tier 4 Sensory Integration - Edge Sensory IP Held
* Tier 5 Motor Execution

## Excluded

* Tier 6 Planning
* Tier 7 Executive Cognition
* VSCC Proprietary Extensions # Vector Symbolic Cognitive Coprocessor - FPGA/RISC-V

## License

AGPLv3

Commercial licensing available separately.

Getting Started:

Read about neuroscience *grin*

git clone https://github.com/neurobotics-sci/neurobotics-biological-substrate.git

Edit ansible/inventory/hosts.ini according to your deployment topology and node/arch variants.
At the very least change 192.168.10.112 to match your machine. It might work just that easily :)

ansible -i ansible/inventory/hosts.ini all -m ping                           # Check topology
ansible-playbook -i ansible/inventory/hosts.ini ansible/site_hardware.yml -C # Check mode
ansible-playbook -i ansible/inventory/hosts.ini ansible/site_hardware.yml    # For real

Here is an example topology ping check followed by a dry run:

(base) [kenneth@n150-bubo neurobotics-biological-substrate-PULL_TEST]$ ansible -i ansible/inventory/hosts.ini all -m ping

hippocampus | SUCCESS =>
    changed: false
    ping: pong
amygdala | SUCCESS =>
    changed: false
    ping: pong
somatosensory | SUCCESS =>
    changed: false
    ping: pong
m1 | SUCCESS =>
    changed: false
    ping: pong
cerebellum | SUCCESS =>
    changed: false
    ping: pong
basal-ganglia | SUCCESS =>
    changed: false
    ping: pong
cingulate | SUCCESS =>
    changed: false
    ping: pong
circadian_clock | SUCCESS =>
    changed: false
    ping: pong
dopaminergic_sys | SUCCESS =>
    changed: false
    ping: pong
ans_autonomic | SUCCESS =>
    changed: false
    ping: pong
vagus_nerve | SUCCESS =>
    changed: false
    ping: pong
hypothalamus | SUCCESS =>
    changed: false
    ping: pong
insula | SUCCESS =>
    changed: false
    ping: pong
thalamus-l | SUCCESS =>
    changed: false
    ping: pong
thalamus-r | SUCCESS =>
    changed: false
    ping: pong
dist-thalami | SUCCESS =>
    changed: false
    ping: pong
reticular | SUCCESS =>
    changed: false
    ping: pong

(base) [kenneth@n150-bubo neurobotics-biological-substrate-PULL_TEST]$ ansible-playbook -i ansible/inventory/hosts.ini ansible/site_hardware.yml -C|grep -E "TASK|PLAY"

PLAY [Brain — Bootstrap all hardware nodes] ************************************
TASK [Gathering Facts] *********************************************************
TASK [common : Update apt cache (Bypass Python 3.6 trap)] **********************
TASK [common : Install system packages (Bypass Python 3.6 trap)] ***************
TASK [common : Create brain user] **********************************************
TASK [common : Create directory structure] *************************************
TASK [common : Push SBALF Codebase from Perforce Workspace to Edge Nodes] ******
TASK [common : Fix SBALF Codebase Permissions (Post-Rsync)] ********************
TASK [common : Ensure brain user owns the pushed SBALF codebase] ***************
TASK [common : Create Python virtualenv] ***************************************
TASK [common : Upgrade core Python build tools] ********************************
TASK [common : Install Python dependencies (Architecture Specific)] ************
TASK [common : Install Brain package (editable)] *******************************
TASK [common : Write node identity] ********************************************
TASK [common : Write BRAIN_PROFILE to environment] *****************************
TASK [common : Generate self-signed TLS certificate] ***************************
TASK [common : 🗣️ Deploy TTS Models (Speech/Hearing Tier)] **********************
PLAY [Brain — Homeostatic tier (hypothalamus, thalami, insula)] ****************
TASK [Gathering Facts] *********************************************************
TASK [brain_node : Install node-specific Python packages] **********************
TASK [brain_node : Generate cluster config for this node] **********************
TASK [brain_node : Create systemd service for brain node] **********************
TASK [brain_node : Enable and start brain node] ********************************
TASK [brain_node : Wait for node heartbeat] ************************************
PLAY [Brain — Limbic tier (amygdala, hippocampus)] *****************************
TASK [Gathering Facts] *********************************************************
TASK [brain_node : Install node-specific Python packages] **********************
TASK [brain_node : Generate cluster config for this node] **********************
TASK [brain_node : Create systemd service for brain node] **********************
TASK [brain_node : Enable and start brain node] ********************************
TASK [brain_node : Wait for node heartbeat] ************************************
PLAY [Brain — Motor tier (cerebellum, BG, M1, premotor)] ***********************
TASK [Gathering Facts] *********************************************************
TASK [brain_node : Install node-specific Python packages] **********************
TASK [brain_node : Generate cluster config for this node] **********************
TASK [brain_node : Create systemd service for brain node] **********************
TASK [brain_node : Enable and start brain node] ********************************
TASK [brain_node : Wait for node heartbeat] ************************************
PLAY [Brain — Cortical tier (PFC-L, PFC-R)] ************************************
TASK [Gathering Facts] *********************************************************
TASK [brain_node : Install node-specific Python packages] **********************
TASK [brain_node : Generate cluster config for this node] **********************
TASK [brain_node : Create systemd service for brain node] **********************
TASK [brain_node : Enable and start brain node] ********************************
TASK [brain_node : Wait for node heartbeat] ************************************
PLAY [Brain — Spinal tier (arms, legs)] ****************************************
PLAY [Brain — Verify deployment] ***********************************************
TASK [Gathering Facts] *********************************************************
TASK [Check Python and ZMQ] ****************************************************
TASK [Deployment summary] ******************************************************
PLAY RECAP *********************************************************************

Check the deployed modules under /etc/systemd/system/brain-*.service via systemctl and debug the python
launch commands on your systems. I'd suggest running the command in the service file directly on the
command line, debug your issues, then move them under systemd management. Good luck! :)
