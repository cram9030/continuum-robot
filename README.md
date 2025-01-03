# continuum-robot
Simulation for exploring the control and path planning of continuum robots


# Development Setup

## Quick Start
1. Clone repository
2. Navigate to the repository `cd continuum-robot`
3. make setup.sh executable if it isn't already `chmod +x /environment/scripts/setup.sh `
3. Run `./environment/scripts/setup.sh `
4. Activate environment: `conda activate continuum_robot`

## Docker Usage
For deployment testing:
```bash
chmod u+x environment/scripts/docker_build.sh
./environment/scripts/docker_build.sh
docker run continuum_robot

or

docker run -it continuum_robot /bin/bash
