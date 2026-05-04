from omegaconf import OmegaConf
from data_simulation import MultiSpeakerSimulator
import os

def run():
    cfg = OmegaConf.load("config.yaml")
    
    print("Initializing MultiSpeakerSimulator")
    simulator = MultiSpeakerSimulator(cfg)
    
    print("Generating simulated audio and RTTM files.")
    simulator.generate_sessions()
    print(f"Simulation complete. Files located in: {cfg.data_simulator.outputs.output_dir}")

if __name__ == "__main__":
    run()