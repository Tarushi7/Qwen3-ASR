from omegaconf import OmegaConf
from data_simulation import MultiSpeakerSimulator

CONFIG_PATH = "/home/tarushi/tarushi_folder/speaker_diarization/config.yaml"

print("Loading config...")
cfg = OmegaConf.load(CONFIG_PATH)

print("Creating simulator...")
simulator = MultiSpeakerSimulator(cfg)

print("Starting generation...")
simulator.generate_sessions()

print("Done!")