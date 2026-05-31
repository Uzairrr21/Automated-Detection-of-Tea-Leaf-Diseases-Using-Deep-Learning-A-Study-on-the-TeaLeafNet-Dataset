import subprocess, sys
 
subprocess.run([
    sys.executable, "-m", "pip", "install", "-q",
    "torch==2.2.0", "torchvision==0.17.0",
    "--index-url", "https://download.pytorch.org/whl/cu118",
    "--upgrade"
], check=True)
 
subprocess.run([
    sys.executable, "-m", "pip", "install", "-q",
    "timm==0.9.16", "grad-cam==1.5.0", "scikit-learn"
], check=True)
 
import warnings
warnings.filterwarnings("ignore")
 
print("✅ Installation complete — RESTART THE KERNEL NOW, then run Cell 2+")