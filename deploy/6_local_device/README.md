# Type 6: Universal Local Device & On-Premise Deployment

Run TARA directly on your local workstation, PC, laptop, or edge hardware with 100% offline privacy and zero external cloud dependency.

## Supported Platforms
- Windows 10 / 11
- macOS (Apple Silicon M1/M2/M3/M4 & Intel)
- Linux (Ubuntu, Fedora, Arch, Debian)
- Edge Single Board Computers (Raspberry Pi 4/5, Jetson)

## Files in this Folder
- `start_tara.bat`: Windows 1-click startup batch script.
- `start_tara.sh`: Linux/macOS 1-click bash launcher script.

## How to Run
- **On Windows**: Double-click `start_tara.bat` (or run `deploy\6_local_device\start_tara.bat` in Command Prompt / PowerShell).
- **On Linux / macOS**: Run `bash deploy/6_local_device/start_tara.sh` in Terminal.

It will start the engine on `http://127.0.0.1:7860` and open your browser automatically.
Zero internet connection needed for inference.
