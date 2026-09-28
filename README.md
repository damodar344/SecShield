# SecShield

Official implementation and experimental resources for:

**SecShield: A Privacy-Preserved Federated Learning Model to Detect Zero-Day Malware Attack**

Damodar Dhital, Sabir Ahmed Khan, Almustapha A. Wakili, Saugat Guni, and Woosub Jung

International Conference on Computer Communications and Networks (ICCCN), 2026.

📄 DOI: https://doi.org/10.1109/ICCCN69946.2026.11662719

## Citation

If you use SecShield in your research, please cite:

```bibtex
@inproceedings{dhital2026secshield,
  title={SecShield: A Privacy-Preserved Federated Learning Model to Detect Zero-Day Malware Attack},
  author={Dhital, Damodar and Khan, Sabir Ahmed and Wakili, Almustapha A. and Guni, Saugat and Jung, Woosub},
  booktitle={2026 International Conference on Computer Communications and Networks (ICCCN)},
  year={2026},
  publisher={IEEE},
  doi={10.1109/ICCCN69946.2026.11662719}
}

## Dataset

SecShield uses power-consumption traces from the publicly available
IoT Malware Data dataset.

Dataset:
https://www.kaggle.com/datasets/sa05042/iot-malware-data

The implementation expects the processed data files:

- `benignData.npy`
- `attackData.npy`

Place both files in the same directory as `SecShield.py` before running
the experiments.

## Installation

Clone the repository and install the required dependencies:

```bash
pip install -r requirements.txt
