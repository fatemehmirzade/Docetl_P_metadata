import subprocess

COMMANDS = [
    ["python", "prepare_data.py", "text_files_Papers_Ian"],
    ["docetl", "run", "01_biological_info.yaml"],
    ["docetl", "run", "02_ms_instruments.yaml"],
    ["docetl", "run", "03_sample_prep.yaml"],
    ["docetl", "run", "04_separation.yaml"],
    ["docetl", "run", "05_data_analysis.yaml"],
    ["docetl", "run", "06_clinical_experimental.yaml"],
    ["python", "merge_and_generate_ann.py"],
]


def run_pipeline():
    """
    run_pipeline()
    Function Name: run_pipeline
    """
    for cmd in COMMANDS:
        command_str = " ".join(cmd)
        print(f"\n Running: {command_str}")
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as exc:
            print(f" Failed at: {command_str}")
            print(exc)
            break
    else:
        print("\n Pipeline finished successfully!")


if __name__ == "__main__":
    run_pipeline()
