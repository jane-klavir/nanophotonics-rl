from nano_mie.config import MieDatasetConfig
from nano_mie.dataset import generate_dataset, save_dataset


def main() -> None:
    config = MieDatasetConfig()

    dataset = generate_dataset(config)
    save_dataset(dataset, config.output_path)

    print(f"Saved dataset to: {config.output_path}")
    print(f"X shape: {dataset['X'].shape}")
    print(f"Y_qabs shape: {dataset['Y_qabs'].shape}")


if __name__ == "__main__":
    main()