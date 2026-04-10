from argparse import ArgumentParser

from mscthesis.config.declaration import ProjectConfig
from mscthesis.core.io import load_dataframe
from mscthesis.core.plotting.scanning import plot_scanning_results
from mscthesis.utilities.ids import validate_sample_id
from mscthesis.utilities.paths import ProjectPaths


def main() -> int:
    parser = ArgumentParser()
    parser.add_argument(
        "sample_id", type=str, help="sample id to regenerate scanning plots for"
    )
    args = parser.parse_args()
    sample_id = validate_sample_id(args.sample_id, required_digits=5)

    config = ProjectConfig()
    paths = ProjectPaths(config.behavior.storage_root)
    df = load_dataframe(paths.sample(sample_id).scanning().scan)
    plot_scanning_results(df, paths.sample(sample_id).scanning().plots)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
