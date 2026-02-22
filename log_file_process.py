import sys
from pathlib import Path

def remove_2025_lines(input_path, output_path=None):
    """
    Remove all lines starting with '2025' or that are blank from a single text/log file.

    :param input_path:  Path to the input file (e.g. 'app.log').
    :param output_path: Path to save the filtered file. If None, overwrites input_path.
    """
    input_path = Path(input_path)
    if output_path is None:
        output_path = input_path
    else:
        output_path = Path(output_path)

    # Read & filter
    with input_path.open('r', encoding='utf-8') as f:
        filtered = [
            line
            for line in f
            # drop lines that start with '2025'
            if not line.startswith('2025')
            # drop lines that are blank or only whitespace
            and line.strip() != ''
        ]

    # Write back
    with output_path.open('w', encoding='utf-8') as f:
        f.writelines(filtered)


def batch_remove_2025_from_logs(directory, in_place=True):
    """
    Remove lines starting with '2025' or blank lines from every .log file in a directory.

    :param directory:   Path to search for '*.log' files.
    :param in_place:    If True, overwrite each .log file.
                        If False, write to a sibling file named '<original>.filtered.log'.
    """
    directory = Path(directory)
    for log_file in directory.glob('*.log'):
        if in_place:
            remove_2025_lines(log_file)
        else:
            filtered_path = log_file.with_name(log_file.stem + '.filtered.log')
            remove_2025_lines(log_file, filtered_path)


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Strip '2025' and blank lines from log files")
    p.add_argument('path', help="File or directory to process")
    p.add_argument('--batch', action='store_true',
                   help="Treat 'path' as a directory and process all .log files in it")
    p.add_argument('--no-inplace', action='store_true',
                   help="When batching, write to '*.filtered.log' instead of overwriting")
    args = p.parse_args()

    if args.batch:
        batch_remove_2025_from_logs(args.path, in_place=not args.no_inplace)
    else:
        remove_2025_lines(args.path)
