from __future__ import absolute_import
import argparse
import json
import logging
import sys

from utils import get_pipeline_driver

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():  # pragma: no cover
    """The main harness that gets the pipeline definition JSON.

    Prints the json to stdout or saves to file.
    """
    parser = argparse.ArgumentParser("Gets the pipeline definition for the pipeline script.")

    parser.add_argument(
        "-n",
        "--module-name",
        dest="module_name",
        type=str,
        help="The module name of the pipeline to import.",
    )
    parser.add_argument(
        "-f",
        "--file-name",
        dest="file_name",
        type=str,
        default=None,
        help="The file to output the pipeline definition json to.",
    )
    parser.add_argument(
        "-kwargs",
        "--kwargs",
        dest="kwargs",
        default=None,
        help="Dict string of keyword arguments for the pipeline generation (if supported)",
    )
    parser.add_argument(
        "--pretty",
        dest="pretty",
        action="store_true",
        help="Pretty print the JSON output",
    )
    args = parser.parse_args()

    if args.module_name is None:
        parser.print_help()
        sys.exit(2)

    try:
        pipeline = get_pipeline_driver(args.module_name, args.kwargs)
        definition = pipeline.definition()
        
        # Pretty print if requested
        if args.pretty:
            definition = json.dumps(json.loads(definition), indent=2, sort_keys=True)
        
        if args.file_name:
            with open(args.file_name, "w") as f:
                f.write(definition)
            logger.info(f"Pipeline definition written to {args.file_name}")
        else:
            print(definition)
    except Exception as e:
        logger.error(f"Exception: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()