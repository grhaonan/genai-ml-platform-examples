from __future__ import absolute_import
import argparse
import json
import logging
import sys

from utils import get_pipeline_driver, convert_struct, get_pipeline_custom_tags

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():  # pragma: no cover
    """The main harness that creates or updates and runs the pipeline.

    Creates or updates the pipeline and runs it.
    """
    parser = argparse.ArgumentParser("Creates or updates and runs the pipeline for the pipeline script.")

    parser.add_argument(
        "-n",
        "--module-name",
        dest="module_name",
        type=str,
        help="The module name of the pipeline to import.",
    )
    parser.add_argument(
        "-kwargs",
        "--kwargs",
        dest="kwargs",
        default=None,
        help="Dict string of keyword arguments for the pipeline generation (if supported)",
    )
    parser.add_argument(
        "-role-arn",
        "--role-arn",
        dest="role_arn",
        type=str,
        help="The role arn for the pipeline service execution role.",
    )
    parser.add_argument(
        "-description",
        "--description",
        dest="description",
        type=str,
        default=None,
        help="The description of the pipeline.",
    )
    parser.add_argument(
        "-tags",
        "--tags",
        dest="tags",
        default=None,
        help="""List of dict strings of '[{"Key": "string", "Value": "string"}, ..]'""",
    )
    parser.add_argument(
        "-p",
        "--parameters",
        dest="parameters",
        default=None,
        help="Dict string of parameters to override pipeline parameters",
    )
    parser.add_argument(
        "-no-wait",
        "--no-wait",
        dest="no_wait",
        action="store_true",
        help="Don't wait for the pipeline execution to finish",
    )
    args = parser.parse_args()

    if args.module_name is None or args.role_arn is None:
        parser.print_help()
        sys.exit(2)
    
    tags = convert_struct(args.tags)
    parameters = convert_struct(args.parameters)

    try:
        # Get the pipeline
        pipeline = get_pipeline_driver(args.module_name, args.kwargs)
        logger.info("###### Creating/updating a SageMaker Pipeline with the following definition:")
        parsed = json.loads(pipeline.definition())
        logger.info(json.dumps(parsed, indent=2, sort_keys=True))

        # Add custom tags
        all_tags = get_pipeline_custom_tags(args.module_name, args.kwargs, tags)
 
        # Upsert the pipeline
        upsert_response = pipeline.upsert(
            role_arn=args.role_arn, description=args.description, tags=all_tags)
        logger.info("\n###### Created/Updated SageMaker Pipeline: Response received:")
        logger.info(upsert_response)

        # Start the pipeline execution with parameters
        execution = pipeline.start(parameters=parameters)
        logger.info(f"\n###### Execution started with PipelineExecutionArn: {execution.arn}")


        # # Extract execution ID from ARN
        # execution_id = execution.arn.split("/")[-1]
        
        # # Add execution ID to parameters if not already present
        # if parameters is None:
        #     parameters = {}
        # parameters["SageMakerPipelineExecutionId"] = execution_id


        # Wait for execution to complete if requested
        if not args.no_wait:
            logger.info("Waiting for the execution to finish...")
            
            # Use custom waiter config
            execution.wait(delay = 60, max_attempts = 200)
            
            logger.info("\n#####Execution completed. Execution step details:")
            logger.info(execution.list_steps())

        
    except Exception as e:
        logger.error(f"Exception: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
