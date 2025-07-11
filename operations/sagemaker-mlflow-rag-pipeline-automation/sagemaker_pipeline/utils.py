from __future__ import absolute_import
import ast
import importlib

def get_pipeline_driver(module_name, passed_args=None):
    """Gets the driver for generating your pipeline definition.

    Pipeline modules must define a get_pipeline() module-level method.

    Args:
        module_name: The module name of your pipeline.
        passed_args: Optional passed arguments that your pipeline may be templated by.

    Returns:
        The SageMaker Workflow pipeline.
    """
    try:
        module = importlib.import_module(module_name)
    except ImportError as e:
        raise ImportError(f"Failed to import module {module_name}: {e}")
    
    if not hasattr(module, 'get_pipeline'):
        raise AttributeError(f"Module {module_name} does not have get_pipeline function")
    
    kwargs = convert_struct(passed_args)
    return module.get_pipeline(**kwargs)


def convert_struct(str_struct=None):
    """Convert the string argument to its proper type

    Args:
        str_struct (str, optional): string to be evaluated. Defaults to None.

    Returns:
        string struct as its actual evaluated type
    """
    if not str_struct:
        return {}
    
    try:
        return ast.literal_eval(str_struct)
    except (SyntaxError, ValueError) as e:
        raise ValueError(f"Error parsing argument string: {str_struct}. Error: {e}")


def get_pipeline_custom_tags(module_name, args, tags):
    """Gets the custom tags for pipeline

    Args:
        module_name: The module name of your pipeline.
        args: Arguments passed to the pipeline.
        tags: Existing tags.

    Returns:
        Custom tags to be added to the pipeline
    """
    tags = tags or []
    
    # Add module name as a tag
    tags.append({"Key": "sagemaker:pipeline-module", "Value": module_name})
    
    # Try to get custom tags from the module
    try:
        module = importlib.import_module(module_name)
        if hasattr(module, 'get_pipeline_custom_tags'):
            kwargs = convert_struct(args)
            # region = kwargs.get('region', 'unknown')
            project_arn = kwargs.get('sagemaker_project_arn', None)
            return module.get_pipeline_custom_tags(tags, project_arn)
    except Exception as e:
        print(f"Error getting custom tags: {e}")
    
    return tags