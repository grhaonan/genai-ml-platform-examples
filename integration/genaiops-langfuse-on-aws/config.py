MODEL_CONFIG = {
    "nova_pro": {
        "model_id": "us.amazon.nova-pro-v1:0",
        "inferenceConfig": {"maxTokens": 4096, "temperature": 0},
    },
    "nova_lite": {
        "model_id": "us.amazon.nova-lite-v1:0",
        "inferenceConfig": {"maxTokens": 2048, "temperature": 0},
    },
    "nova_micro": {
        "model_id": "us.amazon.nova-micro-v1:0",
        "inferenceConfig": {"maxTokens": 2048, "temperature": 0},
    },
}


GUARDRAIL_CONFIG = {
    "guardrailIdentifier": "<guardrailid>", # TODO: Fill the value using "GuardrailId" from the Event Outputs
    "guardrailVersion": "1",
    "trace": "enabled",
}


def get_aws_region(default="us-west-2"):
    """Return the AWS region the workshop environment runs in.

    Order: AWS_REGION / AWS_DEFAULT_REGION env vars, then the boto3 config, then the
    EC2 instance metadata via botocore (the VSCode instance runs in the workshop's
    deployment region), then ``default``. Jupyter kernels started by code-server do not
    source ~/.bashrc, so the metadata lookup is what makes this work inside the IDE.
    """
    import os

    region = os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION")
    if region:
        return region
    try:
        import boto3

        region = boto3.session.Session().region_name
        if region:
            return region
    except Exception:
        pass
    try:
        from botocore.utils import InstanceMetadataRegionFetcher

        region = InstanceMetadataRegionFetcher(timeout=1, num_attempts=1).retrieve_region()
        if region:
            return region
    except Exception:
        pass
    return default


AWS_REGION = get_aws_region()
