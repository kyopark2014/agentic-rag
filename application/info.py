nova_pro_models = [   # Nova Pro
    {   
        "bedrock_region": "us-west-2", # Oregon
        "model_type": "nova",
        "model_id": "us.amazon.nova-pro-v1:0"
    },
    {
        "bedrock_region": "us-east-1", # N.Virginia
        "model_type": "nova",
        "model_id": "us.amazon.nova-pro-v1:0"
    },
    {
        "bedrock_region": "us-east-2", # Ohio
        "model_type": "nova",
        "model_id": "us.amazon.nova-pro-v1:0"
    }
]

nova_lite_models = [   # Nova Pro
    {   
        "bedrock_region": "us-west-2", # Oregon
        "model_type": "nova",
        "model_id": "us.amazon.nova-lite-v1:0"
    },
    {
        "bedrock_region": "us-east-1", # N.Virginia
        "model_type": "nova",
        "model_id": "us.amazon.nova-lite-v1:0"
    },
    {
        "bedrock_region": "us-east-2", # Ohio
        "model_type": "nova",
        "model_id": "us.amazon.nova-lite-v1:0"
    }
]

claude_sonnet_3_5_v1_models = [   # Sonnet 3.5 V1
    {
        "bedrock_region": "us-west-2", # Oregon
        "model_type": "claude",
        "model_id": "anthropic.claude-3-5-sonnet-20240620-v1:0"
    },
    {
        "bedrock_region": "us-east-1", # N.Virginia
        "model_type": "claude",
        "model_id": "anthropic.claude-3-5-sonnet-20240620-v1:0"
    },
    {
        "bedrock_region": "eu-central-1", # Frankfurt
        "model_type": "claude",
        "model_id": "anthropic.claude-3-5-sonnet-20240620-v1:0"
    },
    {
        "bedrock_region": "ap-northeast-1", # Tokyo
        "model_type": "claude",
        "model_id": "anthropic.claude-3-5-sonnet-20240620-v1:0"
    }
]

claude_sonnet_3_5_v2_models = [   # Sonnet 3.5 V2
    {
        "bedrock_region": "us-west-2", # Oregon
        "model_type": "claude",
        "model_id": "anthropic.claude-3-5-sonnet-20241022-v2:0"
    },
    {
        "bedrock_region": "us-east-1", # N.Virginia
        "model_type": "claude",
        "model_id": "us.anthropic.claude-3-5-sonnet-20241022-v2:0"
    },
    {
        "bedrock_region": "us-east-2", # Ohio
        "model_type": "claude",
        "model_id": "us.anthropic.claude-3-5-sonnet-20241022-v2:0"
    }
]

claude_haiku_3_5_models = [   # Haiku 3.5 
    {
        "bedrock_region": "us-west-2", # Oregon
        "model_type": "claude",
        "model_id": "anthropic.claude-3-5-haiku-20241022-v1:0"
    },
    {
        "bedrock_region": "us-east-1", # N.Virginia
        "model_type": "claude",
        "model_id": "us.anthropic.claude-3-5-haiku-20241022-v1:0"
    },
    {
        "bedrock_region": "us-east-2", # Ohio
        "model_type": "claude",
        "model_id": "us.anthropic.claude-3-5-haiku-20241022-v1:0"
    }
]

def get_model_info(model_name):
    parallel_processing_models = []

    if model_name == "Nova Pro":
        parallel_processing_models = nova_pro_models
    elif model_name == "Nova Lite":
        parallel_processing_models = nova_lite_models
    elif model_name == "Claude Sonnet 3.0":
        parallel_processing_models = claude_sonnet_3_5_v1_models
    elif model_name == "Claude Sonnet 3.5":
        parallel_processing_models = claude_sonnet_3_5_v2_models
    elif model_name == "Claude Haiku 3.5":
        parallel_processing_models = claude_haiku_3_5_models

    return parallel_processing_models

STOP_SEQUENCE_CLAUDE = "\n\nHuman:" 
STOP_SEQUENCE_NOVA = '"\n\n<thinking>", "\n<thinking>", " <thinking>"'

def get_stop_sequence(model_name):
    models = get_model_info(model_name)

    model_type = models[0]["model_type"]

    if model_type == "claude":
        return STOP_SEQUENCE_CLAUDE
    elif model_type == "nova":
        return STOP_SEQUENCE_NOVA
    else:
        return ""
