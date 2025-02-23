import * as cdk from 'aws-cdk-lib';
import { Construct } from 'constructs';
import * as iam from 'aws-cdk-lib/aws-iam';
import * as ec2 from 'aws-cdk-lib/aws-ec2';
import * as elbv2 from 'aws-cdk-lib/aws-elasticloadbalancingv2';
import * as elbv2_tg from 'aws-cdk-lib/aws-elasticloadbalancingv2-targets'
import * as secretsmanager from 'aws-cdk-lib/aws-secretsmanager';
import * as s3 from 'aws-cdk-lib/aws-s3';
import * as cloudFront from 'aws-cdk-lib/aws-cloudfront';
import * as origins from 'aws-cdk-lib/aws-cloudfront-origins';
import * as apiGateway from 'aws-cdk-lib/aws-apigateway';
import * as lambda from "aws-cdk-lib/aws-lambda";
import * as opensearch from 'aws-cdk-lib/aws-opensearchservice';
import * as path from "path";
import * as sqs from 'aws-cdk-lib/aws-sqs';
import { SqsEventSource } from 'aws-cdk-lib/aws-lambda-event-sources';
import * as lambdaEventSources from 'aws-cdk-lib/aws-lambda-event-sources';

const projectName = `agentic-rag`; 
const region = process.env.CDK_DEFAULT_REGION;    
const accountId = process.env.CDK_DEFAULT_ACCOUNT;
const targetPort = 8080;
const bucketName = `storage-for-${projectName}-${accountId}-${region}`; 
const ec2RoleName = `role-ec2-for-${projectName}-${region}`
const stage = 'dev';
const s3_prefix = 'docs';

const opensearch_account = "admin";
const opensearch_passwd = "Wifi1234!";
let opensearch_url = "";

const titan_embedding_v2 = [  // dimension = 1024
  {
    "bedrock_region": "us-west-2", // Oregon
    "model_type": "titan",
    "model_id": "amazon.titan-embed-text-v2:0"
  },
  {
    "bedrock_region": "us-east-1", // N.Virginia
    "model_type": "titan",
    "model_id": "amazon.titan-embed-text-v2:0"
  },
  {
    "bedrock_region": "us-east-2", // Ohio
    "model_type": "titan",
    "model_id": "amazon.titan-embed-text-v2:0"
  }
];

const LLM_embedding = titan_embedding_v2;  //  titan_embedding_v2_single

const max_object_size = 102400000; // 100 MB max size of an object, 50MB(default)
const enableHybridSearch = 'true';
const supportedFormat = JSON.stringify(["pdf", "txt", "csv", "pptx", "ppt", "docx", "doc", "xlsx", "py", "js", "md", "jpeg", "jpg", "png"]);  
const enableParentDocumentRetrival = 'true';
const vectorIndexName = projectName

export class CdkAgenticRagStack extends cdk.Stack {
  constructor(scope: Construct, id: string, props?: cdk.StackProps) {
    super(scope, id, props);

    // s3 
    const s3Bucket = new s3.Bucket(this, `storage-${projectName}`,{
      bucketName: bucketName,
      blockPublicAccess: s3.BlockPublicAccess.BLOCK_ALL,
      removalPolicy: cdk.RemovalPolicy.DESTROY,
      autoDeleteObjects: true,
      publicReadAccess: false,
      versioned: false,
      cors: [
        {
          allowedHeaders: ['*'],
          allowedMethods: [
            s3.HttpMethods.POST,
            s3.HttpMethods.PUT,
          ],
          allowedOrigins: ['*'],
        },
      ],
    });
    new cdk.CfnOutput(this, 'bucketName', {
      value: s3Bucket.bucketName,
      description: 'The nmae of bucket',
    });

    // cloudfront for sharing s3
    const distribution_docs = new cloudFront.Distribution(this, `sharing-for-${projectName}`, {
      comment: `CloudFront-for-${projectName} (Document)`,
      defaultBehavior: {
        origin: origins.S3BucketOrigin.withOriginAccessControl(s3Bucket),
        allowedMethods: cloudFront.AllowedMethods.ALLOW_ALL,
        cachePolicy: cloudFront.CachePolicy.CACHING_DISABLED,
        viewerProtocolPolicy: cloudFront.ViewerProtocolPolicy.REDIRECT_TO_HTTPS,
      },
      priceClass: cloudFront.PriceClass.PRICE_CLASS_200,  
    });
    new cdk.CfnOutput(this, `distribution-sharing-DomainName-for-${projectName}`, {
      value: 'https://'+distribution_docs.domainName,
      description: 'The domain name of the Distribution Sharing',
    });

    // role
    const role = new iam.Role(this, `api-role-for-${projectName}`, {
      roleName: `api-role-for-${projectName}-${region}`,
      assumedBy: new iam.ServicePrincipal("apigateway.amazonaws.com")
    });
    role.addToPolicy(new iam.PolicyStatement({
      resources: ['*'],
      actions: [
        'lambda:InvokeFunction',
        'cloudwatch:*'
      ]
    }));
    role.addManagedPolicy({
      managedPolicyArn: 'arn:aws:iam::aws:policy/AWSLambdaExecute',
    }); 

    // opensearch
    // Permission for OpenSearch
    const domainName = projectName
    const accountId = process.env.CDK_DEFAULT_ACCOUNT;
    const resourceArn = `arn:aws:es:${region}:${accountId}:domain/${domainName}/*`
    
    const OpenSearchAccessPolicy = new iam.PolicyStatement({        
      resources: [resourceArn],      
      actions: ['es:*'],
      effect: iam.Effect.ALLOW,
      principals: [new iam.AnyPrincipal()],      
    });  

    const domain = new opensearch.Domain(this, 'Domain', {
      version: opensearch.EngineVersion.OPENSEARCH_2_3,
      
      domainName: domainName,
      removalPolicy: cdk.RemovalPolicy.DESTROY,
      enforceHttps: true,
      fineGrainedAccessControl: {
        masterUserName: opensearch_account,
        // masterUserPassword: cdk.SecretValue.secretsManager('opensearch-private-key'),
        masterUserPassword:cdk.SecretValue.unsafePlainText(opensearch_passwd)
      },
      capacity: {
        masterNodes: 3,
        masterNodeInstanceType: 'r6g.large.search',
        // multiAzWithStandbyEnabled: false,
        dataNodes: 3,
        dataNodeInstanceType: 'r6g.large.search',        
        // warmNodes: 2,
        // warmInstanceType: 'ultrawarm1.medium.search',
      },
      accessPolicies: [OpenSearchAccessPolicy],      
      ebs: {
        volumeSize: 100,
        volumeType: ec2.EbsDeviceVolumeType.GP3,
      },
      nodeToNodeEncryption: true,
      encryptionAtRest: {
        enabled: true,
      },
      zoneAwareness: {
        enabled: true,
        availabilityZoneCount: 3,        
      }
    });
    new cdk.CfnOutput(this, `Domain-of-OpenSearch-for-${projectName}`, {
      value: domain.domainArn,
      description: 'The arm of OpenSearch Domain',
    });
    new cdk.CfnOutput(this, `Endpoint-of-OpenSearch-for-${projectName}`, {
      value: 'https://'+domain.domainEndpoint,
      description: 'The endpoint of OpenSearch Domain',
    });
    opensearch_url = 'https://'+domain.domainEndpoint;

    const apiInvokePolicy = new iam.PolicyStatement({ 
      // resources: ['arn:aws:execute-api:*:*:*'],
      resources: ['*'],
      actions: [
        'execute-api:Invoke',
        'execute-api:ManageConnections'
      ],
    }); 
    
    // Lambda - chat (websocket)
    const roleLambdaDocument = new iam.Role(this, `role-lambda-chat-ws-for-${projectName}`, {
      roleName: `role-lambda-chat-ws-for-${projectName}-${region}`,
      assumedBy: new iam.CompositePrincipal(
        new iam.ServicePrincipal("lambda.amazonaws.com"),
        new iam.ServicePrincipal("bedrock.amazonaws.com"),
      )
    });
    roleLambdaDocument.addManagedPolicy({
      managedPolicyArn: 'arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole',
    });
    const BedrockPolicy = new iam.PolicyStatement({  // policy statement for sagemaker
      resources: ['*'],
      actions: ['bedrock:*'],
    });        
    roleLambdaDocument.attachInlinePolicy( // add bedrock policy
      new iam.Policy(this, `bedrock-policy-lambda-chat-ws-for-${projectName}`, {
        statements: [BedrockPolicy],
      }),
    );        
    const lambdaInvokePolicy = new iam.PolicyStatement({ 
      resources: ['*'],
      actions: [
        "lambda:InvokeFunction"
      ],
    });        
    roleLambdaDocument.attachInlinePolicy( 
      new iam.Policy(this, `lambda-invoke-policy-for-${projectName}`, {
        statements: [lambdaInvokePolicy],
      }),
    );  

    // S3 - Lambda(S3 event) - SQS(fifo) - Lambda(document)
    // DLQ
    let dlq:any[] = [];
    for(let i=0;i<LLM_embedding.length;i++) {
      dlq[i] = new sqs.Queue(this, 'DlqS3EventFifo'+i, {
        visibilityTimeout: cdk.Duration.seconds(600),
        queueName: `dlq-s3-event-for-${projectName}-${i}.fifo`,  
        fifo: true,
        contentBasedDeduplication: false,
        deliveryDelay: cdk.Duration.millis(0),
        retentionPeriod: cdk.Duration.days(14)
      });
    }

    // SQS for S3 event (fifo) 
    let queueUrl:string[] = [];
    let queue:any[] = [];
    for(let i=0;i<LLM_embedding.length;i++) {
      queue[i] = new sqs.Queue(this, 'QueueS3EventFifo'+i, {
        visibilityTimeout: cdk.Duration.seconds(600),
        queueName: `queue-s3-event-for-${projectName}-${i}.fifo`,  
        fifo: true,
        contentBasedDeduplication: false,
        deliveryDelay: cdk.Duration.millis(0),
        retentionPeriod: cdk.Duration.days(2),
        deadLetterQueue: {
          maxReceiveCount: 1,
          queue: dlq[i]
        }
      });
      queueUrl.push(queue[i].queueUrl);
    }
    
    // Lambda for s3 event manager
    const lambdaS3eventManager = new lambda.Function(this, `lambda-s3-event-manager-for-${projectName}`, {
      description: 'lambda for s3 event manager',
      functionName: `lambda-s3-event-manager-for-${projectName}`,
      handler: 'lambda_function.lambda_handler',
      runtime: lambda.Runtime.PYTHON_3_11,
      code: lambda.Code.fromAsset(path.join(__dirname, '../../lambda-s3-event-manager')),
      timeout: cdk.Duration.seconds(60),      
      environment: {
        sqsFifoUrl: JSON.stringify(queueUrl),
        nqueue: String(LLM_embedding.length)
      }
    });
    for(let i=0;i<LLM_embedding.length;i++) {
      queue[i].grantSendMessages(lambdaS3eventManager); // permision for SQS putItem
    }

    // Lambda for document manager
    let lambdDocumentManager:any[] = [];
    for(let i=0;i<LLM_embedding.length;i++) {
      lambdDocumentManager[i] = new lambda.DockerImageFunction(this, `lambda-document-manager-for-${projectName}-${i}`, {
        description: 'S3 document manager',
        functionName: `lambda-document-manager-for-${projectName}-${i}`,
        role: roleLambdaDocument,
        code: lambda.DockerImageCode.fromImageAsset(path.join(__dirname, '../../lambda-document-manager')),
        timeout: cdk.Duration.seconds(600),
        memorySize: 8192,
        environment: {
          s3_bucket: s3Bucket.bucketName,
          s3_prefix: s3_prefix,
          opensearch_account: opensearch_account,
          opensearch_passwd: opensearch_passwd,
          opensearch_url: opensearch_url,
          roleArn: roleLambdaDocument.roleArn,
          path: 'https://'+distribution_docs.domainName+'/', 
          sqsUrl: queueUrl[i],
          max_object_size: String(max_object_size),
          supportedFormat: supportedFormat,
          LLM_embedding: JSON.stringify(LLM_embedding),
          enableParentDocumentRetrival: enableParentDocumentRetrival,
          enableHybridSearch: enableHybridSearch,
          vectorIndexName: vectorIndexName
        }
      });         
      s3Bucket.grantReadWrite(lambdDocumentManager[i]); // permission for s3
      lambdDocumentManager[i].addEventSource(new SqsEventSource(queue[i])); // permission for SQS
    }
    
    // s3 event source
    const s3PutEventSource = new lambdaEventSources.S3EventSource(s3Bucket, {
      events: [
        s3.EventType.OBJECT_CREATED_PUT,
        s3.EventType.OBJECT_REMOVED_DELETE,
        s3.EventType.OBJECT_CREATED_COMPLETE_MULTIPART_UPLOAD
      ],
      filters: [
        { prefix: s3_prefix+'/' },
      ]
    });
    lambdaS3eventManager.addEventSource(s3PutEventSource); 
    
    // EC2 Role    
    const ec2Role = new iam.Role(this, `role-ec2-for-${projectName}`, {
      roleName: ec2RoleName,
      assumedBy: new iam.CompositePrincipal(
        new iam.ServicePrincipal("ec2.amazonaws.com"),
        new iam.ServicePrincipal("bedrock.amazonaws.com"),
      ),
      managedPolicies: [cdk.aws_iam.ManagedPolicy.fromAwsManagedPolicyName('CloudWatchAgentServerPolicy')] 
    });

    const secreatManagerPolicy = new iam.PolicyStatement({  
      resources: ['*'],
      actions: ['secretsmanager:GetSecretValue'],
    });       
    ec2Role.attachInlinePolicy( // for isengard
      new iam.Policy(this, `secret-manager-policy-ec2-for-${projectName}`, {
        statements: [secreatManagerPolicy],
      }),
    );  

    // Secret
    const weatherApiSecret = new secretsmanager.Secret(this, `weather-api-secret-for-${projectName}`, {
      description: 'secret for weather api key', // openweathermap
      removalPolicy: cdk.RemovalPolicy.DESTROY,
      secretName: `openweathermap-${projectName}`,
      secretObjectValue: {
        project_name: cdk.SecretValue.unsafePlainText(projectName),
        weather_api_key: cdk.SecretValue.unsafePlainText(''),
      },
    });
    weatherApiSecret.grantRead(ec2Role) 

    const langsmithApiSecret = new secretsmanager.Secret(this, `langsmith-secret-for-${projectName}`, {
      description: 'secret for lamgsmith api key', // langsmith
      removalPolicy: cdk.RemovalPolicy.DESTROY,
      secretName: `langsmithapikey-${projectName}`,
      secretObjectValue: {
        langchain_project: cdk.SecretValue.unsafePlainText(projectName),
        langsmith_api_key: cdk.SecretValue.unsafePlainText(''),
      }, 
    });
    langsmithApiSecret.grantRead(ec2Role) 

    const tavilyApiSecret = new secretsmanager.Secret(this, `tavily-secret-for-${projectName}`, {
      description: 'secret for tavily api key', // tavily
      removalPolicy: cdk.RemovalPolicy.DESTROY,
      secretName: `tavilyapikey-${projectName}`,
      secretObjectValue: {
        project_name: cdk.SecretValue.unsafePlainText(projectName),
        tavily_api_key: cdk.SecretValue.unsafePlainText(''),
      },
    });
    tavilyApiSecret.grantRead(ec2Role) 

    const codeInterpreterSecret = new secretsmanager.Secret(this, `code-interpreter-secret-for-${projectName}`, {
      description: 'secret for code interpreter api key', // code interpreter
      removalPolicy: cdk.RemovalPolicy.DESTROY,
      secretName: `code-interpreter-${projectName}`,
      secretObjectValue: {
        project_name: cdk.SecretValue.unsafePlainText(projectName),
        code_interpreter_api_key: cdk.SecretValue.unsafePlainText(''),
        code_interpreter_id: cdk.SecretValue.unsafePlainText(''),
      },
    });
    codeInterpreterSecret.grantRead(ec2Role) 
    
    const pvrePolicy = new iam.PolicyStatement({  
      resources: ['*'],
      actions: ['ssm:*', 'ssmmessages:*', 'ec2messages:*', 'tag:*'],
    });       
    ec2Role.attachInlinePolicy( // for isengard
      new iam.Policy(this, `pvre-policy-ec2-for-${projectName}`, {
        statements: [pvrePolicy],
      }),
    );  

    ec2Role.attachInlinePolicy( // add bedrock policy
      new iam.Policy(this, `bedrock-policy-ec2-for-${projectName}`, {
        statements: [BedrockPolicy],
      }),
    );     

    const ec2Policy = new iam.PolicyStatement({  
      resources: ['arn:aws:ec2:*:*:instance/*'],
      actions: ['ec2:*'],
    });
    ec2Role.attachInlinePolicy( // add bedrock policy
      new iam.Policy(this, `ec2-policy-for-${projectName}`, {
        statements: [ec2Policy],
      }),
    );
    
    // getRole
    const getRolePolicy = new iam.PolicyStatement({  
      resources: ['*'],      
      actions: ['iam:GetRole'],
    }); 
    ec2Role.attachInlinePolicy( 
      new iam.Policy(this, `getRole-policy-for-${projectName}`, {
        statements: [getRolePolicy],
      }),
    ); 

    // VPC
    const vpc = new ec2.Vpc(this, `vpc-for-${projectName}`, {
      vpcName: `vpc-for-${projectName}`,
      maxAzs: 2,
      ipAddresses: ec2.IpAddresses.cidr("10.24.0.0/16"),
      natGateways: 1,
      createInternetGateway: true,
      subnetConfiguration: [
        {
          cidrMask: 24,
          name: `public-subnet-for-${projectName}`,
          subnetType: ec2.SubnetType.PUBLIC
        }, 
        {
          cidrMask: 24,
          name: `private-subnet-for-${projectName}`,
          subnetType: ec2.SubnetType.PRIVATE_WITH_EGRESS
        }
      ]
    });  

    // S3 endpoint
    // const s3BucketAcessPoint = vpc.addGatewayEndpoint(`s3Endpoint-${projectName}`, {
    //   service: ec2.GatewayVpcEndpointAwsService.S3,
    // });

    // s3BucketAcessPoint.addToPolicy(
    //   new iam.PolicyStatement({
    //     principals: [new iam.AnyPrincipal()],
    //     actions: ['s3:*'],
    //     resources: ['*'],
    //   }),
    // ); 

    // Bedrock endpoint
    // const bedrockEndpoint = vpc.addInterfaceEndpoint(`bedrock-endpoint-${projectName}`, {
    //   privateDnsEnabled: true,
    //   service: new ec2.InterfaceVpcEndpointService(`com.amazonaws.${region}.bedrock-runtime`, 443)
    // });
    // bedrockEndpoint.connections.allowDefaultPortFrom(ec2.Peer.ipv4(vpc.vpcCidrBlock), `allowBedrockPortFrom-${projectName}`)

    // bedrockEndpoint.addToPolicy(
    //   new iam.PolicyStatement({
    //     principals: [new iam.AnyPrincipal()],
    //     actions: ['bedrock:*'],
    //     resources: ['*'],
    //   }),
    // );
    
    // Knowledge base endpoint
    // const knowledgeBaseEndpoint = vpc.addInterfaceEndpoint(`knowledge-base-endpoint-${projectName}`, {
    //   privateDnsEnabled: true,
    //   service: new ec2.InterfaceVpcEndpointService(`com.amazonaws.${region}.bedrock-agent`, 443)
    // });
    // bedrockEndpoint.connections.allowDefaultPortFrom(ec2.Peer.ipv4(vpc.vpcCidrBlock), `allowKnowledgeBasePortFrom-${projectName}`)

    // bedrockEndpoint.addToPolicy(
    //   new iam.PolicyStatement({
    //     principals: [new iam.AnyPrincipal()],
    //     actions: ['bedrock:*'],
    //     resources: ['*'],
    //   }),
    // );

    // EC2 Security Group
    const ec2Sg = new ec2.SecurityGroup(this, `ec2-sg-for-${projectName}`,
      {
        vpc: vpc,
        allowAllOutbound: true,
        description: "Security group for ec2",
        securityGroupName: `ec2-sg-for-${projectName}`,
      }
    );
    // ec2Sg.addIngressRule(  
    //   ec2.Peer.anyIpv4(),
    //   ec2.Port.tcp(22),
    //   'SSH',
    // );
    // ec2Sg.addIngressRule(
    //   ec2.Peer.anyIpv4(),
    //   ec2.Port.tcp(80),
    //   'HTTP',
    // );
    
    // ALB SG
    const albSg = new ec2.SecurityGroup(this, `alb-sg-for-${projectName}`, {
      vpc: vpc,
      allowAllOutbound: true,
      securityGroupName: `alb-sg-for-${projectName}`,
      description: 'security group for alb'
    });
    ec2Sg.connections.allowFrom(albSg, ec2.Port.tcp(targetPort), 'allow traffic from alb') // alb -> ec2

    // ALB
    const alb = new elbv2.ApplicationLoadBalancer(this, `alb-for-${projectName}`, {
      internetFacing: true,
      vpc: vpc,
      vpcSubnets: {
        subnets: vpc.publicSubnets
      },
      securityGroup: albSg,
      loadBalancerName: `alb-for-${projectName}`
    });
    alb.applyRemovalPolicy(cdk.RemovalPolicy.DESTROY); 

    new cdk.CfnOutput(this, `albUrl-for-${projectName}`, {
      value: `http://${alb.loadBalancerDnsName}/`,
      description: `albUrl-${projectName}`,
      exportName: `albUrl-${projectName}`
    }); 

    // CloudFront
    const CUSTOM_HEADER_NAME = "X-Custom-Header"
    const CUSTOM_HEADER_VALUE = `${projectName}_12dab15e4s31` // Temporary value
    const origin = new origins.LoadBalancerV2Origin(alb, {      
      httpPort: 80,
      customHeaders: {[CUSTOM_HEADER_NAME] : CUSTOM_HEADER_VALUE},
      originShieldEnabled: false,
      protocolPolicy: cloudFront.OriginProtocolPolicy.HTTP_ONLY      
    });
    const distribution = new cloudFront.Distribution(this, `cloudfront-for-${projectName}`, {
      comment: `CloudFront-for-${projectName}`,
      defaultBehavior: {
        origin: origin,
        viewerProtocolPolicy: cloudFront.ViewerProtocolPolicy.REDIRECT_TO_HTTPS,
        allowedMethods: cloudFront.AllowedMethods.ALLOW_ALL,
        cachePolicy: cloudFront.CachePolicy.CACHING_DISABLED,
        originRequestPolicy: cloudFront.OriginRequestPolicy.ALL_VIEWER        
      },
    /*  additionalBehaviors: {
        "/sharing": {
          origin: origins.S3BucketOrigin.withOriginAccessControl(s3Bucket),
          viewerProtocolPolicy: cloudFront.ViewerProtocolPolicy.REDIRECT_TO_HTTPS,
          allowedMethods: cloudFront.AllowedMethods.ALLOW_ALL,
          cachePolicy: cloudFront.CachePolicy.CACHING_DISABLED,
          originRequestPolicy: cloudFront.OriginRequestPolicy.ALL_VIEWER
        }
      }, */
      priceClass: cloudFront.PriceClass.PRICE_CLASS_200
    }); 
    new cdk.CfnOutput(this, `distributionDomainName-for-${projectName}`, {
      value: 'https://'+distribution.domainName,
      description: 'The domain name of the Distribution'
    });   
    
    const userData = ec2.UserData.forLinux();
    const environment = {
      "projectName": projectName,
      "accountId": accountId,
      "region": region,
      "s3_bucket": s3Bucket.bucketName,
      "sharing_url": 'https://'+distribution_docs.domainName,
      "opensearch_url": opensearch_url,
      "LLM_embedding": JSON.stringify(LLM_embedding),
      "opensearch_account": opensearch_account,
      "opensearch_passwd": opensearch_passwd,      
    }    
    new cdk.CfnOutput(this, `environment-for-${projectName}`, {
      value: JSON.stringify(environment),
      description: `environment-${projectName}`,
      exportName: `environment-${projectName}`
    });

    const commands = [
      'yum install git python-pip -y',
      'pip install pip --upgrade',            
      `sh -c "cat <<EOF > /etc/systemd/system/streamlit.service
[Unit]
Description=Streamlit
After=network-online.target

[Service]
User=ec2-user
Group=ec2-user
Restart=always
ExecStart=/home/ec2-user/.local/bin/streamlit run /home/ec2-user/${projectName}/application/app.py

[Install]
WantedBy=multi-user.target
EOF"`,
        `runuser -l ec2-user -c "mkdir -p /home/ec2-user/.streamlit"`,
        `runuser -l ec2-user -c 'cat <<EOF > /home/ec2-user/.streamlit/config.toml
[server]
port=${targetPort}
maxUploadSize = 100

[theme]
base="dark"
primaryColor="#fff700"
EOF'`,
      `json='${JSON.stringify(environment)}' && echo "$json">/home/config.json`,
      `runuser -l ec2-user -c 'cd && git clone https://github.com/kyopark2014/${projectName}'`,
      `runuser -l ec2-user -c 'pip install streamlit streamlit_chat beautifulsoup4 pytz tavily-python'`,        
      `runuser -l ec2-user -c 'pip install boto3 langchain_aws langchain langchain_community'`,
      `runuser -l ec2-user -c 'pip install langgraph opensearch-py PyPDF2 yfinance rizaio'`,      
      'systemctl enable streamlit.service',
      'systemctl start streamlit',
      `yum install -y amazon-cloudwatch-agent`,
      `mkdir /var/log/application/ && chown ec2-user /var/log/application && chgrp ec2-user /var/log/application`,
    ];    
    userData.addCommands(...commands);

    // EC2 instance
    const appInstance = new ec2.Instance(this, `app-for-${projectName}`, {
      instanceName: `app-for-${projectName}`,
      instanceType: new ec2.InstanceType('m5.xlarge'), // m5.large t2.small
      // instanceType: ec2.InstanceType.of(ec2.InstanceClass.T2, ec2.InstanceSize.SMALL),
      machineImage: new ec2.AmazonLinuxImage({
        generation: ec2.AmazonLinuxGeneration.AMAZON_LINUX_2023
      }),
      // machineImage: ec2.MachineImage.latestAmazonLinux2023(),
      vpc: vpc,
      vpcSubnets: {
        subnets: vpc.privateSubnets  
      },
      securityGroup: ec2Sg,
      role: ec2Role,
      userData: userData,
      blockDevices: [{
        deviceName: '/dev/xvda',
        volume: ec2.BlockDeviceVolume.ebs(8, {
          deleteOnTermination: true,
          encrypted: true,
        }),
      }],
      detailedMonitoring: true,
      instanceInitiatedShutdownBehavior: ec2.InstanceInitiatedShutdownBehavior.TERMINATE,
    }); 
    s3Bucket.grantReadWrite(appInstance);
    appInstance.applyRemovalPolicy(cdk.RemovalPolicy.DESTROY);

    // ALB Target
    const targets: elbv2_tg.InstanceTarget[] = new Array();
    targets.push(new elbv2_tg.InstanceTarget(appInstance)); 

    // ALB Listener
    const listener = alb.addListener(`HttpListener-for-${projectName}`, {   
      port: 80,
      open: true
    });     
    const targetGroup = listener.addTargets(`WebEc2Target-for-${projectName}`, {
      targetGroupName: `TG-for-${projectName}`,
      targets: targets,
      protocol: elbv2.ApplicationProtocol.HTTP,
      port: targetPort,
      conditions: [elbv2.ListenerCondition.httpHeader(CUSTOM_HEADER_NAME, [CUSTOM_HEADER_VALUE])],
      priority: 10      
    });
    listener.addTargetGroups("demoTargetGroupInt", {
      targetGroups: [targetGroup]
    })
    const defaultAction = elbv2.ListenerAction.fixedResponse(403, {
        contentType: "text/plain",
        messageBody: 'Access denied',
    })
    listener.addAction(`RedirectHttpListener-for-${projectName}`, {
      action: defaultAction
    });
  }
}