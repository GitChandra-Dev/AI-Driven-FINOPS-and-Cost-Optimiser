Puropose and achiements:

Cloud bills can spiral out of control overnight. A developer might leave a massive EC2 instance running, or a misconfigured Lambda might loop infinitely, resulting in thousands of dollars in NAT Gateway charges.

By building an "AI FinOps Optimizer", we achieve:

Daily Cost Profiling: AWS Cost Explorer API is queried every morning.
AI Spending Analysis: AWS Bedrock compares yesterday's spend to the 7-day average. It detects anomalies (e.g., "S3 bandwidth spiked by 400%").
Automated Recommendations: The AI generates specific, actionable recommendations to cut costs (e.g., "Add an S3 VPC Endpoint to reduce NAT charges").
Executive Reporting: A formatted FinOps report is automatically emailed to Engineering Managers via Amazon SNS.
ARCHITECTURE

Amazon EventBridge (Triggers daily at 8:00 AM) ↓ AWS Lambda (Python 3.11) ↓ (Queries AWS Cost Explorer API for previous 7 days) AWS Bedrock (Analyzes JSON cost data for spikes & optimizations) ↓ Amazon SNS (Sends beautifully formatted Email/Slack message)

================================ PART 1 — COST EXPLORER & SNS SETUP=======================================

Step 1 — Enable Cost Explorer AWS Console → Billing and Cost Management → Cost Explorer. (If already enabled, skip — no action needed.)

Step 2 — Create the SNS Topic Why: Simple Notification Service (SNS) allows us to securely broadcast our AI reports to multiple email addresses.

AWS Console -> SNS -> Topics -> Create topic.
Type: Standard. Name: Daily-FinOps-Reports.
Create Topic.
Click Create subscription -> Protocol: Email -> Endpoint: chandansahu1303@gmail.com.
Check your email and click the confirmation link.
SAVE the Topic ARN.
=================================== PART 3 — THE AI FINOPS LAMBDA====================================

Step 3 — Create the Lambda Function

AWS Console -> Lambda -> Create function: FinOps-AI-Analyzer.
Runtime: Python 3.11. Timeout: 1 minute.
IAM Role Attachments:
AWSBillingReadOnlyAccess (covers Cost Explorer)
AmazonBedrockFullAccess
AmazonSNSFullAccess
Environment Variables:
SNS_TOPIC_ARN = <your-topic-arn>


Step 4 — Write the Analysis Logic Why: The code must calculate date ranges, fetch granular cost data, and prompt the AI effectively.

Code lambda_function.py:

import json
import logging
import os
import re
from datetime import datetime, timedelta
from typing import Any, Dict, List, Tuple

import boto3
from botocore.exceptions import ClientError

# Setup structured logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

class FinOpsReportManager:
    """
    Orchestrates the collection of AWS cost data, AI-driven analysis 
    via Amazon Bedrock, and delivery via Amazon SNS.
    """
    
    def __init__(self, region: str = 'ap-south-1'):
        self.region = region
        # Clients initialized lazily or here; shared across the class instance
        self.ce_client = boto3.client('ce', region_name=self.region)
        self.bedrock_client = boto3.client('bedrock-runtime', region_name=self.region)
        self.sns_client = boto3.client('sns', region_name=self.region)
        self.topic_arn = os.environ.get('SNS_TOPIC_ARN')

    def _calculate_window(self, days: int = 7) -> Tuple[str, str]:
        """Calculates the start and end dates for the reporting period."""
        now = datetime.today()
        start = now - timedelta(days=days)
        return start.strftime('%Y-%m-%d'), now.strftime('%Y-%m-%d')

    def _fetch_raw_costs(self, start: str, end: str) -> List[Dict[str, Any]]:
        """Queries AWS Cost Explorer for daily unblended costs grouped by service."""
        try:
            response = self.ce_client.get_cost_and_usage(
                TimePeriod={'Start': start, 'End': end},
                Granularity='DAILY',
                Metrics=['UnblendedCost'],
                GroupBy=[{'Type': 'DIMENSION', 'Key': 'SERVICE'}]
            )
            return response.get('ResultsByTime', [])
        except ClientError as e:
            logger.error(f"Cost Explorer API failure: {e}")
            raise

    def _structure_cost_data(self, raw_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Transforms raw CE API response into a cleaned list of daily service spends.
        Filters out negligible costs (below 0.001).
        """
        processed_data = []
        for entry in raw_results:
            day_start = entry['TimePeriod']['Start']
            service_costs = {}
            
            for group in entry.get('Groups', []):
                service_name = group['Keys'][0]
                amount = float(group['Metrics']['UnblendedCost']['Amount'])
                
                if amount > 0.001:
                    service_costs[service_name] = round(amount, 4)
            
            processed_data.append({
                "Date": day_start, 
                "Services": service_costs
            })
        return processed_data

    def _generate_ai_analysis(self, costs: List[Dict[str, Any]], start: str, end: str) -> str:
        """
        Sends structured cost data to Bedrock and cleans the AI response.
        """
        prompt_template = (
            f"You are an Expert AWS FinOps Architect. Analyze the last 7 days of AWS spend:\n"
            f"{json.dumps(costs)}\n\n"
            f"Write a daily AWS Cost Report email with these sections:\n\n"
            f"DAILY AWS FINOPS REPORT\n"
            f"Period: {start} to {end}\n"
            f"============================================================\n\n"
            f"1. SPEND SUMMARY\n(Total spend, daily trend, highest spend day)\n\n"
            f"2. TOP SERVICES BY COST\n(List each service, its total cost, and % of total bill)\n\n"
            f"3. ANOMALIES DETECTED\n(Any unusual spikes or patterns)\n\n"
            f"4. TOP 3 RECOMMENDATIONS\n(Specific, actionable steps to reduce cost)\n\n"
            f"IMPORTANT FORMATTING RULES:\n"
            f"- Use PLAIN TEXT ONLY. No markdown, no **, no ##, no asterisks.\n"
            f"- Use ALL CAPS for section headers.\n"
            f"- Use dashes (-) for bullet points.\n"
            f"- Separate sections with a line of dashes (---\n"
            f"- Keep it concise, professional, and easy to read in an email."
        )

        try:
            payload = {
                "messages": [{"role": "user", "content": prompt_template}]
            }
            
            response = self.bedrock_client.invoke_model(
                modelId='openai.gpt-oss-safeguard-120b',
                body=json.dumps(payload),
                contentType='application/json'
            )
            
            body = json.loads(response['body'].read().decode())
            content = body['choices'][0]['message']['content']
            
            # Post-process: Remove reasoning tags and clean whitespace
            cleaned_report = re.sub(r'<reasoning>.*?</reasoning>', '', content, flags=re.DOTALL)
            return cleaned_report.strip().lstrip(': \n')

        except (ClientError, KeyError, json.JSONDecodeError) as e:
            logger.error(f"Bedrock AI analysis failed: {e}")
            return "Error generating AI report: Analysis service unavailable."

    def _publish_to_sns(self, message: str) -> None:
        """Publishes the final report string to the configured SNS topic."""
        if not self.topic_arn:
            raise ValueError("SNS_TOPIC_ARN environment variable is missing.")
            
        try:
            self.sns_client.publish(
                TopicArn=self.topic_arn,
                Subject="Daily AI FinOps Report",
                Message=message
            )
        except ClientError as e:
            logger.error(f"SNS Publication failed: {e}")
            raise

    def execute(self) -> str:
        """Main orchestration flow for report generation."""
        logger.info("Starting FinOps report generation process...")
        
        # Step 1: Time Window
        start_date, end_date = self._calculate_window()
        
        # Step 2: Data Acquisition & Transformation
        raw_data = self._fetch_raw_costs(start_date, end_date)
        refined_costs = self._structure_cost_data(raw_data)
        
        # Step 3: AI Analysis
        final_report = self._generate_ai_analysis(refined_costs, start_date, end_date)
        
        # Step 4: Delivery
        self._publish_to_sns(final_report)
        
        logger.info("FinOps report successfully delivered.")
        return "Report Sent"

def lambda_handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """
    AWS Lambda Entry Point.
    """
    try:
        manager = FinOpsReportManager()
        result = manager.execute()
        return {
            "statusCode": 200, 
            "body": result
        }
    except Exception as e:
        logger.exception(f"Critical failure in FinOps Pipeline: {str(e)}")
        return {
            "statusCode": 500, 
            "body": f"Internal Server Error: {str(e)}"
        }


========================PART 4 — EVENTBRIDGE SCHEDULING========================

Automate the execution Why: FinOps is only effective if it happens proactively every day.

AWS Console -> EventBridge -> Create rule.
Name: Daily-FinOps-Trigger.
Rule type: Schedule -> Cron expression.
Cron: 0 8 * * ? * (8:00 AM UTC daily).
Target: Lambda function FinOps-AI-Analyzer.
Click Create.
================================================PART 5 — TESTING =============================

Step 5 — Test the pipeline

Go to your Lambda function and click "Test".
Check your email inbox.
Observation: You will receive a beautifully formatted email: Subject: Daily AI FinOps Report Spend Trend: Your total spend increased by 15% yesterday, driven primarily by Amazon EC2. Anomalies: "EC2 - Other" costs (NAT Gateway Data Processing) spiked from $10/day to $45 yesterday. Recommendations:
The NAT Gateway spike suggests an internal resource is downloading large files from S3. Implement an S3 Gateway VPC Endpoint to route this traffic for free.
Ensure auto-scaling groups are scaling down during off-peak hours.

=================================================CLEANUP======================================================
Delete every resources so that it dont cost you anything in your free tier aws account 

Delete EventBridge Rule.
Delete Lambda.
Delete SNS Topic.
